import os
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from .detection import get_detector, get_textbox, get_textboxes
from .utils import group_text_box
from .recognition import get_recognizer
from .craft import CRAFT
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

detector_path = "/home/luoxuan/.EasyOCR/model/craft_mlt_25k.pth"
recognizer_path = "/home/luoxuan/.EasyOCR//model/chinese_sim.pth"
device = "cuda"

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def vis_model(fc_out=10):
    model = models.resnet50(pretrained=True)
    model.fc = HiddeFC(2048, fc_out)
    return model


def get_det_model(detector="craft", detector_params=None):
    if detector == "craft":
        net = CRAFT()
        net.load_state_dict(copyStateDict(torch.load(detector_path, map_location=device)))
        net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = False
    else:
        net = None
    return net


def get_rec_model():
    pass


class HiddeFC(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self, idim=2048, odim=10):
        super(HiddeFC, self).__init__()
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.fc(x)
        return out


class FinalLayer(nn.Module):
    """modified last layer for resnet50 for our dataset"""
    def __init__(self):
        super(FinalLayer, self).__init__()
        self.fc = nn.Linear(2048, 12)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out

def modified_resnet50():
    # load pretrained resnet50 with a modified last fully connected layer
    model = models.resnet50(pretrained = True)
    model.fc = FinalLayer()
    # uncomment following lines if you wnat to freeze early layers
    # i = 0
    # for child in model.children():
    #     i += 1
    #     if i < 4:
    #         for param in child.parameters():
    #             param.requires_grad = False
    return model


class JointVisDet(nn.Module):
    def __init__(self, idim=13, odim=2):
        super(JointVisDet, self).__init__()
        self.vis_model = vis_model()
        self.head = torch.nn.Linear(idim, odim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, bfts):
        vis_out = self.vis_model(img)   # (b, 1000)
        # extract bboxes features
        #box_fts = extract_bboxes_feature(bboxes)  # [b, 3]
        jot_out = torch.cat((vis_out, bfts), 1)
        out = self.head(jot_out)
        out = self.sigmoid(out)
        return out



class JointVisDetREC(nn.Module):
    def __init__(self, tidim=1029, todim=1029, vodim=10, odim=5):
        """

        :param idim:
        :param odim: protest + sign + 5 classification
        """
        super(JointVisDetREC, self).__init__()
        self.vis_model = vis_model(fc_out=vodim)
        #self.txt_fc = torch.nn.Linear(tidim, todim)
        self.head = torch.nn.Linear(todim + vodim, odim)
        self.softmax = F.softmax

    def forward(self, img, txt_enc):
        """
        :param img:   torch ([b, c, h, w])
        :param bboxes: torch([b, x1, y1, x2, y2, x3, y3, x4, y4])
        :param rec: torch([b, t1, t2, ...])
        :param confi: torch([b, 1])
        :return:
        """
        vis_out = self.vis_model(img)   # (b, 1000)
        #t_out = self.txt_fc(txt_enc)
        jot_out = torch.cat((vis_out, txt_enc), 1)
        out = self.head(jot_out)
        out = self.softmax(out, dim=1)
        return out



def extract_bboxes_feature(bboxes):
    """
    bboxes : torch([bboxex_num, 4])
    :param bboxes:
    :return:
    """
    num_bbox = bboxes.size(0)
    b_width_sum = torch.tensor()
    b_heigth_sum = torch.tensor()
    for i, b in enumerate(bboxes):
        b_width = torch.abs(b[0] - b[2])
        b_heigth = torch.abs(b[3] - b[5])
        b_width_sum += b_width
        b_heigth_sum += b_heigth
    return torch.tensor([num_bbox, b_width_sum / torch.min(torch.tensor(num_bbox, 1)),
                         b_heigth_sum / torch.min(torch.tensor(num_bbox, 1))])




class JointVisDet_bk(nn.Module):
    def __init__(self, idim=1003, odim=2, detector="craft", detector_params=None):
        super(JointVisDet_bk, self).__init__()
        self.vis_model = vis_model()
        #self.det_model = get_detector(detector_path, "gpu")

        # Only used as inference
        self.det_model = get_det_model(detector="craft", detector_params=None)

        self.head = torch.nn.Linear(idim, odim)
    def forward(self, img):
        vis_out = self.vis_model(img)
        #det_out = self.det_model(img)

        # post process of text score and link score


        text_box = get_textboxes(self.det_model, img, canvas_size=2560, mag_ratio=1, \
                               text_threshold=0.7, link_threshold=0.4, low_text=0.4, \
                               poly=False, device=device, optimal_num_chars=None)

        horizontal_list, free_list = group_text_box(text_box, slope_ths=0.1, \
                                                    ycenter_ths=0.5, height_ths=0.5, \
                                                    width_ths=0.5, add_margin=0.1, \
                                                    sort_output=True)

        jot_out = torch.cat((vis_out, horizontal_list), 1)
        out = self.head(jot_out)


        return out


class JointVisDetFineGrained(nn.Module):
    def __init__(self, idim=1512, odim=5, recog_network="standard"):
        super(JointVisDetFineGrained, self).__init__()
        self.vis_model = vis_model()
        self.det_model = get_detector(detector_path, self.device, quantize)

        if recog_network == 'standard':
            network_params = {
                'input_channel': 1,
                'output_channel': 512,
                'hidden_size': 512
                }
        elif recog_network == 'lite':
            network_params = {
                'input_channel': 1,
                'output_channel': 256,
                'hidden_size': 256
                }
        else:
            network_params = recog_config['network_params']

        self.rec_model, self.converter = get_recognizer(recog_network, network_params,\
                                                         self.character, separator_list,\
                                                         dict_list, model_path, device = self.device, quantize=quantize)
        self.head = torch.nn.Linear(idim, odim)

    def forward(self, img):
        vis_out = self.vis_model(img)
        detrec_out = self.detrec_model(img)
        jot_out = torch.cat((vis_out, detrec_out), 1)
        out = self.head(jot_out)
        return out


if __name__ == '__main__':
    img_f = ""
    joint_det = JointVisDet()
    joint_det.eval()

    joint_det(img_f)