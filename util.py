"""
created by: Donghyeon Won
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import torch
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


class ProtestDataset_txtfts(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, id_label_trans_train_f, id_label_trans_f, id_path_f, transform = None):
        """
        Args:
            id_label_trans_f: Path to txt file with annotation
            id_path_f:
            transform: Optional transform to be applied on a sample.
        """

        self.id_path_df = pd.read_csv(id_path_f, delimiter=",")
        self.id_lab_train_trans = pd.read_csv(id_label_trans_train_f, delimiter=",")  # ck
        self.id_lab_trans = pd.read_csv(id_label_trans_f, delimiter=",")  # ck

        self.transform = transform
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=30, norm="l2", ngram_range=(1, 2), encoding="utf-8", stop_words="english")
        trans = self.id_lab_trans["trans"]
        self.tfidf.fit_transform(trans.values.astype('U')).toarray()
    def __len__(self):
        return len(self.id_path_df)
    def __getitem__(self, idx):
        imgpath = self.id_path_df.iloc[idx, 1]
        image = pil_loader(imgpath)

        protest_demand = self.id_lab_train_trans.iloc[idx, 1:2].to_numpy().astype('float32')
        label = {'protest_demand':protest_demand}

        trans = np.array([self.id_lab_train_trans.iloc[idx, 2]]).astype('U')
        tfidf_fts = self.tfidf.transform(trans).toarray()[0].astype('float32')

        sample = {"image": image, "label": label, "text_fts": tfidf_fts}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample


class ProtestDataset_fts(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, bfts_file, img_dir,  transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
                id
            bfts_file:
                id bbox_ft1, bbox_ft2, bbox_ft3
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.bbox_fts_frame = pd.read_csv(bfts_file, delimiter=",")  # ck
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)

        protest = self.label_frame.iloc[idx, 1:2].to_numpy().astype('float')
        sign = self.label_frame.iloc[idx, 3:4].to_numpy().astype('float')
        #violence = self.label_frame.iloc[idx, 2:3].to_numpy().astype('float')
        #visattr = self.label_frame.iloc[idx, 3:].to_numpy().astype('float')
        label = {'protest':protest, 'sign':sign}

        bbox_feats = self.bbox_fts_frame.iloc[idx, 1:].to_numpy().astype('float32')

        sample = {"image": image, "label": label, "bbox_feats": bbox_feats}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample



class ProtestDataset(Dataset):
    """
    dataset for training and evaluation
    """
    def __init__(self, txt_file, img_dir, transform = None):
        """
        Args:
            txt_file: Path to txt file with annotation
            img_dir: Directory with images
            transform: Optional transform to be applied on a sample.
        """
        self.label_frame = pd.read_csv(txt_file, delimiter="\t").replace('-', 0)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.label_frame)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.label_frame.iloc[idx, 0])
        image = pil_loader(imgpath)

        protest = self.label_frame.iloc[idx, 1:2].to_numpy().astype('float')
        violence = self.label_frame.iloc[idx, 2:3].to_numpy().astype('float')
        visattr = self.label_frame.iloc[idx, 3:].to_numpy().astype('float')
        label = {'protest':protest, 'violence':violence, 'visattr':visattr}

        sample = {"image":image, "label":label}
        if self.transform:
            sample["image"] = self.transform(sample["image"])
        return sample


class ProtestDatasetEval(Dataset):
    """
    dataset for just calculating the output (does not need an annotation file)
    """
    def __init__(self, img_dir):
        """
        Args:
            img_dir: Directory with images
        """
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                                ])
        self.img_list = sorted(os.listdir(img_dir))
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        imgpath = os.path.join(self.img_dir,
                                self.img_list[idx])
        image = pil_loader(imgpath)
        # we need this variable to check if the image is protest or not)
        sample = {"imgpath":imgpath, "image":image}
        sample["image"] = self.transform(sample["image"])
        return sample

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count

class Lighting(object):
    """
    Lighting noise(AlexNet - style PCA - based noise)
    https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/recognition/dataset/minc.py
    """
    
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
