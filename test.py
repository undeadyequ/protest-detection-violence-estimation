import easyocr
import os


if __name__ == '__main__':
    reader = easyocr.Reader(['ch_sim', 'en'])  # need to run only once to load model into memory
    img_dir = "bing/black+protest/"
    res_txt = "result.txt"
    res_ls = []

    for img in os.listdir(img_dir):
        img_p = os.path.join(img_dir, img)
        try:
            result = reader.readtext(img_p)
            res_ls.append(result)
        except:
            print("{} didn't work".format(img_p))

    with open(res_txt, "w") as f:
        for row in res_ls:
            pos, char, confidence = row
            pos_one = ["{}  {}".format(x, y) for x, y in pos]
            f.write("{} {}  {}".format("  ".join(pos_one), char, confidence))
