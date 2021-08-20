import numpy as np
from PIL import Image
import cv2
def get_palette():
    unlabelled = [0, 0, 0]
    car        = [64, 0, 128]
    person     = [64, 64, 0]
    bike       = [0, 128, 192]
    curve      = [0, 0, 192]
    car_stop   = [128, 128, 0]
    guardrail  = [64, 64, 128]
    color_cone = [192, 128, 128]
    bump       = [192, 64, 0]
    palette    = np.array([unlabelled,car, person, bike, curve, car_stop, guardrail, color_cone, bump])
    return palette


def visualize(dir, image_name):
    palette = get_palette()
    gray = cv2.imread(dir)
    img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            cid = gray[i,j]

            img[i,j,:] = palette[cid][0]
    img = Image.fromarray(np.uint8(img))
    img.save('/Pred_' + image_name)


dir = './data/labels/00351D.png'
Image_name = dir.split('/')[-1]
visualize(dir,Image_name)

