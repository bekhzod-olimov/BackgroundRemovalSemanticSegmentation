import os

import ast
from PIL import Image
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import torch

from torchvision import transforms

def numpy_to_tensor(input_image, gt_mask):
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor()
    ])
    
    # cv2 로 들어온다고 가정하고
    # image = Image.open(input_image)
    # gt = Image.open(gt_mask)
    image = Image.fromarray(input_image)
    gt = Image.fromarray(gt_mask)
    
    return transform(image), transform(gt), image.size

def tensor_to_numpy(tensor_img, input_size):
    img_np = np.array(tensor_img.cpu().detach().squeeze(0)*255, np.uint8)
    img_np = img_np.transpose(1,2,0).squeeze()
    img_np = cv2.resize(img_np, dsize=input_size)
    # img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    return img_np

def f1_score(src, target):
    # A = 예측 결과, B = GT(정답)
    # src = A, target = B
    intersection = (src * target).sum()
    score = (2.*intersection + 1) / (src.sum() + target.sum() + 1)
    return score.detach().cpu().numpy()