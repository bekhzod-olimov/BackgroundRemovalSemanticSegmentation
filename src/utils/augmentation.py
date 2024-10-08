import numpy as np
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import random

class RandomBlur(object):

    def __call__(self,image=None, mask=None):
        # random blur
        img = image.copy()
        mask = mask.copy()
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        max_area = 0.0
        max_contour = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area:
                # coor = tuple(contour[random.randint(0, len(contour))][0])
                max_contour = contour
                max_area = area

        h,w = img.shape[:2]

        repeat = random.randint(1,5)
        for a in range(repeat):
            target_coor = tuple(max_contour[random.randint(0, len(max_contour) - 1)][0])
            cy = target_coor[1]
            cx = target_coor[0]
            patch_size = random.randint(100, 200)
            
            x1 = cx - patch_size
            y1 = cy - patch_size
            x2 = cx + patch_size
            y2 = cy + patch_size
            
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h
            
            patch = img[y1:y2, x1:x2]
            kernel_size = random.randint(6,20)
            patch = cv2.blur(patch, (kernel_size, kernel_size))
            img[y1:y2, x1:x2] = patch

        return {'image':img, 'mask':mask}