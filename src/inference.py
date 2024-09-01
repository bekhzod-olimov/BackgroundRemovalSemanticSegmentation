import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models.nets import DISNet
from utils.general import tensor_to_numpy as tn
from models.common.model_enum import NetType

import argparse


def tensor_to_numpy(input):
    input = tn(input)
    input = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
    return cv2.resize(input, (1024, 1024))

def main(parser):
    args = parser.parse_args()
    
    image_path = args.img_name
    print(os.path.isfile(image_path))
    device = args.device
    save_outputs_path = args.save_path
    os.makedirs(save_outputs_path, exist_ok=True)
    
    # model = GTNet(1,1)
    model = DISNet(3,1)
    # model.load_gtnet(gtnet)
    saved_model_path = 'saved_models/DISNET_crop_img/isnet-epoch=322-val_loss=30.70-batch_size=8.ckpt'
    # saved_model_path = 'saved_models/best_models/disnet.ckpt'
    if os.path.isfile(saved_model_path):
        new_state_dict = {}
        state_dict = torch.load(saved_model_path, map_location=device)['state_dict']
        for key, value in state_dict.items():
            # if 'u2net' in key:
            if 'model' in key and 'gtnet' not in key:
                key = key.replace('model.', '')
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        print('model loaded successfully')
        print()

    model.to(device)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])

    output_masks = []
    image = Image.open(image_path)
    image_size = image.size
    image = transform(image)
    image = image.float()
    image = image.to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image.to(device))
        outputs = outputs[0]

    output_img = tensor_to_numpy(outputs[0])
    output_img = cv2.resize(output_img, dsize=image_size)
    image_nm = os.path.basename(image_path)
    output_path = os.path.splitext(image_nm)[0]+'.png'
    print(os.path.join(save_outputs_path, output_path))
    cv2.imwrite(f'{os.path.join(save_outputs_path, output_path)}', output_img)

    print('inference finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DIS inference arguments')
    parser.add_argument('--img_name', type=str, help='image path')
    parser.add_argument('--device', type=str, help='ex. cuda:0, cuda:1 ...')
    parser.add_argument('--save_path', type=str, help='output directory')
    main(parser)