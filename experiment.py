import numpy as np
import torch
from models.unet import UNet
from utils import *

from ddpm import Diffusion


def main(args):
    device = args.device
    model = UNet(num_classes=args.num_classes, device=device).to(device)
    diffusion = Diffusion(img_size=args.image_size, device=device)
    
    labels = torch.arange(10).long().to(device)
    sampled_images = diffusion.sample(model, n=len(labels), labels=labels)

    b, c, w, h = sampled_images.shape

    for i in range(b):
        img = sampled_images[i]
        img = img.squeeze().detach().cpu().numpy()
        img = img.transpose(1,2,0)
        plt.imshow(img)
        plt.show() 

    breakpoint()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = 'DDPM_c'
    args.epochs = 100
    args.batch_size = 4
    args.image_size = 32
    args.num_classes = 10
    args.dataset_path = '../dataset'
    args.device = 'cuda'
    args.lr = 3e-5
    main(args)