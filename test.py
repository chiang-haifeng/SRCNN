import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from torch.utils.data import DataLoader
from PIL import Image
import metrics
from SRdataset import ValDataset
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, AverageMeter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='output/x4/epoch_1.pth')
    parser.add_argument('--eval-file', type=str, default='C:/workspace/SR3/dataset/CASIAV1_56_224')
    parser.add_argument('--scale', type=int, default=4)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)
    eval_dataset = ValDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    epoch_psnr = AverageMeter()
    idx = 0
    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(inputs).clamp(0.0, 1.0)

        labels = metrics.tensor2img(labels)
        preds = metrics.tensor2img(preds)
        curr_psnr = metrics.calculate_psnr(preds, labels)
        epoch_psnr.update(curr_psnr, len(inputs))
        metrics.save_img(preds, '{}/{}_sr.png'.format('./out', idx))
        # preds = Image.fromarray(preds)
        # preds.save("your_file.jpeg")
        idx = idx + 1
    print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

