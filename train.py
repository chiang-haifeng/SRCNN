import argparse
import os
import copy
import numpy as np
import torch
from PIL import Image
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import metrics
from models import SRCNN
from SRdataset import TrainDataset, ValDataset
from utils import AverageMeter, calc_psnr, calculate_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='data/Iris224trainhq_56_224')
    parser.add_argument('--eval-file', type=str, default='data/Iris224validhq_56_224')
    parser.add_argument('--outputs-dir', type=str, default='./output')
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--weights-file', type=str, default=None)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    # 判断是否加载预训练模型
    if not args.weights_file is None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)

    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = ValDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)

                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            # 看看情况
            # inputs = metrics.tensor2img(inputs)
            # labels = metrics.tensor2img(labels)
            # metrics.save_img(inputs, 'inputs.png')
            # metrics.save_img(labels, 'labels.png')


            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            labels = metrics.tensor2img(labels)
            preds = metrics.tensor2img(preds)
            metrics.save_img(preds, 'preds.png')
            metrics.save_img(labels, 'labels.png')

            preds = Image.open("preds.png")
            labels = Image.open("labels.png")
            preds = np.array(preds)
            labels = np.array(labels)
            curr_psnr = metrics.calculate_psnr(preds, labels)
            epoch_psnr.update(curr_psnr, len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
