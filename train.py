import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
from dataset import HelioDataset
import torch.nn.functional as F
import numpy as np
import cv2, sys, os
import matplotlib.pyplot as plt


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))

# def save_chart(loss):
#     plt.plot(range(len(loss)), loss, c='C0')
#     plt.savefig("logs/loss_chart.png")


# def save_example(img, gt_mask, pred_mask):
#     img = img.cpu().data.numpy()[0]
#     gt_mask = gt_mask.cpu().data.numpy()[0]
#     pred_mask = pred_mask.cpu().data.numpy()[0]
#
#     img += 128
#     gt_mask *= 150 # should be 255 but would saturate image
#     pred_mask *= 150 # making it impossible to see if it is doing it right
#     pred_mask[pred_mask>255] = 255
#
#     #img = cv2.resize(img, (350,350))
#     #gt_mask = cv2.resize(gt_mask, (350,350))
#     #pred_mask = cv2.resize(pred_mask, (350,350))
#
#     img = np.expand_dims(img, axis=2)
#     gt_mask = np.expand_dims(gt_mask, axis=2)
#     pred_mask = np.expand_dims(pred_mask, axis=2)
#
#     img_gt = np.concatenate((img, img, gt_mask), axis=2)
#     img_pred = np.concatenate((img, img, pred_mask), axis=2)
#     gt = np.concatenate((gt_mask, gt_mask, gt_mask), axis=2)
#     pred = np.concatenate((pred_mask, pred_mask, pred_mask), axis=2)
#
#     gt_concat = np.concatenate((img_gt,gt), axis=1)
#     pred_concat = np.concatenate((img_pred,pred), axis=1)
#
#     example = pred_concat = np.concatenate((gt_concat,pred_concat), axis=0)
#
#     cv2.imwrite('logs/example_image.png', example.astype(np.uint8))


def train():
    batch_size = 1
    init_lr = 0.01

    dataset = HelioDataset('./data/SIDC_dataset.csv',
                           'data/sDPD2014.txt',
                           10)

    model = UNet().cpu()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    opt.zero_grad()
    epoch = 0

    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]):
            print("=> loading checkpoint '{}'".format(sys.argv[1]))
            checkpoint = torch.load(sys.argv[1])
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            opt.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(sys.argv[1], checkpoint['epoch']))

    loss_record = []
    for epoch in range(epoch, 100):
        data_loader = DataLoader(dataset)

        lr = init_lr * (0.1 ** (epoch // 10))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        for idx, batch_data in enumerate(data_loader):
            for i in range(0,len(batch_data['img'][0])):
                patch_input = Variable(batch_data['img'][0][i:i+1]).cpu()
                patch_gt_mask = Variable(batch_data['mask'][0][i:i+1]).cpu()

                pred_mask = model(patch_input)

                loss = loss_fn(pred_mask, patch_gt_mask)
                loss += dice_loss(F.sigmoid(pred_mask), patch_gt_mask)
                loss.backward()

                print('Epoch:',epoch+1,'| Batch:',idx+1,'| Patch:', i+1, '| lr:',lr,'| Loss:',loss.cpu().data.numpy())
                loss_record.append(loss.cpu().data.numpy())

                if (i+1) % 3 == 0:
                    print("\nUpdate weights...")
                    opt.step()
                    opt.zero_grad()


        if (epoch+1) % 1 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
            }
            torch.save(checkpoint, 'checkpoints/unet1024-{}'.format(epoch+1))
        del data_loader


if __name__ == '__main__':
    train()
