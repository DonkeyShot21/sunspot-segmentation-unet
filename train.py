import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from unet import UNet
from mias_dataset import MIASDataset
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))

def save_chart(loss):
    plt.plot(range(len(loss)), loss, c='C0')
    plt.savefig("logs/loss_chart.png")


def save_example(img, gt_mask, pred_mask):
    img = img.cpu().data.numpy()[0]
    gt_mask = gt_mask.cpu().data.numpy()[0]
    pred_mask = pred_mask.cpu().data.numpy()[0]

    img += 128
    gt_mask *= 150 # should be 255 but would saturate image
    pred_mask *= 150 # making it impossible to see if it is doing it right
    pred_mask[pred_mask>255] = 255

    #img = cv2.resize(img, (350,350))
    #gt_mask = cv2.resize(gt_mask, (350,350))
    #pred_mask = cv2.resize(pred_mask, (350,350))

    img = np.expand_dims(img, axis=2)
    gt_mask = np.expand_dims(gt_mask, axis=2)
    pred_mask = np.expand_dims(pred_mask, axis=2)

    img_gt = np.concatenate((img, img, gt_mask), axis=2)
    img_pred = np.concatenate((img, img, pred_mask), axis=2)
    gt = np.concatenate((gt_mask, gt_mask, gt_mask), axis=2)
    pred = np.concatenate((pred_mask, pred_mask, pred_mask), axis=2)

    gt_concat = np.concatenate((img_gt,gt), axis=1)
    pred_concat = np.concatenate((img_pred,pred), axis=1)

    example = pred_concat = np.concatenate((gt_concat,pred_concat), axis=0)

    cv2.imwrite('logs/example_image.png', example.astype(np.uint8))


def train():
    batch_size = 1 # with less then 32gb ram better to use 1 / 2
    grad_accu_times = 3
    init_lr = 0.01
    img_dir = 'data/all-mias'
    metadata_path = 'metadata.csv'
    dataset = MIASDataset(img_dir, metadata_path)

    model = UNet().cpu()

    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=init_lr)
    opt.zero_grad()

    epoch = 0
    forward_times = 0
    loss_record = []
    for epoch in range(100):
        data_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)

        lr = init_lr * (0.1 ** (epoch // 10))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

        for idx, batch_data in enumerate(data_loader):

            batch_input = Variable(batch_data['img']).cpu()
            batch_gt_mask = Variable(batch_data['mask']).cpu()

            pred_mask = model(batch_input)
            forward_times += 1

            loss = loss_fn(pred_mask, batch_gt_mask)
            loss += dice_loss(F.sigmoid(pred_mask), batch_gt_mask)
            loss.backward()

            print('Epoch:',epoch+1,'| Batch:',idx+1,'| lr:',lr,'| Loss:',loss.cpu().data.numpy())
            loss_record.append(loss.cpu().data.numpy())

            if forward_times == grad_accu_times:
                print("\nUpdate weights...")
                opt.step()
                opt.zero_grad()
                forward_times = 0

                print("\nSave example...")
                save_example(batch_input[0], batch_gt_mask[0], F.sigmoid(pred_mask[0]))

                print("Save chart...\n")
                save_chart(loss_record)


        if (epoch+1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : opt.state_dict(),
            }
            torch.save(checkpoint, 'checkpoints/unet1024-{}'.format(epoch+1))
        del data_loader


if __name__ == '__main__':
    train()
