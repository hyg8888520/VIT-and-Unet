import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from VOCdata.vocload import VOC_SEG
from nets import Unet
import os
import numpy as np


# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


# 根据混淆矩阵计算Acc和mIou
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    return acc, acc_cls, mean_iu


def main():
    # 1. load dataset
    root = "VOCdata/VOCdevkit/VOC2007"
    batch_size = 8
    height = 224
    width = 224
    voc_train = VOC_SEG(root, width, height, train=True)
    voc_test = VOC_SEG(root, width, height, train=False)
    train_dataloader = DataLoader(voc_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(voc_test, batch_size=batch_size, shuffle=True)

    # 2. load model
    in_ch, out_ch = 3, 21
    model = Unet(in_ch,out_ch)
    device = torch.device('cuda')
    model = model.to(device)

    # 3. prepare super parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.7)
    epoch = 777

    # 4. train
    val_acc_list = []
    out_dir = "./checkpoints/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for epoch in range(0, epoch):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            length = len(train_dataloader)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # torch.size([batch_size, out_ch, width, height])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            predicted = torch.argmax(outputs.data, 1)

            label_pred = predicted.data.cpu().numpy()
            label_true = labels.data.cpu().numpy()
            acc, acc_cls, mean_iu = label_accuracy_score(label_true, label_pred, out_ch)

            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Acc_cls: %.03f%% |Mean_iu: %.3f'
                  % (epoch + 1, (batch_idx + 1 + epoch * length), sum_loss / (batch_idx + 1),
                     100. * acc, 100. * acc_cls, mean_iu))

        # get the ac with testdataset in each epoch
        print('Waiting Val...')
        mean_iu_epoch = 0.0
        mean_acc = 0.0
        mean_acc_cls = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_dataloader):
                model.eval()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicted = torch.argmax(outputs.data, 1)

                label_pred = predicted.data.cpu().numpy()
                label_true = labels.data.cpu().numpy()
                acc, acc_cls, mean_iu = label_accuracy_score(label_true, label_pred, out_ch)

                # total += labels.size(0)
                # iou = torch.sum((predicted == labels.data), (1,2)) / float(width*height)
                # iou = torch.sum(iou)
                # correct += iou
                mean_iu_epoch += mean_iu
                mean_acc += acc
                mean_acc_cls += acc_cls

            print('Acc_epoch: %.3f%% | Acc_cls_epoch: %.03f%% |Mean_iu_epoch: %.3f'
                  % ((100. * mean_acc / len(val_dataloader)), (100. * mean_acc_cls / len(val_dataloader)),
                     mean_iu_epoch / len(val_dataloader)))

            val_acc_list.append(mean_iu_epoch / len(val_dataloader))

        torch.save(model.state_dict(), out_dir + "last.pt")
        if mean_iu_epoch / len(val_dataloader) == max(val_acc_list):
            torch.save(model.state_dict(), out_dir + "best.pt")
            print("save epoch {} model".format(epoch))


if __name__ == "__main__":
    main()