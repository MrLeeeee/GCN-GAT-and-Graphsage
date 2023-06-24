# coding: UTF-8
import time
import torch
import numpy as np
import models
from config import opt
import data
from sklearn import metrics
from tqdm import tqdm
import visdom
from torch.utils.data import DataLoader
import torch.nn.functional as F


def train(**kwargs):
    opt._parse(kwargs)
    adj, features, labels, idx_train, idx_val, idx_test = data.load_data(opt)
    train = data.Dataload(labels, idx_train)
    val = data.Dataload(labels, idx_val)
    test = data.Dataload(labels, idx_test)
    if opt.model is 'PyGCN':
        model = getattr(models, opt.model)(features.shape[1], 128, max(labels) + 1).train()
    elif opt.model is 'PyGAT':
        model = getattr(models, opt.model)(features.shape[1], 8, max(labels) + 1, dropout=0.6, alpha=0.2, nheads=8).train()
    elif opt.model is 'PyGraphsage':
        model = getattr(models, opt.model)(features.shape[1], 8, max(labels) + 1)
    else:
        print("Please input the correct model name: PyGCN, PyGAT or PyGraphsage")
        return
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model = model.to(opt.device)
    adj = adj.to(opt.device)
    features = features.to(opt.device)   # 将模型以及在模型中需要使用到的矩阵加载到设备中
    train_dataloader = DataLoader(train, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_dataloader = DataLoader(test, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    criterion = F.nll_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    lr = opt.lr
    for epoch in range(opt.max_epoch):
        for trains, labels in tqdm(train_dataloader):
            labels = labels.to(opt.device)
            trains = trains.to(opt.device)
            optimizer.zero_grad()
            outputs = model(features, adj)
            loss = criterion(outputs[trains], labels)
            loss.backward()
            optimizer.step()
        lr = lr * opt.lr_decay
        for param_groups in optimizer.param_groups:
            param_groups['lr'] = lr
        evalute(opt, model, val_dataloader, epoch, features, adj)
        model.train()
        model.save()
    evalute(opt, model, test_dataloader, 'Test', features, adj)

def evalute(opt, model, val_dataloader, epoch, features, adj):

    model.eval()
    loss_total = 0
    predict_all = list()
    labels_all = list()
    critetion = F.nll_loss
    with torch.no_grad():
        for evals, labels in tqdm(val_dataloader):
            labels = labels.to(opt.device)
            evals = evals.to(opt.device)
            outputs = model(features, adj)
            loss = critetion(outputs[evals], labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs[evals].data, 1)[1].cpu().numpy()
            labels = list(labels)
            predic = list(predic)
            labels_all.extend(labels)
            predict_all.extend(predic)
    acc = metrics.accuracy_score(labels_all, predict_all)
    print("The acc for Epoch %s is %f" % (str(epoch), acc))
    return acc

if __name__ == '__main__':
    train()
    print("test")