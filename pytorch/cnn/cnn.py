import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from lib import load_data, CCNNetwork
import time

torch.manual_seed(41)

model = CCNNetwork()
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

start_time = time.time()
train_loader, test_loader = load_data()
for i in range(epochs):
    train_corr = 0
    test_corr = 0
    for b, (feat_train, lable_train) in enumerate(train_loader):
        b += 1
        pred = model(feat_train)
        loss = crit(pred, lable_train)
        batch_corr = (torch.max(pred.data, 1)[1] == lable_train).sum()
        train_corr += batch_corr

        opt.zero_grad()
        loss.backward()
        opt.step()

        if b % 1000 == 0:
            print(f'{i+1} - B: {b} - L: {loss.item()}')
    train_losses.append(loss)
    train_correct.append(train_corr)

torch.save(model.state_dict(), 'mnist-cnn.pt')
print('-----------------')

with torch.no_grad():
    for b, (feat_test, lable_test) in enumerate(test_loader):
        b += 1
        pred = model(feat_test)
        loss = crit(pred, lable_test)
        batch_corr = (torch.max(pred.data, 1)[1] == lable_test).sum()
        test_corr += batch_corr

    test_losses.append(loss)
    test_correct.append(test_corr)


print(f'Time: {(time.time() - start_time) / 60}')
train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.show()
