import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import torch.utils
import torch.utils.data
from model import Model
from sklearn.model_selection import train_test_split

pd.set_option('future.no_silent_downcasting', True)


def load_data():
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    df = pd.read_csv(url)
    df['variety'] = df['variety'].replace('Setosa', 0)
    df['variety'] = df['variety'].replace('Versicolor', 1)
    df['variety'] = df['variety'].replace('Virginica', 2)
    features = df.drop('variety', axis=1).to_numpy()
    lables = np.array(df['variety'], dtype=np.float16)

    feat_train, feat_test, lables_train, lables_test = train_test_split(
        features, lables)
    feat_train = torch.FloatTensor(feat_train)
    feat_test = torch.FloatTensor(feat_test)
    lables_train = torch.LongTensor(lables_train)
    lables_test = torch.LongTensor(lables_test)

    return feat_train, feat_test, lables_train, lables_test


torch.manual_seed(10)
model = Model()
feat_train, feat_test, lables_train, lables_test = load_data()
feat_train = torch.FloatTensor(feat_train)
feat_test = torch.FloatTensor(feat_test)
lables_train = torch.LongTensor(lables_train)
lables_test = torch.LongTensor(lables_test)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
epochs = 300
losses = []

for i in range(epochs):
    pred = model.forward(feat_train)
    loss = criterion(pred, lables_train)
    losses.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch: {i} and loss: {loss}')

# Validate
correct = 0
with torch.no_grad():
    for i, data in enumerate(feat_test):
        pred = model.forward(data)
        pred_item = pred.argmax().item()
        if pred_item == lables_test[i]:
            correct += 1

print(f'Test correct: {correct} / {len(feat_test)}')

torch.save(model.state_dict(), 'iris.pt')
