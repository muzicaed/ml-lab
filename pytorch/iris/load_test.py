import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import Model

pd.set_option('future.no_silent_downcasting', True)

new_model = Model()
new_model.load_state_dict(torch.load('iris.pt'))

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = pd.read_csv(url)
df['variety'] = df['variety'].replace('Setosa', 0)
df['variety'] = df['variety'].replace('Versicolor', 1)
df['variety'] = df['variety'].replace('Virginica', 2)
features = df.drop('variety', axis=1).to_numpy()
lables = np.array(df['variety'], dtype=np.float16)
features = torch.FloatTensor(features)
lables = torch.LongTensor(lables)

# Validate
criterion = nn.CrossEntropyLoss()
correct = 0
with torch.no_grad():
    for i, data in enumerate(features):
        pred = new_model.forward(data)
        if pred.argmax().item() == lables[i]:
            correct += 1

print(f'Test correct: {correct} / {len(features)}')
