import numpy as np
import torch 
import random 
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from nltk_utils import tokenize,stem,bag_of_words
from model import NeuralNet
import json

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags  = []
xy = []

for intent in intents['intents']:
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, intent['tag']))

ignore_words = ['?',',','.']
all_words  = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))
tags =  sorted(set(tags))

# print(len(all_words))
# print(len(tags))

x_train = []
y_train = []

for (words,tag) in xy:
    bags = bag_of_words(words,all_words)
    x_train.append(bags)
    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

# print(x_train.shape)
# print(y_train.shape)

num_epochs = 500
batch_size = 8
learning_rate = 0.001
input_size = len(x_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDateset(Dataset):
    def __init__(self):
        self.n_sample = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
    
    def __len__(self):
        return self.n_sample
    
dataset = ChatDateset()
# print(dataset)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=input_size, hidden_size=hidden_size, out_size=output_size)

cross_intropy = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words,labels) in dataloader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = cross_intropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}
file = 'data.pth'
torch.save(data,file)
print(f'training complete. file saved to {file}')