import random
import json 
import torch 
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize,stem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

file = 'data.pth'
data = torch.load(file)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, output_size, hidden_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'sam'
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input('You: ')
    if sentence.lower() == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    print(output)
    _,predict = torch.max(output,dim=1)
    # print(predict)
    tag = tags[predict.item()]
    for intent in intents['intents']:
        # print(tag)
        if tag == intent["tag"]:
            responses = random.choice(intent["responses"])
    else:
        print(f"{bot_name}: I do not understand...")