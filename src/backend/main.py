import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from text_formatter import bag_of_words, tokenizator, normal_word
from neural_network import NeuralNetwork
from nltk.corpus import stopwords

with open('../tags.json', 'r', encoding='UTF-8') as f:
    links = json.load(f)

with open('../answers.json', 'r', encoding='UTF-8') as f:
    answers = json.load(f)

all_words = []
tags = []
prepared_data = []

train_data = pd.read_excel('../DataSet.xlsx')
x = np.array(train_data['Обращение'])
y = np.array(train_data['Тип обращения'])

# распаршиваем запрос
for i in range(len(x)):
    if y[i].lower() not in tags:
        tags.append(y[i].lower())
    tokens = tokenizator(x[i])
    # собираем саписок всех слов
    all_words.extend(tokens)
    prepared_data.append((tokens, y[i].lower()))


# приводим слова к одиноковому морфологическому виду и убираем лишние
ignored_words = ['?', '.', '!', ',', '(', ')', '-', '...', ':', ', ', '/', '—',
                 '«', '»', 'хочу', 'хотеть', 'хотеться', 'хотя', 'добрый',
                 'день', 'вечер', 'здравствть', 'здравствуй', 'здравствуйте',
                 'здравствуйте.', 'здраствовать', 'пожалуйста', 'пожалуцст',
                 'привет']
stop_words = stopwords.words("russian") + ignored_words
all_words = [normal_word(word) for word in all_words if normal_word(word) not in stop_words]
all_words = sorted(set(all_words))
tags = sorted(tags)

# print(len(tags), "Тэги:", tags)
# print(len(all_words), "Уникальные слова:", all_words)

# собираем данные для обучения
X_train = []
y_train = []

for (pattern_sentence, tag) in prepared_data:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Параметры нейронной сети
count_epochs = 150
batch_size = 20
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 60
output_size = len(tags)


class DatasetForChat(Dataset):
    def __init__(self):
        self.samples_count = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.samples_count


dataset = DatasetForChat()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# критерий и функция оптимизации
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# обучение
for epoch in range(count_epochs):
    for (words, tags_indexes) in train_loader:
        words = words.to(device)
        tags_indexes = tags_indexes.to(dtype=torch.long).to(device)
        outputs = model(words)
        losses = criterion(outputs, tags_indexes)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{count_epochs}], Loss: {losses.item():.4f}')

print(f'final losses: {losses.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
    "links": links,
    "answers": answers
}

# сохраняем обученную модель
FILE = "model.pth"
torch.save(data, FILE)

print(f'Обучение завершено. файл сохранен как: {FILE}')
