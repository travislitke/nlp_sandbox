import csv
import Dataset
import json
import Models
from Models import LogisticRegression as lr
import nltk
import Vectorizer
from nltk.corpus import brown, gutenberg

filepath = "C:\\Datasets\\IMDB Dataset.csv"


data = []
with open(file=filepath,mode='r',encoding='utf-8') as file:
    lines = csv.reader(file)
    for line in lines:
        data.append((line[0],line[1]))

train_size = 2000
eval_size = int(train_size*0.8)
training_data = Dataset.Dataset(data[:train_size])
test_data = Dataset.Dataset(data[train_size:train_size+eval_size])
n = 200

vectorizer = Vectorizer.CountVectorizer(dataset=training_data, n=200)

model = lr.LogisticRegression(vectorizer=vectorizer)
rates = [0.1,0.2,0.3]

for i in range(10):
    print(f"Epoch {i+1}:")
    model.train(training_data,learning_rate=0.1)
model.test(test_data)