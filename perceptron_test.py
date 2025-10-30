import csv
from Models import Perceptron
from tqdm import tqdm

E = 2.718281828459045235360287471352

"""Datasets:"""

pdtb_path = 'C:\\Datasets\\pdtb\\train.json'
movie_review_path = 'C:\\Datasets\\IMDB Dataset.csv'

"""Load data"""

pos_sentiment = []
with open("C:\\Datasets\\positive_sentiment_lexicon.txt") as file:
    for line in file:
        pos_sentiment.append(line.strip())

neg_sentiment = []
with open("C:\\Datasets\\negative_sentiment_lexicon.txt") as file:
    for line in file:
        neg_sentiment.append(line.strip())

movie_review_data = []
with open(movie_review_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for line in reader:
        movie_review_data.append(line)

movie_review_data.remove(movie_review_data[0])

data = []
for item in movie_review_data:
    sent = " ".join([token.lower().strip() for token in item[0].split()])
    instance = sent, item[1]
    data.append(instance)

label_to_idx = {"positive": 1,
                "negative": 0
                }

idx_to_label = {1: "positive",
                0: "negative"
                }

raw_training_data = data[:int(len(data)*0.7)]
raw_test_data = data[int(len(data)*0.7):]

"""Featurize the data instances for the machine"""
training_data = []
for instance in tqdm(raw_training_data, desc="Featurizing training data"):
    sent = instance[0]
    label = instance[1]
    feature_vector = [0, 0, 0]
    for token in sent.split():
        if token in pos_sentiment:
            feature_vector[0] += 1
        if token in neg_sentiment:
            feature_vector[1] += 1
    feature_vector[2] = len(sent)
    label_vector = label_to_idx[label]
    vectorized = (feature_vector, label_vector)
    training_data.append(vectorized)

test_data = []
for instance in tqdm(raw_test_data, desc="Featurizing test data"):
    sent = instance[0]
    label = instance[1]
    feature_vector = [0, 0, 0]
    for token in sent.split():
        if token in pos_sentiment:
            feature_vector[0] += 1
        if token in neg_sentiment:
            feature_vector[1] += 1
    feature_vector[2] = len(sent)
    label_vector = label_to_idx[label]
    vectorized = (feature_vector, label_vector)
    test_data.append(vectorized)

perceptron = Perceptron.Perceptron(input_dim=len(feature_vector), bias=10)
perceptron.train(training_data, learning_rate=0.75)
perceptron.evaluate(test_data)
# end of file
