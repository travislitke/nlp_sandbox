import csv


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
