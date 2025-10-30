import csv
import linear_algebra
import re
from collections import Counter
from Models import NaiveBayes as nb
from tqdm import tqdm
from vocabulary import Vocabulary


E = 2.718281828459045235360287471352

"""Datasets:"""

pdtb_path = 'C:\\Datasets\\pdtb\\train.json'
movie_review_path = 'C:\\Datasets\\IMDB Dataset.csv'
macbeth = 'C:\\Datasets\\macbeth.txt'
shakes = 'C:\\Datasets\\shakespeare_full.txt'


corpus = ""
with open(shakes, 'r', encoding='utf-8') as file:
    for line in tqdm(file, desc='Loading corpus'):
        line = re.sub(r'[A-Z]{2,}','',line)
        line = re.sub(r'\n',' ',line)
        corpus += line

bayes = nb.NaiveBayes(corpus=corpus)
bayes.make_ngrams(2)
bayes.probs()

bayes.generate(200)