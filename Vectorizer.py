import numpy as np
from collections import Counter
from tqdm import tqdm

class OneHotVectorizer():
    
    def __init__(self, dataset, n:int) -> None:
        vocab_counter = Counter()
        self.vocab=[]
        self.dim = n
        for instance in dataset:
            tokens = tokenize(instance.features)
            vocab_counter.update(tokens)
        
        for i in vocab_counter.most_common(n):
            self.vocab.append(i[0])
    
    def vectorize(self,input):
        return np.array([1 if item in tokenize(input) else 0 for item in self.vocab])
        
class CountVectorizer():
    
    def __init__(self, dataset, n:int) -> None:
        vocab_counter = Counter()
        self.vocab_dict = {}
        self.dim = n
        for instance in dataset:
            tokens = tokenize(instance.features)
            vocab_counter.update(tokens)
        
        for i in vocab_counter.most_common(n):
            self.vocab_dict[i[0]] = i[1]
        

    def vectorize(self,input):
        return np.array([self.vocab_dict[item] if item in tokenize(input) else 0 for item in self.vocab_dict])
        
    
    def token2idx(self):
        pass
    
    def idk2token(self):
        pass


class FeatureVectorizer():
    
    def __init__(self, num_features:int) -> None:
        self.vector = np.zeros(num_features)
    
    def vectorize(self):
        pass


class TFIDF_Vectorizer():
    """Term-Frequency/Inverse-Document-Frequency Vectorizer"""
    def __init__(self, dataset) -> None:
        self.N = len(dataset)
        self.doc_frequency = {}
        
        for instance in dataset:
            tokens = set(tokenize(instance.features))
            for tok in tokens:
                if tok in self.doc_frequency:
                    self.doc_frequency[tok] += 1
                else:
                    self.doc_frequency[tok] = 1
        
        self.dim = len(self.doc_frequency)
            
        
        
        
    def vectorize(self,input):
        tf_counts = Counter(tokenize(input))
        vector = np.zeros(len(self.doc_frequency))
        idx = 0
        for i,v in self.doc_frequency.items():
            if i not in tf_counts:
                vector[idx]=0
            else:
                tf = tf_counts[i]
                idf = np.log(self.N/v)
                vector[idx] = tf*idf
            idx+=1
        return vector
        
    
    def token2idx(self):
        pass
    
    def idk2token(self):
        pass
    
@staticmethod
def tokenize(text):
    return [tok.lower().strip() for tok in text.split()]
    
    
