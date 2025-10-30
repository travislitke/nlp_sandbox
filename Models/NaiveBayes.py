import re
import numpy as np

class NaiveBayes():
    END = "EOS"
    
    def __init__(self, corpus:str):
        self.corpus = corpus
        self.tokens = re.findall(r'\w+|[^\w\s]', corpus)
        self.total = len(self.tokens)
        self.grams = None
        self.prob_dict = None
        
        
    def count(self):
        count_dict = {}
        for tok in self.tokens:
            if tok not in count_dict:
                count_dict[tok] = 1
            else:
                count_dict[tok] += 1
        return count_dict
            
    
    def make_ngrams(self, grams:int =2):
        bigrams = []
        for i in range(len(self.tokens)-1):
            bigrams.append((self.tokens[i],self.tokens[i+1]))
        self.grams = bigrams
        return bigrams
    
    def probs(self):
        probs_dict = {}
        for item in self.grams:
            lead = item[0]
            follow = item[1]
            if lead not in probs_dict:
                probs_dict[lead] = {follow:1}
            elif follow not in probs_dict[lead]:
                probs_dict[lead][follow] = 1
            else:
                probs_dict[lead][follow] += 1
        
        for tok in probs_dict:
            total = sum(v for _,v in probs_dict[tok].items())
            for item in probs_dict[tok]:
                probs_dict[tok][item] = probs_dict[tok][item]/total

        self.prob_dict = probs_dict
        return probs_dict
                    
    
    def generate(self, num_sents = 1) -> str:
        text = []
        sents = 0
        text.append(np.random.choice([tok for tok in self.prob_dict['.']]))
        while sents < num_sents:
            next = np.random.choice(a=[tok for tok in self.prob_dict[text[-1]]],p=[v for _,v in self.prob_dict[text[-1]].items()])
            if next in ['.','!','?']:
                sents += 1
            text.append(next)

        print(" ".join(text))
        return text
    
    