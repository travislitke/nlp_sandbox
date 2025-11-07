import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

class NaiveBayes():
    END = "EOS"
    
    def __init__(self, filepath:str):
        self.data = filepath
        self.tokens = []
        self.total = None
        self.grams = None
        self.prob_dict = None
        self.probs_built = False
        
        
    def count(self):
        count_dict = {}
        for tok in self.tokens:
            if tok not in count_dict:
                count_dict[tok] = 1
            else:
                count_dict[tok] += 1
        return count_dict
    
    def tokenize_corpus(self):
        file_path = Path(self.data)
        token_filepath = Path(self.data+'\\tokens.txt')
        omit = ['*','_']
        if token_filepath.exists():
            with open(token_filepath, 'r', encoding='utf-8') as file:
                for line in file:
                    self.tokens.append(line)
        else:
            corpus = self.load_corpus(file_path)
            for tok in corpus.split():
                if tok in omit:
                    pass
                elif re.match(r'^[^\w_].*[^\w_]$',tok):                
                    pass
                elif re.match(r'^[_]|[_]$',tok):
                    pass
                elif re.match(r'\b[A-Z]{2,}\b',tok):
                    pass
                elif re.match(r'\b.+[.?!]$',tok):
                    self.tokens.extend([tok[:-1],tok[-1],'EOS'])

                    
                else:
                    self.tokens.append(tok)    
                
                
    def make_ngrams(self, grams:int =2):
        bigrams = []
        for i in range(len(self.tokens)-1):
            bigrams.append((self.tokens[i],self.tokens[i+1]))
        self.grams = bigrams
        return bigrams
    
    def build_probs(self):
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
        if not self.prob_dict:
            raise NotImplementedError
        
        text = []
        
        starts = [tok for tok in self.prob_dict['EOS']]
        seed = np.random.choice(a=starts,p=[v for _,v in self.prob_dict['EOS'].items()])
        sentence = [seed]
        while len(text) < num_sents:
            next = np.random.choice(a=[tok for tok in self.prob_dict[seed]],p=[v for _,v in self.prob_dict[seed].items()])
            if next == 'EOS':
                if len(sentence)<2:
                    pass
                sentence[-2]+= sentence[-1]
                sentence[0].capitalize()
                sentence.remove(sentence[-1])
                completed = " ".join(sentence)
                
                text.append(completed)
                sentence = []
                seed = next
            # if next in ['.','!','?']:
            #     sentence[-1] += next
                
            #     completed = " ".join(sentence)
            #     completed[0].capitalize()
            #     text.append(completed)
            #     seed = next
            #     sentence = []
            elif next in [',']:
                sentence[-1] += next
                seed = next
            else: 
                sentence.append(next)
                seed = next
        print(" ".join(text))
        return text
        
    @staticmethod
    def load_corpus(filepath:str)->str:
        corpus = ""
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Loading corpus"):
                corpus += line+' '
        
        return corpus
    
"""
NOTES:

Probability of event A given that event B has occurred. 
P(A)= Prior probability of event A 
P(B)= The probability of event B 
P(A|B) = Posterior probability: prob of A given that B has occurred.
P(B|A) = Likelihood- the probability of B given that A has occurred. 

P(A|B) = P(B|A)*P(A)
        -------------
            P(B)
            
Conditional probability:

P(B|A) = the probability of B and A both occuring divided by the probability of A 
    occurring independently:
    
    P(B|A) = P(B U A)/P(A)
"""