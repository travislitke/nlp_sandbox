
import re
from typing import Optional
from collections import Counter

class Vocabulary:
    
    def __init__(self, corpus:Optional[str]=None, num_most_common:Optional[int]=None):
        self.vocab = ['UNK']
        
        if corpus:
            tokens = []
            for token in corpus.split():
                cleaned = re.sub(r'[,.!?]','',token)
                if cleaned.lower() not in tokens:
                    tokens.append(cleaned.lower())
            tokens.sort()
            self.vocab.extend(tokens)
                
    def __str__(self):
        return str(self.vocab)