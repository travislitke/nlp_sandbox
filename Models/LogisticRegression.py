import LossFunction
import math
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
class LogisticRegression():
    def __init__(self, vectorizer):
        self.input_dim = vectorizer.dim
        self.weights = np.zeros(self.input_dim)
        self.bias = 0
        self.vectorizer = vectorizer
    def __setitem__(self):
        pass
    
    def forward(self, features):
        z = np.matmul(features,self.weights)+self.bias
        return z
    
    def back(self, gradient:float, learning_rate:float):
        gradients = [self.weights[i]*gradient for i in range(len(self.weights))]
        for idx in range(len(gradients)):
            self.weights[idx] -= gradients[idx]*learning_rate
            
            # print(f"Gradient = {learning_rate*loss*x[idx]}")
            
        
        self.bias -= (learning_rate*gradient)
                
    def clip(self,x,min=-1,max=1):
        if x <=min:
            return 1e-8
        elif x >= max:
            return 1-1e-8
        else:
            return x
    
    def sigmoid(self,z:float):
        # sigmoid = 1/(1+exp(-z))
        # print((1/(1+math.exp(-1*self.clip(z,-700,700)))))
        return (1/(1+math.exp(-1*self.clip(z,-700,700))))
    
    def train(self, data, learning_rate:float=0.1):

        total_loss = 0

        for instance in tqdm(data, desc="Training"):
            x=self.vectorizer.vectorize(instance.features) #feature vector
            l2 = math.sqrt(sum([feature**2 for feature in x]))
            x = [elem/l2 for elem in x]
            y=data.label_dict[instance.label] #label
            z=self.forward(x) #logit
            y_hat = self.sigmoid(z)
            loss = LossFunction.cross_entropy_loss(y_hat,y)
            gradient = y_hat-y
            total_loss += loss
            self.back(gradient=gradient, learning_rate=learning_rate) 

            

        print(f"Average loss: {total_loss/len(data):.4f}")    
            
    
    def test(self, data):
        correct = 0
        for instance in tqdm(data, desc='Evaluating'):
            x = self.vectorizer.vectorize(instance.features)
            y = data.label_dict[instance.label]
            z = self.forward(x)
            y_hat = self.sigmoid(z)
            predicted = 1 if y_hat >= 0.5 else 0
            
            if predicted == y:
                correct += 1
        
        print(f"Accuracy: {correct/len(data)}")
        return correct/len(data)
        
    def experiment(self, training_data, dev_data=None, test_data=None, lr=[0.1],epochs:int=10):
        best = {"learning rate":{rate:0.0 for rate in lr}
                            }
        
        
        print(f"Evaluating learning rate {0.1}")
        self.train(training_data, learning_rate=0.1)
        accuracy = self.test(test_data)
        best["learning rate"][0.1]=accuracy

        print(best)
            




    