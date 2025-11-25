import random
from tqdm import tqdm

class Perceptron():
    """Basic perceptron from scratch"""
    def __init__(self, training_data:list, test_data:list):
        self.training_data = training_data
        self.test_data = test_data
        self.input_dim = len(training_data[0])
        self.weights = [random.uniform(-1,1) for _ in range(self.input_dim)]
        self.bias = random.uniform(-1,1)
        self.stats = {
                    'training_loss':{}
                    
                    }

    def sign(self, z):
        return 1 if z >= 0 else 0
    
    '''
    Train loop takes number of epochs to train over.
    '''
    def train(self, learning_rate:float=0.1, epochs:int=1000):
        self.stats['training_loss'] = {f'Epoch{epoch+1}':0 for epoch in range(epochs)}
        current_epoch = 0
        for epoch in tqdm(range(epochs), desc= f"Training model"):
            avg_loss = 0
            for item in self.training_data:
                vector = item[0]
                label = item[1]
                dot_product = 0
                for i in range(len(self.weights)):
                    dot_product += (vector[i]*self.weights[i])
                logit = dot_product + self.bias
                y_hat = 1 if logit >= 0 else 0
                loss = label - y_hat
                """if the loss is 1, label was pos and guess was neg
                    if the loss is -1, label was neg and guess was pos"""
                if loss != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += learning_rate*loss*vector[i]
                avg_loss += label-y_hat
            self.stats['training_loss'][f'Epoch{epoch+1}'] = avg_loss/len(self.training_data)
            current_epoch += 1

    '''
    Given a learning rate, model will continue to train until the model converges.
    '''
    def converge(self, learning_rate:float=0.1):
        loss_for_epoch = 1*len(self.training_data)
        total_epochs = 0
        converged = False
        
        while loss_for_epoch > 0:
            loss_for_epoch=0
            # run a full training epoch
            for item in self.training_data:
                vector = item[0]
                label = item[1]
                dot_product = 0
                for i in range(len(self.weights)):
                    dot_product += (vector[i]*self.weights[i])
                logit = dot_product + self.bias
                y_hat = 1 if logit >= 0 else 0
                loss = label - y_hat
                if loss != 0:
                    for i in range(len(self.weights)):
                        self.weights[i] += learning_rate*loss*vector[i]
                    self.bias += learning_rate*loss
                loss_for_epoch += abs(label-y_hat)
            total_epochs += 1
            if total_epochs > 5000:
                break
            
        converged = True if total_epochs <= 500 else False
        if converged:
            print(f"Perceptron converged after {total_epochs} epochs.")
        else:
            print("Model failed to converge.")
            

    def evaluate(self):
        correct = 0
        total = 0
        for item in tqdm(self.test_data, desc= 'Evaluating model'):
            vector = item[0]
            label = item[1]
            dot_product = 0
            for i in range(len(self.weights)):
                dot_product += (vector[i]*self.weights[i])
            logit = dot_product + self.bias
            y_hat = 1 if logit >= 0 else 0
            if y_hat == label:
                correct += 1
            total += 1
        print(f"Perceptron accuracy = {correct/total:.2f}")
