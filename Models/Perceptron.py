

class Perceptron():
    """Basic perceptron from scratch"""
    def __init__(self, input_dim: int, bias: int = 1):
        self.weights = [0]*input_dim
        self.loss = 1
        self.bias = bias

    def sign(self, z):
        return 1 if z >= 0 else 0

    def train(self, data, learning_rate=0.1, epochs=1000):

        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.........................")
            loss_for_epoch = 0
            for item in data:
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
                loss_for_epoch += label-y_hat

            self.loss = loss_for_epoch/len(data)
            print(f"""Epoch {epoch+1} Stats:
                    Weights = {self.weights},
                    Loss = {loss_for_epoch/len(data)}""")

    def evaluate(self, data):
        correct = 0
        total = 0
        for item in data:
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
        print(f"Perceptron accuracy = {correct/total}")
