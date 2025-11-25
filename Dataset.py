import numpy as np

class DataInstance():
    
    def __init__(self, features, label):
        self.features = features
        self.label = label
    
    def __str__(self):
        return f"{self.features}, {self.label}"
# end of file
class Dataset():
    
    def __init__(self,data:list[tuple]) -> None:
        self.data = []
        self.labels = []
        self.label_dict = {}
        
        for item in data:
            self.add(item)

    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data[idx]
    
    def add(self, instance):
        instance = DataInstance(instance[0],instance[1])
        self.data.append(instance)
        if instance.label not in self.labels:
            self.labels.append(instance.label)
        for i,v in enumerate(sorted(self.labels)):
            self.label_dict[v] = i
        
    def remove(self, idx):
        del(self.data[idx])
