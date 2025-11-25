import math
def clip(z, min=0, max=1):
    if z <= min:
        return min+1e-15
    if z >= max:
        return max-1e-15
    else:
        return z
    
def cross_entropy_loss(predicted, actual,epsilon:float=1e-15):
    
    return -1*((actual*math.log(clip(predicted)))+((1-actual)*(math.log(1-clip(predicted)))))