class Vector:
    def __init__(self,data):
        self.vector = data
    
    def __str__(self):
        return str(self.vector)
    
    def shape(self):
        return (len[self.vector],len(self.vector[0]))
    
    def transpose(self):
        transposed_vector = []
        for row in self.vector:
            transposed_vector.append([i] for i in row)
        self.vector = transposed_vector
        return self.vector
    
    
        
class Matrix:
    def __init__(self, data):
        self.matrix = data




def ones(num_rows:int,num_cols:int):
    container = []
    row = [1 for _ in range(num_cols)]
    for _ in range(num_rows):
        container.append(row)
    
    if num_rows == 1 or num_cols == 1:
        return Vector(container)
    else:
        return Matrix(container)

def matrix_multiply(A, B):
    """Matrices represented by lists
    multiply row by column, add results"""

    dim_A = (len(A[0]), len(A))

    return dim_A
