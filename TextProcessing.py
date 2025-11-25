import os
'''
From a list of instances, write strings to file.
Parameters:
data (list): list of text data items.
output_file (str): file to write to. 

'''
def build_corpus(data:list,output_filepath:str, enc:str='utf-8'):
    try:
        with open(file=output_filepath, mode='x',encoding=enc) as file:
            print(file)
    
    except FileExistsError:
        print("File exists")
        
