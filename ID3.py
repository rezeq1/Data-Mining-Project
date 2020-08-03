import pandas as pd
import numpy as np
from Entropy import InfoGain
import random
'''
Omar Hmdea 206635922
Iz Adeeb Alkoran 207101429
Reziq Abu Mdeagm 211606801
'''

def read_structure(FileName):
    '''
    reading the structure file and return a dict that contain the structure of the file
    :param FileName: the file name of the structure file
    :return: a dict that contain the structure of the file
    '''
    struct = pd.read_csv(FileName, sep=' ', names=['type', 'feature', 'data'])
    values = {}
    for i in range(0, struct.shape[0]):
        row = struct.iloc[i].tolist()
        x = (row[2].split(','))
        values[row[1]] = set([i.replace('}', '').replace('{', '') for i in x]) if len(x) > 1 else x[0]
    return values

def Get_Decision_Tree(data, columns):
    '''
    build and return the model of the algorithm that we build from the data
    :param data: data frame that we want to build the model for it
    :param columns: the columns of the data
    :return: a model
    '''
    # Recursion stop conditions
    if len(columns) == 0:
        return np.unique(data['class'])[np.argmax(np.unique(data['class'], return_counts=True)[1])]

    if len(np.unique(data['class'])) == 1:
        return np.unique(data['class'])[0]

    # finding the biggest feature's info gain
    temp = {}
    for i in columns:
        temp[i] = InfoGain(data, i)
    max_info = max(temp, key=temp.get)

    # removing the biggest feature's info gain from the features list
    new_col = []
    for i in columns:
        if i != max_info:
            new_col.append(i)

    # Recursion to the sup trees untill arrive to leafs
    temp = {}
    for value in set(data[max_info]):
        temp[value] = Get_Decision_Tree(data[data[max_info] == value], new_col)

    return {max_info: temp}


def Classification_Row(tree, row, columns,deafult):
    '''
    return a classification of the row by the given model
    :param tree: the model
    :param row: a row that we want to classification
    :param columns: the columns names
    :param deafult: a deafult value the we return
    :return: a classification of the row
    '''
    root = list(tree.keys())[0]
    result = row[columns.index(root)]
    classification = random.choice(deafult)
    if type(result) != str:
        for intervl in tree[root]:
            if result in intervl:
                classification = tree[root][intervl]
                break
    else:
        if result in tree[root]:
            classification = tree[root][result]

    if type(classification) != dict:
        return classification
    else:
        return Classification_Row(classification,row,columns,deafult)


def ID3(Path,train):
    '''
    return and build a model on the given data frame
    :param Path: the path of the files
    :param train: the train data frame
    :return: a model
    '''
    # get the columns of test file
    columns = train.columns.tolist()

    # build the model
    columns.remove('class')
    return Get_Decision_Tree(train, columns)

def Testing_model(Path,model,test):
    '''
    return the classifications of the rows of the given data frame
    :param Path: a path that contain the files
    :param model: the model
    :param test: the data frame that we want to test the model on it
    :return:classifications of the rows of the given data frame
    '''
    # get the rows and the columns of test file
    columns = test.columns.tolist()
    class_attrs = list(set(test['class'].tolist()))
    rows = []
    for i in range(0, test.shape[0]):
        rows.append(test.iloc[i].tolist())

    # testting the model
    results=[]
    for row in rows:
        results.append(Classification_Row(model, row, columns,class_attrs))

    return results
