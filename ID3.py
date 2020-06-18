import pandas as pd
import numpy as np
from Entropy import InfoGain

def read_structure(FileName):
    struct = pd.read_csv(FileName, sep=' ', names=['type', 'feature', 'data'])
    values = {}
    for i in range(0, struct.shape[0]):
        row = struct.iloc[i].tolist()
        x = (row[2].split(','))
        values[row[1]] = set([i.replace('}', '').replace('{', '') for i in x]) if len(x) > 1 else x[0]
    return values

def Get_Decision_Tree(data, columns):
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


def Classification_Row(tree, row, columns):

    root = list(tree.keys())[0]
    result = row[columns.index(root)]
    classification = None
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
        return Classification_Row(classification,row,columns)


def ID3(Test_File,Train_File,Structure_File,NumOfBins):
    # Load files
    test = pd.read_csv(Test_File)
    train = pd.read_csv(Train_File)
    struct = read_structure(Structure_File)

    # get the rows and the columns of test file
    columns = test.columns.tolist()
    rows = []
    for i in range(0, test.shape[0]):
        rows.append(test.iloc[i].tolist())

    # fill nan values
    nan_columns = train.columns[train.isna().any()].tolist()
    for col in nan_columns:
        train[col] = train[col].fillna(method='ffill')

    nan_columns = test.columns[test.isna().any()].tolist()
    for col in nan_columns:
        test[col] = test[col].fillna(method='ffill')

    # Discretization
    for col in columns:
        if struct[col] == 'NUMERIC':
            train[col] = pd.qcut(train[col], NumOfBins, duplicates='drop')
    print('build the model')
    # build the model
    columns.remove('class')
    tree = Get_Decision_Tree(train, columns)
    print('testting the model')
    # testting the model
    columns.append('class')
    wrongs = 0
    for row in rows:
        result = Classification_Row(tree, row, columns)
        if result != row[-1]:
            wrongs += 1

    # showing info
    print("Number of wrongs:{0}  From Total:{1}".format(wrongs, len(rows)))
    print("Accuracy:{:.2f}%".format(float((len(rows) - wrongs) / len(rows) * 100)))

