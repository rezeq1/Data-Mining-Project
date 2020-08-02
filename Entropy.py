from math import log
import numpy as np


def entropy(list):
    '''

    :param list: list of data
    :return: entropy of the list
    '''
    temp = set(list)
    ent = 0
    for i in temp:
        ent += -((list.count(i) / len(list)) * log(list.count(i) / len(list),2))
    return ent




def InfoGain(data, attr):
    '''
    :param data:dictionary data
    :param attr:attribute of the wanted data
    :return:information gain of the attr according to thw class
    '''
    total_entropy = entropy(data["class"].tolist())
    values, counts = np.unique(data[attr], return_counts=True)
    weightes = []
    for i in range(0, len(values)):
        weightes.append((counts[i] / sum(counts)) * entropy(data[data[attr] == values[i]]['class'].tolist()))
    return total_entropy - sum(weightes)



def conditionalEntropy(list1, list2):
    '''

    :param list1:list of data
    :param list2: list of data
    :return: the conditional entropy of the lists
    '''
    if len(list1) != len(list2):
        raise ValueError("the length of the lists must be equal")

    temp1 = []
    for i in range(len(list1)):
        temp1.append((list1[i], list2[i]))

    temp2 = set(temp1)
    ent = 0
    for i in temp2:
        ent += -((temp1.count(i) / len(temp1)) * log((temp1.count(i) / len(temp1)) / (list2.count(i[1]) / len(list1)),
                                                     2))
    return ent




def mutualInformation(X, Y):
    '''

    :param X: list of data
    :param Y: list of data
    :return: mutual information of the lists
    '''
    HY = entropy(Y)
    HYX = conditionalEntropy(Y, X)
    return HY - HYX




