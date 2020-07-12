from math import log2
import numpy as np


def entropy(data):
    sum = 0
    if len(data) <= 1:
        return 0
    temp = {}

    for i in data:
        if i in temp:
            temp[i] += 1
        else:
            temp[i] = 1
    for key in temp:
        p = temp[key] / len(data)
        sum += p * log2(p)
    return -1 * sum


def InfoGain(data, attr):
    total_entropy = entropy(data["class"])
    values, counts = np.unique(data[attr], return_counts=True)
    weightes = []
    for i in range(0, len(values)):
        weightes.append((counts[i] / sum(counts)) * entropy(data[data[attr] == values[i]]['class'].tolist()))
    return total_entropy - sum(weightes)


from math import log
from pyitlib import discrete_random_variable as drv

from scipy.stats import entropy as ent


def entropy1(list):
    temp = set(list)
    ent = 0
    for i in temp:
        ent += -((list.count(i) / len(list)) * log(list.count(i) / len(list), 2))
    return ent




from math import log
from pyitlib import discrete_random_variable as drv


def conditionalEntropy(list1, list2):
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



from pyitlib import discrete_random_variable as drv
from math import log


def conditionalEntropy(list1, list2):
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


def entropy1(list):
    temp = set(list)
    ent = 0
    for i in temp:
        ent += -((list.count(i) / len(list)) * log(list.count(i) / len(list), 2))
    return ent


def mutualInformation(X, Y):
    HY = entropy(Y)
    HYX = conditionalEntropy(Y, X)
    return HY - HYX




from math import log
#from info_gain import info_gain


def entropy1(list):
    temp = set(list)
    ent = 0
    for i in temp:
        ent += -((list.count(i) / len(list)) * log(list.count(i) / len(list), 2))
    return ent


def informationGain(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("the length of the lists must be equal")
    median = (min(list1) + max(list1)) / 2
    right, left = [], []
    for i in range(len(list1)):
        if list1[i] < median:
            left.append(list2[i])
        else:
            right.append(list2[i])
    perentE, leftE, rigthE = entropy(list2), entropy(left), entropy(right)
    splitE = (len(left) / len(list1)) * leftE + (len(right) / len(list1)) * rigthE
    gain = perentE - splitE
    return gain


