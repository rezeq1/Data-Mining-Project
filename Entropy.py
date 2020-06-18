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