import pandas as pd
from math import log2
import Entropy
from EnropyTree import EntropyTree as Tree


class Discritization:
    class interval:
        def __init__(self, min, max, binNumber=None):
            self.min = min
            self.max = max
            self.binNumber = binNumber

        def __iter__(self, X):
            return self

        def __str__(self):
            return f'[{self.min},{self.max}]' if self.binNumber is None else f'{self.binNumber}'

        def __repr__(self):
            return f'[{self.min},{self.max}]' if self.binNumber is None else f'{self.binNumber}'

        def in_(self, x):
            return self.min <= x <= self.max

    def __init__(self, data, numOfbis):
        self.data = data
        self.numberOfbins = numOfbis

    def EqualFrequencyDiscretization(self):
        array = list(self.data)
        array.sort()
        helpList = None
        k = self.numberOfbins
        if type(k) == int:
            lenB = len(self.data) // k
        else:
            lenB = len(self.data) // len(k)
            helpList = k
            k = len(k)

        newDis = []
        helpm = []
        for i in range(k):
            temp = []
            for j in range(lenB * i, lenB * i + lenB):
                temp.append(array.pop(0))
            helpm.append(temp)
        if len(array) > 0:
            for x in array:
                helpm[k - 1].append(x)
        for x in self.data:
            for i in range(len(helpm)):
                if x in helpm[i]:
                    # newDis.append(f'({min(helpm[i])},{max(helpm[i])})' if not helpList else helpList[i])
                    newDis.append(self.interval(min(helpm[i]), max(helpm[i])))

                    break

        return newDis

    def Equal_width(self):
        arr = []
        temp = []
        k = self.numberOfbins
        w = int((max(self.data) - min(self.data)) / k)
        if (max(self.data) - min(self.data)) / k - w != 0:
            w += 1

        for i in range(0, len(self.data)):
            if self.data[i] < min(self.data) + w:
                temp = temp + [self.data[i]]
        arr.append(temp)

        for i in range(1, k - 1):
            temp = []
            for j in range(0, len(self.data)):
                if (min(self.data) + w * i) <= self.data[j] < (min(self.data) + w * (i + 1)):
                    temp = temp + [self.data[j]]
            arr.append(temp)
        temp = []
        for i in range(0, len(self.data)):
            if (min(self.data) + w * (k - 1)) <= self.data[i]:
                temp = temp + [self.data[i]]
        arr.append(temp)
        newDis = []
        for x in self.data:
            for i in range(len(arr)):
                if x in arr[i]:
                    # newDis.append(f'({min(helpm[i])},{max(helpm[i])})' if not helpList else helpList[i])
                    newDis.append(self.interval(min(arr[i]), max(arr[i])))

                    break
        return newDis

    def pandas_cut(self):
        return pd.cut(self.data, self.numberOfbins)

    def pandas_qcut(self):
        return pd.qcut(self.data, self.numberOfbins)

    def Enropy_Discretization(self):
        return 0


def Equal_frequency(data, k):
    arr = []
    help = data.copy()
    help.sort()

    size = int(len(data) / k)
    if len(data) / k - size != 0:
        size += 1

    for i in range(0, k - 1):
        temp = []
        flag = 0
        for j in range(i * size, (i + 1) * size):
            if j >= len(data):
                flag = -1
            if flag == 0:
                temp = temp + [help[j]]
        arr.append(temp)
    temp = []
    for i in range((k - 1) * size, len(data)):
        temp = temp + [help[i]]
    arr.append(temp)
    return arr


def EntropyBased(data, attr, Class, k):
    data = data.sort_values(by=attr)
    EntTree = Tree(data)
    split = [bestSplitPoint(data, attr, Class)]
    bins = [data.loc[data[attr] <= split[0][0]], data.loc[data[attr] > split[0][0]]]
    finalSplit = []
    depth = log2(k)

    if (int(depth) - depth) != 0:
        depth = int(depth) + 1

    for i in range(int(depth)):
        helpBin = []
        tree = EntTree
        bins = EntTree.getLeafs()
        for b in bins:
            data = b.getRoot()
            split = bestSplitPoint(data, attr, Class)
            b.setSplit(split[0])
            b.setEntropy(split[1])
            b.setLeft(Tree(data.loc[data[attr] <= split[0]]))
            b.setRight(Tree(data.loc[data[attr] > split[0]]))
    leafs=EntTree.getLeafs()
    for i in leafs:
        i.setEntropy(Entropy.entropy(i.getRoot()[Class].to_list()))

    return EntTree

def Enropy_Discretization(data, attr, Class, k):
    tree=EntropyBased(data,attr,Class,k)
    rootEnt=tree.entropy
    bins=tree.getLevel_h()
    finalBins=[]

    bin=0
    while bin<k:
        gain=0
        node=None
        helpBins = list(finalBins)
        for i in bins:
            right=i.getRight()
            left=i.getLeft()
            helpBins.append(left)
            helpBins.append(right)
            for x in bins:
                if not (x is i):
                    helpBins.append(x)
            infoD=0
            for x in helpBins:
                infoD+=(len(x.data[attr].to_list())/len(data[attr].to_list()))*x.entropy
            gainD=rootEnt-infoD
            if gainD>=gain:
                gain=gainD
                node=i
        finalBins.append(node.left)
        finalBins.append(node.right)
        bins.remove(node)
        if len(bins+finalBins)==k:
            bin=k
            finalBins= bins+finalBins
        bin+=1
    return tuple([(min(x.data[attr].to_list()),max(x.data[attr].to_list())) for x in finalBins])






def getMaxGain(list):
    maxG = list[0]
    for i in list:
        if i[1] >= maxG[1]:
            maxG = i
    list.remove(maxG)
    return maxG


def bestSplitPoint(data, attr, Class, gainD=None):
    '''
    :param data:sorted data frame by the attr
    :param attr:the data frame column that you need to split
    :param Class:class column
    :return:best split point and the gain
    '''
    list1 = data[attr].to_list()
    if len(list1) == 1:
        return (list1[0], 0)
    entropyD = Entropy.entropy(data[Class].to_list())
    bestS = (list1[0] + list1[1]) / 2
    firstS = data.loc[data[attr] <= bestS]
    lastS = data.loc[data[attr] > bestS]
    if gainD is None:
        infoD = (len(firstS[attr]) / len(list1)) * Entropy.entropy(firstS[Class].to_list()) + (
                len(lastS[attr]) / len(list1)) * Entropy.entropy(lastS[Class].to_list())
        gainD = entropyD - infoD

    for i in range(1, len(list1) - 1):
        best = (list1[i] + list1[i + 1]) / 2
        firstS = data.loc[data[attr] <= best]
        lastS = data.loc[data[attr] > best]
        infoD = (len(firstS[attr]) / len(list1)) * Entropy.entropy(firstS[Class].to_list()) + (
                len(lastS[attr]) / len(list1)) * Entropy.entropy(lastS[Class].to_list())
        gain = entropyD - infoD
        if gain >= gainD:
            bestS = best
            gainD = gain

    return (bestS, entropyD)


data = {"attr": [4, 8, 5, 12, 15, 1, 2, 3, 4, 5], 'class': ['N', 'N', 'Y', 'Y', 'Y'] * 2}
df = pd.DataFrame(data=data)

x = Enropy_Discretization(df, 'attr', 'class', 4)
print(x)