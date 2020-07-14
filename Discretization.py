import pandas as pd
from math import log2
import Entropy
from EnropyTree import EntropyTree as Tree
from pyitlib import discrete_random_variable as drv

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

    def EqualFrequencyDiscretization(self, attr):
        array = list(self.data[attr].to_list())
        array.sort()
        helpList = None
        k = self.numberOfbins
        if type(k) == int:
            lenB = len(self.data[attr].to_list()) // k
        else:
            lenB = len(self.data[attr].to_list()) // len(k)
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
        for x in self.data[attr].to_list():
            for i in range(len(helpm)):
                if x in helpm[i]:
                    # newDis.append(f'({min(helpm[i])},{max(helpm[i])})' if not helpList else helpList[i])
                    newDis.append(self.interval(min(helpm[i]), max(helpm[i])))

                    break

        return newDis

    def Equal_width(self, attr):
        arr = []
        temp = []
        k = self.numberOfbins
        w = int((max(self.data[attr].to_list()) - min(self.data[attr].to_list())) / k)
        if (max(self.data[attr].to_list()) - min(self.data[attr].to_list())) / k - w != 0:
            w += 1

        for i in range(0, len(self.data[attr].to_list())):
            if self.data[attr].to_list()[i] < min(self.data[attr].to_list()) + w:
                temp = temp + [self.data[attr].to_list()[i]]
        arr.append(temp)

        for i in range(1, k - 1):
            temp = []
            for j in range(0, len(self.data[attr].to_list())):
                if (min(self.data[attr].to_list()) + w * i) <= self.data[attr].to_list()[j] < (
                        min(self.data[attr].to_list()) + w * (i + 1)):
                    temp = temp + [self.data[attr].to_list()[j]]
            arr.append(temp)
        temp = []
        for i in range(0, len(self.data[attr].to_list())):
            if (min(self.data[attr].to_list()) + w * (k - 1)) <= self.data[attr].to_list()[i]:
                temp = temp + [self.data[attr].to_list()[i]]
        arr.append(temp)
        newDis = []
        for x in self.data[attr].to_list():
            for i in range(len(arr)):
                if x in arr[i]:
                    # newDis.append(f'({min(helpm[i])},{max(helpm[i])})' if not helpList else helpList[i])
                    newDis.append(self.interval(min(arr[i]), max(arr[i])))

                    break
        return newDis

    def pandas_cut(self, attr):
        return pd.cut(self.data[attr].to_list(), self.numberOfbins)

    def pandas_qcut(self, attr):
        return pd.qcut(self.data[attr].to_list(), self.numberOfbins)

    def Enropy_Discretization(self, attr, Class):
        data = self.data
        k = len(self.numberOfbins) if type(self.numberOfbins) is list else self.numberOfbins
        tree = EntropyBased(data, attr, Class, k)
        rootEnt = tree.entropy
        bins = tree.getLevel_h()
        finalBins = []

        bin = 0
        while bin < k:
            gain = 0
            node = None
            helpBins = list(finalBins)
            for i in bins:
                right = i.getRight()
                left = i.getLeft()
                helpBins.append(left)
                helpBins.append(right)
                for x in bins:
                    if not (x is i):
                        helpBins.append(x)
                infoD = 0
                for x in helpBins:
                    infoD += (len(x.data[attr].to_list()) / len(data[attr].to_list())) * x.entropy
                gainD = rootEnt - infoD
                if gainD >= gain:
                    gain = gainD
                    node = i
            finalBins.append(node.left)
            finalBins.append(node.right)
            bins.remove(node)
            if len(bins + finalBins) == k:
                bin = k
                finalBins = bins + finalBins
            bin += 1

        dis = [(min(x.data[attr].to_list()), max(x.data[attr].to_list())) for x in finalBins]
        dataD = data[attr].to_list()
        lastBins = []
        for i in range(len(dataD)):
            for x in dis:
                if x[0] <= dataD[i] <= x[1]:
                    dataD[i] = self.interval(x[0], x[1], self.numberOfbins[dis.index(x)] if type(
                        self.numberOfbins) is list else None)
                    break

        return dataD

    def built_Enropy_Discretization(self, attr, Class):

        data = pd.DataFrame({attr:self.data[attr].to_list(),Class:convertStringToNum(self.data[Class].to_list())})
        k = len(self.numberOfbins) if type(self.numberOfbins) is list else self.numberOfbins
        tree = built_EntropyBased(data, attr, Class, k)
        rootEnt = tree.entropy
        bins = tree.getLevel_h()
        finalBins = []

        bin = 0
        while bin < k:
            gain = 0
            node = None
            helpBins = list(finalBins)
            for i in bins:
                right = i.getRight()
                left = i.getLeft()
                helpBins.append(left)
                helpBins.append(right)
                for x in bins:
                    if not (x is i):
                        helpBins.append(x)
                infoD = 0
                for x in helpBins:
                    infoD += (len(x.data[attr].to_list()) / len(data[attr].to_list())) * x.entropy
                gainD = rootEnt - infoD
                if gainD >= gain:
                    gain = gainD
                    node = i
            finalBins.append(node.left)
            finalBins.append(node.right)
            bins.remove(node)
            if len(bins + finalBins) == k:
                bin = k
                finalBins = bins + finalBins
            bin += 1

        dis = [(min(x.data[attr].to_list()), max(x.data[attr].to_list())) for x in finalBins]
        dataD = data[attr].to_list()
        lastBins = []
        for i in range(len(dataD)):
            for x in dis:
                if x[0] <= dataD[i] <= x[1]:
                    dataD[i] = self.interval(x[0], x[1], self.numberOfbins[dis.index(x)] if type(
                        self.numberOfbins) is list else None)
                    break

        return dataD



def EntropyBased(data, attr, Class, k):
    data = data.sort_values(by=attr)
    EntTree = Tree(data)
    split = [bestSplitPoint(data, attr, Class)]
    # bins = [data.loc[data[attr] <= split[0][0]], data.loc[data[attr] > split[0][0]]]
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
    leafs = EntTree.getLeafs()
    for i in leafs:
        i.setEntropy(Entropy.entropy(i.getRoot()[Class].to_list()))

    return EntTree

def built_EntropyBased(data, attr, Class, k):
    data = data.sort_values(by=attr)
    EntTree = Tree(data)
    split = [bestSplitPoint(data, attr, Class)]
    # bins = [data.loc[data[attr] <= split[0][0]], data.loc[data[attr] > split[0][0]]]
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
    leafs = EntTree.getLeafs()
    for i in leafs:
        i.setEntropy(drv.entropy(i.getRoot()[Class].to_list()))

    return EntTree


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

def built_bestSplitPoint(data, attr, Class, gainD=None):
    '''
    :param data:sorted data frame by the attr
    :param attr:the data frame column that you need to split
    :param Class:class column
    :return:best split point and the gain
    '''
    list1 = data[attr].to_list()
    if len(list1) == 1:
        return (list1[0], 0)
    entropyD = drv.entropy(data[Class].to_list())
    bestS = (list1[0] + list1[1]) / 2
    firstS = data.loc[data[attr] <= bestS]
    lastS = data.loc[data[attr] > bestS]
    if gainD is None:
        infoD = (len(firstS[attr]) / len(list1)) * drv.entropy(firstS[Class].to_list()) + (
                len(lastS[attr]) / len(list1)) * drv.entropy(lastS[Class].to_list())
        gainD = entropyD - infoD

    for i in range(1, len(list1) - 1):
        best = (list1[i] + list1[i + 1]) / 2
        firstS = data.loc[data[attr] <= best]
        lastS = data.loc[data[attr] > best]
        infoD = (len(firstS[attr]) / len(list1)) * drv.entropy(firstS[Class].to_list()) + (
                len(lastS[attr]) / len(list1)) * drv.entropy(lastS[Class].to_list())
        gain = entropyD - infoD
        if gain >= gainD:
            bestS = best
            gainD = gain

    return (bestS, entropyD)
def convertStringToNum(list1):
    sett=list(set(list1))
    dict1={}
    for i in sett:
        dict1[i]=sett.index(i)
    for i in range(len(list1)):
        list1[i]=dict1[list1[i]]
    return list1



data = {"attr": [4, 8, 5, 12, 15, 1, 2, 3, 4, 5], 'class': ['N', 'N', 'Y', 'Y', 'Y'] * 2}
df = pd.DataFrame(data=data)

x = Discritization(df, 4)
#x=x.built_Enropy_Discretization('attr','class')
print(x.built_Enropy_Discretization('attr','class'))

