import pandas as pd
from math import log2
import Entropy
from EnropyTree import EntropyTree as Tree
from pyitlib import discrete_random_variable as drv

'''
Omar Hmdea 206635922
Iz Adeeb Alkoran 207101429
Reziq Abu Mdeagm 211606801
'''

class Discritization:
    '''
    equal width and equal frequency and entropy based discrezitation
    '''
    def __init__(self, data, numOfbis):
        '''
        :param data :pandas data frame to discritize
        :param numOfbis: requisted number of bins
        '''
        self.data = data
        self.numberOfbins = numOfbis

    def EqualFrequencyDiscretization(self, attr):
        '''

        :param attr:the attribute to discritize
        :return:attribute's list after discritize with equal frequency discretization
        '''
        array = list(self.data[attr].to_list())
        array.sort()
        k = self.numberOfbins
        if type(k) == int:
            lenB = len(self.data[attr].to_list()) // k
        else:
            lenB = len(self.data[attr].to_list()) // len(k)
            k = len(k)

        newDis = []
        helpm = []
        array = [float('-inf')] + array
        array.append(float('inf'))

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

                    if i == 0:
                        newDis.append(pd.Interval(left=min(helpm[i]), right=max(helpm[i])))
                    else:
                        newDis.append(pd.Interval(left=max(helpm[i-1]), right=max(helpm[i])))
                    break


        return newDis

    def EqualWidthDiscretization(self, attr):
        '''

            :param attr:the attribute to discritize
            :return:attribute's list after discritize with equal width discretization
        '''
        array = self.data[attr].to_list()
        k = self.numberOfbins
        maxa, mina = max(array), min(array)

        if type(k) == int:
            width = round((maxa - mina) / k, 3)

        else:
            width = (maxa - mina) / len(k)
            k = len(k)

        newW = 0
        newDis = []

        leftVal=0
        for i in range(k):
            if newW == 0:

                for x in array:
                    if x <= width:
                        newDis.append(pd.Interval(left=float('-inf'), right=width + mina))
                        leftVal=width + mina


            elif 0 <i < k - 1:

                for x in array:

                    if newW < x <= newW + width:

                        newDis.append(pd.Interval(left=leftVal, right=newW + width))
                        leftVal = newW + width





            else:

                for x in array:
                    if x > newW:
                        newDis.append(pd.Interval(left=leftVal, right=float('inf')))

            newW += width
        finalDis=[]

        for i in array:
            for x in newDis:
                if i in x:
                    finalDis.append(x)
                    break


        return finalDis


    def Enropy_Discretization(self, attr, Class):
        '''

               :param attr:the attribute to discritize
               :param Class:class attribute in the data frame
               :return:attribute's list after discritize with entroby based  discretization
               '''
        data = self.data
        k = len(self.numberOfbins) if type(self.numberOfbins) is list else self.numberOfbins
        tree = EntropyBased(data, attr, Class, k)
        rootEnt = tree.entropy
        bins = tree.getLevel_h()
        allSplits = tree.getNodes()
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

        for i in bins:
            allSplits.remove(i)

        splits = [x.split for x in allSplits]
        splits.sort()
        dis = [(float('-inf'), splits[0])]
        for i in range(len(splits)):
            if i == len(splits) - 1:
                dis.append((splits[i], float('inf')))
                print(splits[i])
            else:
                dis.append((splits[i], splits[i + 1]))

        dataD = data[attr].to_list()
        for i in range(len(dataD)):
            for x in dis:
                if x[0] < dataD[i] <= x[1]:
                    dataD[i] = pd.Interval(left=x[0], right=x[1])
                    break

        return dataD

    def built_Enropy_Discretization(self, attr, Class):

        '''

        :param attr:the attribute to discritize
        :param Class:class attribute in the data frame
        :return:attribute's list after discritize with entroby based discretization

        using built entroby method from pyitlib library
        '''
        data=self.data
        k = len(self.numberOfbins) if type(self.numberOfbins) is list else self.numberOfbins
        tree = built_EntropyBased(data, attr, Class, k)
        rootEnt = tree.entropy
        bins = tree.getLevel_h()
        allSplits = tree.getNodes()
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

        for i in bins:
            allSplits.remove(i)

        splits = [x.split for x in allSplits]
        splits.sort()
        dis = [(float('-inf'), splits[0])]
        for i in range(len(splits)):
            if i == len(splits) - 1:
                dis.append((splits[i], float('inf')))
            else:
                dis.append((splits[i], splits[i + 1]))

        dataD = data[attr].to_list()

        for i in range(len(dataD)):
            for x in dis:
                if x[0] <= dataD[i] <= x[1]:
                    dataD[i] = pd.Interval(left=x[0], right=x[1])
                    break

        return dataD


def EntropyBased(data, attr, Class, k):
    '''
    :param data: pandas data frame
    :param attr:the attribute to discritize
    :param Class:class attribute in the data frame
    :param k:number of buns
    :return: Entropy tree with 2^([log(k)]+1) leaves
    '''
    data = data.sort_values(by=attr)
    EntTree = Tree(data)
    depth = log2(k)

    if (int(depth) - depth) != 0:
        depth = int(depth) + 1

    for i in range(int(depth)):

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
    '''
        :param data: pandas data frame
        :param attr:the attribute to discritize
        :param Class:class attribute in the data frame
        :param k:number of buns
        :return: Entropy tree with 2^([log(k)]+1) leaves
        using built entroby method from pyitlib library
        '''
    data = data.sort_values(by=attr)
    EntTree = Tree(data)
    depth = log2(k)

    if (int(depth) - depth) != 0:
        depth = int(depth) + 1

    for i in range(int(depth)):
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
    if len(list1) == 1:
        raise Exception("can't split the data,try smaller number of bins")
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

