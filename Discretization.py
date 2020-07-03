import pandas as pd


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

        def in_(self,x):
            return self.min<=x<=self.max

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
                    #newDis.append(f'({min(helpm[i])},{max(helpm[i])})' if not helpList else helpList[i])
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
                    #newDis.append(f'({min(helpm[i])},{max(helpm[i])})' if not helpList else helpList[i])
                    newDis.append(self.interval(min(arr[i]), max(arr[i])))

                    break
        return newDis

    def pandas_cut(self):
        return pd.cut(self.data, self.numberOfbins)

    def pandas_qcut(self):
        return pd.qcut(self.data, self.numberOfbins)


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


def EntropyBased(data, k):
    return 0


x = Discritization([1, 9, 3, 4, 5, 67, 7, 8, 9],3)
print(x.Equal_width())
print(x.EqualFrequencyDiscretization())
#print(x.pandas_cut())
#print(x.pandas_qcut())
