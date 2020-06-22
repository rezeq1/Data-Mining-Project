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


def Equal_width(data, k):
    arr = []
    temp = []
    w = int((max(data) - min(data)) / k)
    if (max(data) - min(data)) / k - w != 0:
        w += 1

    for i in range(0, len(data)):
        if data[i] < min(data) + w:
            temp = temp + [data[i]]
    arr.append(temp)

    for i in range(1, k - 1):
        temp = []
        for j in range(0, len(data)):
            if (min(data) + w * i) <= data[j] < (min(data) + w * (i + 1)):
                temp = temp + [data[j]]
        arr.append(temp)
    temp = []
    for i in range(0, len(data)):
        if (min(data) + w * (k - 1)) <= data[i]:
            temp = temp + [data[i]]
    arr.append(temp)
    return arr

def EntropyBased(data ,k):
    return 0

