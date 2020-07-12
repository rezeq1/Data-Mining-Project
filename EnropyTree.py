class EntropyTree:
    def __init__(self, data, left=None, right=None, split=None,entropy=None):
        self.data = data
        self.split = split
        self.entropy=entropy
        self.right = right
        self.left = left

    def getRoot(self):
        return self.data

    def getSplit(self):
        return self.split

    def getLeft(self):
        return self.left

    def getRight(self):
        return self.right

    def getLeafs(self):
        if self.isLeaf():
            return [self]
        return self.left.getLeafs() + self.right.getLeafs()

    def getNodes(self):
        if self.right.isLeaf() and self.left.isLeaf():
            return [self]
        return self.left.getNodes() + [self] + self.right.getNodes()

    def getLevel_h(self):
        if self.right.isLeaf() and self.left.isLeaf():
            return [self]
        return self.left.getLevel_h() + self.right.getLevel_h()

    def setLeft(self, node):
        self.left = node

    def setRight(self, node):
        self.right = node

    def setSplit(self, split):
        self.split = split

    def setEntropy(self,Entropy):
        self.entropy=Entropy

    def isLeaf(self):
        return self.left is None and self.right is None


x = EntropyTree(1)
y = EntropyTree(1)
z = EntropyTree(2, x, y)
y = z.getLevel_h()
