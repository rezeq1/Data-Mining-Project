class EntropyTree:
    '''
    tree contain data frame and
    the split point to split the data to right and lift
    thr entropy of the data and
    left right nodes
    '''
    def __init__(self, data, left=None, right=None, split=None,entropy=None):
        '''

        :param data:pandas data frame
        :param left: left node
        :param right: right node
        :param split: the split point
        :param entropy: the entropy of the data
        '''

        self.data = data
        self.split = split
        self.entropy=entropy
        self.right = right
        self.left = left

    def getRoot(self):
        '''

        :return: the root of the tree
        '''
        return self.data

    def getSplit(self):
        '''

        :return: data split point
        '''
        return self.split

    def getLeft(self):
        '''

        :return: left node
        '''
        return self.left

    def getRight(self):
        '''

        :return: right node
        '''
        return self.right

    def getLeafs(self):
        '''

        :return: all leaves
        '''
        if self.isLeaf():
            return [self]
        return self.left.getLeafs() + self.right.getLeafs()

    def getNodes(self):
        '''

        :return: all the nodes without leaves
        '''
        if self.right.isLeaf() and self.left.isLeaf():
            return [self]
        return self.left.getNodes() + [self] + self.right.getNodes()

    def getLevel_h(self):
        '''

        :return: the nodes in level h-1
        '''
        if self.right.isLeaf() and self.left.isLeaf():
            return [self]
        return self.left.getLevel_h() + self.right.getLevel_h()

    def setLeft(self, node):
        '''

        :param node: EntropyTree object to set it on left node
        '''
        self.left = node

    def setRight(self, node):
        '''
        :param node: EntropyTree object to set it on right node
        '''
        self.right = node

    def setSplit(self, split):
        '''
        :param split: split point of the data

        '''
        self.split = split

    def setEntropy(self,Entropy):
        '''

        :param Entropy: entropy of the data
        :return:
        '''
        self.entropy=Entropy

    def isLeaf(self):
        '''

        :return: true if the node is leaf
        '''
        return self.left is None and self.right is None


