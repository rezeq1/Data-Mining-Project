import pickle

class Processing():

    def Build_Model(self,Path,Algorithm):
        return {}

    def Save_Model(self,Path,model={'rezeq':19,'ahmad':22,'omar':21,'iz':24}):
        filename = Path+'\model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def Load_Model(self,Path):
        filename =Path+'\model.sav'
        return pickle.load(open(filename, 'rb'))
