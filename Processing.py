import pickle
import NaiveBayesClassifier
import ID3
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from Discretization import Discritization

class PreProcessing():
    '''
    class that do pre processing
    '''
    def  Is_Empty(self,df):
        '''
        check if data frame is empty if yes return True else False
        :param df: a data frame
        :return: nothing
        '''
        return df.empty

    def Delete_Nan_Class_Row(self,df):
        '''
        delete the rows that contain class empty value.
        :param df: a data frame
        :return: nothing
        '''
        df.dropna(subset=['class'], inplace=True)

    def Fill_Nan_Values(self,df,Structure):
        '''
        function that fill the nan values.
        if the values of the column is  numeric so we fill the average of the values
        else we fill the most frequent value of the values.
        :param df: data frame
        :param Structure: dict that conatin the struct of the data frame
        :return:nothing
        '''
        nan_columns = df.columns[df.isna().any()].tolist()
        for col in nan_columns:
            if Structure[col] == 'NUMERIC':
                df[col].fillna(int(df[col].mean()), inplace=True)
            else:
                Most_Frequent=df[col].value_counts().idxmax()
                df[col].fillna(Most_Frequent, inplace=True)

    def Normalize(self,df,column):
        '''
        normalize the values of a given column name
        :param df: data frame
        :param column: the name of the column that we want to normalize her values
        :return:nothing
        '''
        c = np.array([df[column].tolist()])
        df[column] = (list(preprocessing.normalize(c)[0]))

    def Discretization(self,df,Structure,Discretization_type,NumOfBins):
        '''

        :param df: data frame
        :param Structure: dict that conatin the struct of the data frame
        :param Discretization_type: the type of the wanted discretization
        :param NumOfBins: number of bins for the discretization .
        :return: nothing
        '''
        if Discretization_type !='Without Discretization':
            discrize=Discritization(df,NumOfBins)
            Disc={'Equal frequency':discrize.EqualFrequencyDiscretization,
                  'Equal width':discrize.EqualWidthDiscretization,
                  'Based entropy':discrize.Enropy_Discretization}
            for key in  Structure:
                if Structure[key] == 'NUMERIC':
                    #df[key] = pd.qcut(df[key], NumOfBins, duplicates='drop')
                    if Discretization_type =='Based entropy':
                        df[key]=Disc['Based entropy'](key,'class')
                    else:
                        df[key] = Disc[Discretization_type](key)





    def read_structure(self,FileName):
        '''
        the function return a dict that conatin the struct of the data frame by reading the structure file .
        :param FileName: the file name of the structure file.
        :return: a dict that conatin the struct of the data frame
        '''
        struct = pd.read_csv(FileName, sep=' ', names=['type', 'feature', 'data'])
        values = {}
        for i in range(0, struct.shape[0]):
            row = struct.iloc[i].tolist()
            x = (row[2].split(','))
            values[row[1]] = set([i.replace('}', '').replace('{', '') for i in x]) if len(x) > 1 else x[0]
        return values

    def Clean_Data(self,df,Strcture,Discretization_type,NumOfBins):
        '''
        take a data frame and do for it binnig and remove the nan class rows and fill the nan values .
        :param df: data frame
        :param Structure: dict that conatin the struct of the data frame
        :param Discretization_type: the type of the wanted discretization
        :param NumOfBins: number of bins for the discretization .
        :return: nothing
        '''
        if self.Is_Empty(df):
            raise Exception("Empty File")

        self.Delete_Nan_Class_Row(df)
        self.Fill_Nan_Values(df,Strcture)
        self.Discretization(df,Strcture,Discretization_type,NumOfBins)

    def Save_Data(self,df,Path,FileName):
        '''
        save a data frame as a clean file in the give the file.
        :param df: data frame
        :param Path: the path of the files
        :param FileName: the file name of the file that we want to save
        :return: noting
        '''
        df.to_csv(Path+'/'+FileName+'_clean.csv')


class Processing(PreProcessing):
    '''
    this class for running the algorithm (our implementaion) and for saving the model and building the model.
    '''
    def Build_Model(self,Path,Algorithm,train):
        '''

        :param Path: the path of the files
        :param Algorithm: which algorithm we want to build the model for it
        :param train: data frame that we want to build the model for it.
        :return: a model
        '''
        if Algorithm == 'naive bayes classifier (our)':
            return NaiveBayesClassifier.NaiveBayesClassifier(Path,train)

        if Algorithm == 'ID3 (our)':
            return ID3.ID3(Path,train)

    def Save_Model(self,Path,model):
        '''
        save the given model in the given the path as sav file  by the library pickle
        :param Path: the path of the files
        :param model: the model that we want to save
        :return: nothing
        '''
        filename = Path+'/model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def Load_Model(self,Path):
        '''
        load the model that are in the given path and returns it by the library pickle
        :param Path: the path of the files
        :return: the model
        '''
        filename =Path+'/model.sav'
        return pickle.load(open(filename, 'rb'))

    def Running_Algorithm(self,Path,Algorithm,train,test):
        '''
        running the given algorithm for the train and test files and return the results .
        :param Path: the path of the files
        :param Algorithm: the wanted algorithm
        :param train: the train data frame
        :param test: the test data frame
        :return: a dict that contain the results for the train and the test files.
        '''
        if Algorithm == 'naive bayes classifier (our)':
            prediction_train = NaiveBayesClassifier.Testing_model(Path,self.Load_Model(Path),train)
            prediction_test  = NaiveBayesClassifier.Testing_model(Path,self.Load_Model(Path),test)

        if Algorithm == 'ID3 (our)':
            prediction_train = ID3.Testing_model(Path,self.Load_Model(Path),train)
            prediction_test  = ID3.Testing_model(Path,self.Load_Model(Path),test)

        train_targets=train['class'].tolist()
        test_targets=test['class'].tolist()

        train_results = {}
        train_results['Confusion Matrix'] = list(metrics.confusion_matrix(train_targets, prediction_train))
        report = metrics.classification_report(train_targets, prediction_train, output_dict=True)['weighted avg']
        train_results['Accuracy'] = accuracy_score(train_targets, prediction_train)
        train_results['Precision'] = report['precision']
        train_results['Recall'] = report['recall']
        train_results['F-measure'] = report['f1-score']

        test_results = {}
        test_results['Confusion Matrix'] = list(metrics.confusion_matrix(test_targets, prediction_test))
        report = metrics.classification_report(test_targets, prediction_test, output_dict=True)['weighted avg']
        test_results['Accuracy'] = accuracy_score(test_targets, prediction_test)
        test_results['Precision'] = report['precision']
        test_results['Recall'] = report['recall']
        test_results['F-measure'] = report['f1-score']

        return {'train': train_results, 'test': test_results}



class BuildAlgorithm(PreProcessing):
    '''
    this class for running the algorithms (librarys implementaion) and for saving the model and building the model.
    '''
    def Convert_Strings_To_Numbers(self,Path,Discretization_type, NumOfBins):
        '''
        convert all the strings in the data frame to numbers by a built function and clean the files.
        :param Path: the path of the files
        :return: the test and train data frame after running the function on them
        '''
        pre = PreProcessing()
        try:

            test = pd.read_csv(Path+'\\test.csv')
            train = pd.read_csv(Path+'\\train.csv')
            struct = pre.read_structure(Path+'\\Structure.txt')
        except Exception:
            raise Exception('Files dose not exist')

        if pre.Is_Empty(train) or pre.Is_Empty(test):
            raise Exception('They are at least one empty file')

        pre.Clean_Data(train, struct, Discretization_type, NumOfBins)
        pre.Clean_Data(test, struct, Discretization_type, NumOfBins)

        pre.Save_Data(train, Path, 'train')
        pre.Save_Data(test, Path, 'test')

        le = preprocessing.LabelEncoder()
        for col in struct:
            if col != 'NUMERIC':
                # Converting string labels into numbers.
                train[col]=le.fit_transform(train[col].tolist())
                test[col]=le.fit_transform(test[col].tolist())
            else:
                train[col]=list(map(str,train[col].tolist()))
                test[col]=list(map(str,test[col].tolist()))
                train[col]=le.fit_transform(train[col].tolist())
                test[col]=le.fit_transform(test[col].tolist())


        return train,test

    def Run(self,Algorithm,train,test,Path,n_neighbors=2):
        '''
        running the given algorithm for the train and test files and return the results .
        :param Path: the path of the files
        :param Algorithm: the wanted algorithm
        :param train: the train data frame
        :param test: the test data frame
        :param n_neighbors: the number of neighbors or clusters for the Knn and Kmeans algorithm.
        :return: a dict that contain the results for the train and the test files.
        '''

        train_features = train.drop('class', axis=1)
        train_targets = train['class']
        test_features = test.drop('class', axis=1)
        test_targets = test['class']
        processing = Processing()

        # build model
        if Algorithm == 'naive bayes classifier':
            model = GaussianNB()

        if Algorithm == 'ID3':
            model = DecisionTreeClassifier(criterion='entropy')

        if Algorithm == 'KNN':
            model = KNeighborsClassifier(n_neighbors=n_neighbors)


        if Algorithm != 'K-MEANS':

            # Building model
            model.fit(train_features, train_targets)

            # Saving model
            processing.Save_Model(Path,model)

            # Testing model
            prediction_train = model.predict(train_features)
            prediction_test= model.predict(test_features)

            train_results={}
            train_results['Confusion Matrix']=list(metrics.confusion_matrix(train_targets, prediction_train))
            report=metrics.classification_report(train_targets, prediction_train,output_dict=True)['weighted avg']
            train_results['Accuracy']=accuracy_score(train_targets, prediction_train)
            train_results['Precision']=report['precision']
            train_results['Recall']=report['recall']
            train_results['F-measure']=report['f1-score']

            test_results={}
            test_results['Confusion Matrix']=list(metrics.confusion_matrix(test_targets, prediction_test))
            report=metrics.classification_report(test_targets, prediction_test,output_dict=True)['weighted avg']
            test_results['Accuracy']=accuracy_score(test_targets, prediction_test)
            test_results['Precision']=report['precision']
            test_results['Recall']=report['recall']
            test_results['F-measure']=report['f1-score']

            return {'train':train_results,'test':test_results}
        else:
            info={}
            for x,y,z in [(train_features,train_targets,"train"),(test_features,test_targets,"test")]:

                x=np.array(x).astype(float)
                y=np.array(y)

                #Normalize
                scaler= preprocessing.MaxAbsScaler()
                x=scaler.fit_transform(x)

                #building the model
                kmeans = KMeans(n_clusters=n_neighbors)
                kmeans.fit(x)

                # Saving model
                processing.Save_Model(Path, kmeans)

                #testing the model
                results=[]
                for i in range(len(x)):
                    predict_me = np.array(x[i].astype(float))
                    predict_me = predict_me.reshape(-1, len(predict_me))
                    prediction = kmeans.predict(predict_me)
                    results.append(prediction[0])

                temp = {}
                temp['Confusion Matrix'] = list(metrics.confusion_matrix(y, results))
                report = metrics.classification_report(y, results, output_dict=True)['weighted avg']
                temp['Accuracy'] = accuracy_score(y, results)
                temp['Precision'] = report['precision']
                temp['Recall'] = report['recall']
                temp['F-measure'] = report['f1-score']
                info[z]=temp

            return info


