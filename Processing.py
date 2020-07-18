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


class PreProcessing():

    def  Is_Empty(self,df):
        return df.empty

    def Delete_Nan_Class_Row(self,df):
        df.dropna(subset=['class'], inplace=True)

    def Fill_Nan_Values(self,df,Structure):
        nan_columns = df.columns[df.isna().any()].tolist()
        for col in nan_columns:
            if Structure[col] == 'NUMERIC':
                df[col].fillna(int(df[col].mean()), inplace=True)
            else:
                Most_Frequent=df[col].value_counts().idxmax()
                df[col].fillna(Most_Frequent, inplace=True)

    def Normalize(self,df,column):
        c = np.array([df[column].tolist()])
        df[column] = (list(preprocessing.normalize(c)[0]))

    def Discretization(self,df,Structure,Discretization_type,NumOfBins):
        if Discretization_type !='Without Discretization':
            for key in  Structure:
                if Structure[key] == 'NUMERIC':
                    df[key] = pd.qcut(df[key], NumOfBins, duplicates='drop')

    def read_structure(self,FileName):
        struct = pd.read_csv(FileName, sep=' ', names=['type', 'feature', 'data'])
        values = {}
        for i in range(0, struct.shape[0]):
            row = struct.iloc[i].tolist()
            x = (row[2].split(','))
            values[row[1]] = set([i.replace('}', '').replace('{', '') for i in x]) if len(x) > 1 else x[0]
        return values

    def Clean_Data(self,df,Strcture,Discretization_type,NumOfBins):
        if self.Is_Empty(df):
            raise Exception("Empty File")

        self.Delete_Nan_Class_Row(df)
        self.Fill_Nan_Values(df,Strcture)
        self.Discretization(df,Strcture,Discretization_type,NumOfBins)

    def Save_Data(self,df,Path,FileName):
        df.to_csv(Path+'\\'+FileName+'_clean.csv')


class Processing():

    def Build_Model(self,Path,Algorithm,train):
        if Algorithm == 'naive bayes classifier':
            return NaiveBayesClassifier.NaiveBayesClassifier(Path,train)

        if Algorithm == 'ID3':
            return ID3.ID3(Path,train)

    def Save_Model(self,Path,model):
        filename = Path+'\model.sav'
        pickle.dump(model, open(filename, 'wb'))

    def Load_Model(self,Path):
        filename =Path+'\model.sav'
        return pickle.load(open(filename, 'rb'))

    def Running_Algorithm(self,Path,Algorithm,train,test):

        if Algorithm == 'naive bayes classifier':
            prediction_train = NaiveBayesClassifier.Testing_model(Path,self.Load_Model(Path),train)
            prediction_test  = NaiveBayesClassifier.Testing_model(Path,self.Load_Model(Path),test)

        if Algorithm == 'ID3':
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



class BuildAlgorithm():
    def Convert_Strings_To_Numbers(self,Path):
        pre = PreProcessing()
        try:
            test = pd.read_csv(Path+'\\test.csv')
            train = pd.read_csv(Path+'\\train.csv')
            struct = pre.read_structure(Path+'\\Structure.txt')
        except Exception:
            raise Exception('Files dose not exist')

        if pre.Is_Empty(train) or pre.Is_Empty(test):
            raise Exception('They are at least one empty file')

        pre.Delete_Nan_Class_Row(train)
        pre.Fill_Nan_Values(train, struct)
        pre.Delete_Nan_Class_Row(test)
        pre.Fill_Nan_Values(test, struct)

        le = preprocessing.LabelEncoder()
        for col in struct:
            if col != 'NUMERIC':
                # Converting string labels into numbers.
                train[col]=le.fit_transform(train[col].tolist())
                test[col]=le.fit_transform(test[col].tolist())

        return train,test

    def Run(self,Algorithm,train,test,Path,n_neighbors=2):

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
                kmeans = KMeans(n_clusters=n_neighbors, max_iter=600, algorithm='auto')
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

'''
model=process.Build_Model(Path,'naive bayes classifier',train)
process.Save_Model(Path,model)
train = pd.read_csv('train.csv')
pre.Delete_Nan_Class_Row(train)
pre.Fill_Nan_Values(train,struct)
process.Running_Algorithm(Path,'naive bayes classifier',train,test)
'''
