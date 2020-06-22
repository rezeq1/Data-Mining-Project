from sklearn import preprocessing
import numpy as np


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

    def Discretization(self,df,Structure,method,NumOfBins):
        for key in  Structure:
            if Structure[key] == 'NUMERIC':
                df[key] = pd.qcut(df[key], NumOfBins, duplicates='drop')

    def Clean_Data(self,df,Strcture,method,NumOfBins):
        if self.Is_Empty(df):
            raise Exception("Empty File")

        self.Delete_Nan_Class_Row(df)
        self.Fill_Nan_Values(df)
        self.Discretization(df,Strcture,method,NumOfBins)

    def Save_Data(self,df,Path,FileName):
        df.to_csv(Path+FileName+'_clean.csv')


import pandas as pd

a=PreProcessing()
x=pd.DataFrame({'r':[np.nan,2,3,4],'names':['rezeq',np.nan,'omar',np.nan],'class':[True,False,np.nan,True]},columns=['r','names','class'])
print(x)
a.Delete_Nan_Class_Row(x)
print(x)
a.Fill_Nan_Values(x,{'r':'NUMERIC','names':'Attrs','class':'Attrs'})
print(x)
'''
a.Normalize(x,'r')
print(x)
a.Discretization(x,{'r':'NUMERIC','names':'Attrs','class':'Attrs'},lambda x:1,2)
print(x)
'''
path='C:\\Users\\rezeq\Desktop\\'
a.Save_Data(x,path,'train')

