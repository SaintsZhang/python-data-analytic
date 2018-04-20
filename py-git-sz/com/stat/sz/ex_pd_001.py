import urllib
import os
import pandas as pd
import numpy as np
import seaborn as sbn
import matplotlib.pyplot as plt

def Init_Data(filename,url):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url,filename)
        
    data = pd.read_csv(filename, sep = '\s+',
                       names=['city1', 'city2', 'pop1', 'pop2',
                              'dist', 'fare_2000', 'nb_passengers_2000',
                              'fare_2001', 'nb_passengers_2001']
                       )
    return data
def Prep_Data(data):
    data_2000 = pd.DataFrame(data.iloc[:,0:7],
                          columns = ['city1', 'city2', 'pop1', 'pop2',
                                     'dist', 'fare', 'nb_passengers', 'year'])
    data_2000.iloc[:,7]=2000
    #nb passengers 2000
    data_2000.iloc[:,6]=data.iloc[:,6]
    #fare 2000
    data_2000.iloc[:,5]=data.iloc[:,5]
    
    data_2001 = pd.DataFrame(data.iloc[:,[0,1,2,3,4,7,8]],
                          columns = ['city1', 'city2', 'pop1', 'pop2',
                                     'dist', 'fare', 'nb_passengers', 'year'])
    
    data_2001.iloc[:,7]=2001
    #nb passengers 2001
    data_2001.iloc[:,6]=data.iloc[:,8]
    #fare 2001
    data_2001.iloc[:,5]=data.iloc[:,7]
    
    data_flat = pd.concat([data_2000,data_2001])
    return data_flat
def Visu_Data(df):
    sbn.pairplot(df, vars=['fare', 'dist', 'nb_passengers'],
                 kind='reg', markers='.')
    plt.show()
def main():
    df        = Init_Data('airfares.txt', 'http://www.stat.ufl.edu/~winner/data/airq4.dat')
    data_flat = Prep_Data(df)
    Visu_Data(data_flat)
if __name__ == '__main__':
    main()