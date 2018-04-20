import urllib
import os
import pandas as pd
import numpy as np

if not os.path.exists('airfares.txt'):
    urllib.request.urlretrieve('http://www.stat.ufl.edu/~winner/data/airq4.dat',
                               'airfares.txt')
    
data = pd.read_csv('airfares.txt', sep = '\s+',
                   names=['city1', 'city2', 'pop1', 'pop2',
                          'dist', 'fare_2000', 'nb_passengers_2000',
                          'fare_2001', 'nb_passengers_2001']
                   )

data_2000 = pd.DataFrame(data.iloc[:,0:7],
                      columns = ['city1', 'city2', 'pop1', 'pop2',
                                 'dist', 'fare', 'nb_passengers', 'year'])
data_2000.iloc[:,7]=2000
data_2000.iloc[:,6]=data.iloc[:,6]
data_2000.iloc[:,5]=data.iloc[:,5]

data_2001 = pd.DataFrame(data.iloc[:,[0,1,2,3,4,7,8]],
                      columns = ['city1', 'city2', 'pop1', 'pop2',
                                 'dist', 'fare', 'nb_passengers', 'year'])

data_2001.iloc[:,7]=2001
data_2001.iloc[:,6]=data.iloc[:,6]
data_2001.iloc[:,5]=data.iloc[:,5]

