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

n_data = pd.DataFrame(data.iloc[:,0:7],
                      columns = ['city1', 'city2', 'pop1', 'pop2',
                                 'dist', 'fare', 'nb_passengers', 'year'])
n_data.iloc[:,7]=2000
n_data.iloc[:,6]=data.iloc[:,6]
n_data.iloc[:,5]=data.iloc[:,5]
print(n_data.head(10))