import os
import shutil
from os.path import join, getsize, splitext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dir_dtl = []

#print(os.getcwd())
#os.chdir(r'c:\saints')
#os.system('mkdir today')

#print(os.getlogin())
#print(os.listdir(r'C:\Saints\Work\Doc\Python'))

# copy file
#shutil.copy('Test.py', 'Test_copy.py')

for root, dirs, files in os.walk(r'C:\Saints'):
    #print(root, "consumes ", end="")
    #size = sum([getsize(join(root, name)) for name in files])
    #print(size)
    #print(" bytes in", len(files), "non-directory files")
    for name in files:
        file_type = splitext(name)[1]
        file_type =str.upper(file_type[1:])
        dir_dtl.append([root, name, file_type, getsize(join(root, name))])
        #print(root, name,file_type, getsize(join(root, name)))
        
    if 'CSV' in dirs:
        dirs.remove('CSV')  # don't visit CVS directories

dir_frame = pd.DataFrame(dir_dtl, columns = ['dir','file','type','size'])
#print(dir_frame.size)
#print(dir_frame.head(5))
gt = dir_frame.groupby(by =['dir','type']).sum()
dir_frame.to_csv(r'c:\saints\test.csv')
#gt.plot.hist()
#plt.show()
#print(gt.head(5))