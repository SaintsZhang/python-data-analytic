# -*- coding: utf-8 -*-
'''
Created on 2019��10��14��

@author: SZHANG3
'''
# numpy, scipy, matplotlib, pandas

import numpy as np
import pandas as pd
# 1D array
a = np.array((1,2,3), ndmin = 1 )
# 2D array
b = np.array([[1.5,2,3],[4,5,6]])
# 3D array
c = np.array([[[1.5,2,3],[4,5,6]],[[3,2,1],[4,5,6]]])
d = np.arange(10,25,5)
e = np.full((2,2),7)
f = np.eye(2)
g = a - b 
h = a.view()
np.copy(a)
h = a.copy()
# fancy indexing
#print(b[[1,0,1,0],[0,1,2,0]])
#print(b.ravel())
#print(g,g.reshape(3,-1))
#print(np.zeros((3,4)))
#print(np.ones((2,3,4)))
print(c.resize((2,6)))
#print(a,len(a),len(b), len(c))
#print(b.cumsum(axis = 0))
#print(np.sort(a))

#print(np.linspace(0,2,9))