# -*- coding: utf-8 -*-

from saints.python.funcs import Funcs

# math expression, eval('8/(3-8/3)')
# 2,2,2,9
# 2,7,8,9
# 1,2,7,7
# 4,4,10,10
# 6,9,9,10
# 1,5,5,5
# 2,5,5,10
# 1,4,5,6
# 3,3,7,7
# 3,3,8,8


f = Funcs.arithmetic_calc

d = [3, 6, 7, 10]
ret = {}
res_24 = {}
res_final ={}

for i in range(len(d)):
    d1 = d.copy()
    d1.remove(d[i])
    for j in range(len(d1)):
        d2 = d1.copy()
        d2.remove(d1[j])
        temp = f(d2[0],d2[1])
        ret.update(temp)
        for k, v in ret.items():
            temp = f(d1[j],k)
            res_24.update(temp)
        # initialize the middle dictionary for each new loop    
        ret = {}    
    for k,v in res_24.items():
        temp = f(d[i],k)
        res_final.update(temp)
    # initialize the middle dictionary for each new loop    
    res_24 = {}
    
# iterate dictionary to get the expression = 24
for k, v in res_final.items():
    if abs(v-24) <= 0.00001:
        print(k)