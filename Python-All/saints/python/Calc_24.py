from saints.python.funcs import Funcs

# math expression, eval('8/(3-8/3)')

f = Funcs.arithmetic_calc

d = [8, 8, 3, 10]
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