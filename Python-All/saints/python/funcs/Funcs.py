'''
put the user-defined functions here

'''
# Fibonacci series:the sum of two elements defines the next
def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end = ' ')
        a, b = b, a + b 
    print()

def fib2(n):
    a, b = 0 , 1
    result = []
    while a < n:
        result.append(a)
        a, b = b, a + b 
    return result

def arithmetic_calc(n1,n2):
    # math expression, eval('8/(3-8/3)')
    result = {}
    if n1 >= eval(str(n2)):
        result['(' + str(n1) + '-' + str(n2) + ')'] = abs(n1-eval(str(n2)))
    else:
        result['(' + str(n2) + '-' + str(n1) + ')'] = abs(n1-eval(str(n2)))
    result['(' + str(n1) + '+' + str(n2)+ ')'] = n1 + eval(str(n2))
    result['(' + str(n1) + '*' + str(n2)+ ')'] = n1 * eval(str(n2))
    if eval(str(n2)) != 0:
        result['(' + str(n1) + '/' + str(n2)+ ')'] = n1/eval(str(n2))
    if n1 != 0:
        result['(' + str(n2) + '/' + str(n1)+ ')'] = eval(str(n2))/n1
    return result;

def print_props(df_list, prop = '.head()'):
    for df in df_list:
        if (prop == '.head()'):
            title = '\tFirst 5 rows of '
            data = df.head()
        elif (prop == '.tail()'):
            title = '\tLast 5 rows of '
            data = df.tail()
        elif (prop == '.columns'):
            title = '\tColumn Features of '
            data = df.columns
        elif (prop == '.dtypes'):
            title = '\tData Types of '
            data = df.dtypes    
        elif (prop == '.shape'):
            title = '\tShape of '
            data = df.shape
        elif (prop == '.isnull().sum()'):
            title = '\tNull Values in '
            data = df.isnull().sum()
        elif (prop == '.describe()'):
            title = '\tSummary Statistics of '
            data = df.describe()
        
        print(title + df.name )
        print('----------------------------------------')
        print(data)
        print()    

def compare_values(act_col, sat_col):
    act_vals = []
    sat_vals = []
    for a_val in act_col:
        act_vals.append(a_val)
    for s_val in sat_col:
        sat_vals.append(s_val)
    print('Values in ACT only: ')
    for val_a in act_vals:
        if(val_a not in sat_vals):
            print(val_a)
    
    print('----------------------------')
        
    print('Values in SAT only: ')
    for val_s in sat_vals:
        if (val_s not in act_vals):
            print(val_s)            

def convert_to_float(df):
    features = [col for col in df.columns if col != 'State']
    df[features] = df[features].astype(float)
    return df                       
                
def initlog(*args):
    pass    # Remember to implement this!