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

def initlog(*args):
    pass    # Remember to implement this!