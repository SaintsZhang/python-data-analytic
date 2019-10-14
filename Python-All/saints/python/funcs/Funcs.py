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
    result = []
    result.append(abs(n1-n2))
    result.append(n1+n2)
    result.append(n1*n2)
    result.append(n1/n2)
    result.append(n2/n1)
    return result;

def initlog(*args):
    pass    # Remember to implement this!