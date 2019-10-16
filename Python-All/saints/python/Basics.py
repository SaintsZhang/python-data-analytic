# -*- coding: utf-8 -*-
#from math import exp, floor, ceil

# this is the first comment

'''
a block of comment
+ - * / 
//  # floor division discards the fractional part 
%   # returns the remainder of the division
**  # the powers
abs, exp
floor, ceil
3 * 'abc' # 'abc' repeated 3 times
len(s)
str
 
'''
#String - immutable
#indexing
word = 'Python'
#word[0] -> P, word[-1] ->n
# Slicing
#word[0:2], word[:2] ->'Py'; word[-2:] -> 'on'

#Lists - mutable

squares = [1, 4, 9, 16, 25]
squares.append(6**2)

# Fibonacci series:the sum of two elements defines the next
a, b = 0 , 1
while a < 10:
    print(a, end=' ')
    a, b = b, a + b
print('\n')    
# Nine-nine multiplication table: 9x9
# range(1,10,3)
i = 1
while i < 10:
    for j in range(i):
        print(str(i)+'*' +str(j+1) + '=' + str(i*(j+1)),end =' ')    
    i = i + 1
    print('\n')

# break, continue, pass
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n//x)
            break
    else:
        print(n, 'is a prime number')
        
        
        
