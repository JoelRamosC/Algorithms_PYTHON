# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 15:59:10 2021
@author: joelr
Estudos da Biblioteca Numpy
"""
import numpy as np

my_array = np.array([[1, 2, 3, 4 , 5], [6, 7, 8, 9, 10 ]])

my_array = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

x = my_array.copy()

print(my_array)
print(my_array.shape[0]) #Mostra n elementos
print(np.arange(10)) #Cria vetor sequencial
print(my_array[1,1])

my_array = np.empty([2,5]) #cria valores aleatórios
for i in range(2):
    for j in range(5):
        my_array[i,j] = i*j
        print(my_array[i,j])
    
my_array = np.zeros([2,5]) #cria matriz de zeros
my_array = np.ones([2,5]) #cria matriz de 1
aleatorio = np.random.random()
my_array = np.random.random([2,5])
print(my_array[1,0:4]) 
print(my_array[:,3])
print(my_array[1][2])
print(type(my_array))


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

soma = a+b
difference = a-b
product = a*b
quotient = a/b

matrix_product = a.dot(b)

for i in range(my_array.shape[0]):
    for j in range(my_array.shape[1]):
        for k in range(my_array.shape[2]):
            print(my_array[i,j,k])
            
my_array = np.array([1, 2, 3, 4], ndmin=5)
print(my_array.ndim)

print(my_array[1, -1])#indice negativo acessa o final do vetor
print(my_array[:,0:4:2])
print(my_array[:,-3:-1])
print(my_array[:,::2])

my_array2 = np.array([1, 2, 3, 4], dtype='S')
my_array2= np.array(['banana', 'maçã','3'])
print(my_array.dtype)

for i in my_array:
    for j in i:
        print(j)


arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

for x in arr:
  print("x represents the 2-D array:")
  print(x)

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.concatenate((arr1, arr2))

print(arr)


arr1 = np.array([[1, 2], [3, 4]])

arr2 = np.array([[5, 6], [7, 8]])

arr = np.concatenate((arr1, arr2), axis=1)

print(arr)

arr1 = np.array([1, 2, 3])

arr2 = np.array([4, 5, 6])

arr = np.stack((arr1, arr2), axis=1)

print(arr)

word = "apartamento"
for i in range(len(word)-1,-1,-1):
    print(word[i])

np.array_split(my_array,2)

x = np.where(my_array == 4) #tuple com linha e coluna 

x = np.where(my_array[0,:]%2 == 0) #posição do número é impar
x = np.where(my_array[0,:]%2 == 1) #posição do número par


arr = np.array(['banana', 'cherry', 'apple'])
arr = np.array([True, False, True])
arr = np.array([[3, 2, 4], [5, 0, 1]])

print(np.sort(arr))

#Filtros

arr = np.array([41, 42, 43, 44])#mask

x = [True, False, True, False]

newarr = arr[x]

print(newarr)


my_array = np.array([41, 42, 43, 44])

# Create an empty list
filter_arr = []

# go through each element in arr
for element in arr:
  # if the element is higher than 42, set the value to True, otherwise False:
  if element > 42:
    filter_arr.append(True)
  else:
    filter_arr.append(False)

new_array = my_array[filter_arr]

print(filter_arr)
print(new_array)

arr = np.array([41, 42, 43, 44])

filter_arr = arr > 42

newarr = arr[filter_arr]

print(filter_arr)
print(newarr)

#Comandos Ramdom
from numpy import random
x = random.randint(100)#Random inteiro
x = random.rand()
x = random.rand(5)
x = random.rand(3,5)
x=random.randint(100, size=(5,5))
x = random.choice([3, 5, 7, 9]) #retorna um valor do vetor
x = random.choice([3,4,5, 6,7,8], size=(3,5)) #Constrói a matrix puxando números da base referida

#configura probabolidade de escolha dos números
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(100))
x = random.choice([3, 5, 7, 9], p=[0.1, 0.3, 0.6, 0.0], size=(3, 5))

my_array = np.array([1, 2, 3, 4, 5])

random.shuffle(my_array) #embaralha os números e altera a matriz
random.permutation(my_array) #embaralha os números criando uma nova matriz


import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot([0, 1, 2, 3, 4, 5], hist = True)
plt.show()

#Distribuição Normal Analógica
# loc - (Mean) where the peak of the bell exists.

# scale - (Standard Deviation) how flat the graph distribution should be.

# size - The shape of the returned array.

x = random.normal(size=(100, 1000))
x = random.normal(loc=0, scale=5, size=(100, 100))
sns.distplot(x, hist = True)
plt.show()


#distribuição binomial (Normal Digital)
# n - number of trials.
# p - probability of occurence of each trial (e.g. for toss of a coin 0.5 each).

# size - The shape of the returned array.
x = random.binomial(n=10, p=0.5, size=10)
x = random.binomial(n=10, p=0.9, size=1000)
sns.distplot(x, hist=True, kde=False)
plt.show()


#comparação entre métodos
sns.distplot(random.normal(loc=50, scale=5, size=1000), hist=False, label='normal')
sns.distplot(random.binomial(n=100, p=0.5, size=1000), hist=False, label='binomial')
plt.show()

# Poisson Distribution is a Discrete Distribution.

# It estimates how many times an event can happen in a specified time. e.g. If someone eats twice a day what is probability he will eat thrice?

# It has two parameters:

# lam - rate or known number of occurences e.g. 2 for above problem.

# size - The shape of the returned array.

x = random.poisson(lam=2, size=10)
sns.distplot(random.poisson(lam=7, size=10000), kde=False)
plt.show()

#comparação entre normal e poisson
sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')
plt.show()

# Uniform Distribution
# Used to describe probability where every event has equal chances of occuring.

# E.g. Generation of random numbers.

# It has three parameters:

# a - lower bound - default 0 .0.

# b - upper bound - default 1.0.

# size - The shape of the returned array.

sns.distplot(random.uniform(size=100), hist=False)
plt.show()

# Logistic Distribution
# Logistic Distribution is used to describe growth.

# Used extensively in machine learning in logistic regression, neural networks etc.

# It has three parameters:

# loc - mean, where the peak is. Default 0.

# scale - standard deviation, the flatness of distribution. Default 1.

# size - The shape of the returned array.

sns.distplot(random.logistic(size=1000), hist=False)
plt.show()

#gaussiana x logistic distribuition
sns.distplot(random.normal(scale=2, size=1000), hist=False, label='normal')
sns.distplot(random.logistic(size=1000), hist=False, label='logistic')
plt.show()

# Pareto Distribution
# A distribution following Pareto's law i.e. 80-20 distribution (20% factors cause 80% outcome).

# It has two parameter:

# a - shape parameter.

# size - The shape of the returned array.

sns.distplot(random.pareto(a=2, size=1000), kde=False)
plt.show()


### Functions
#Somar elementos entre vetores
x = [1, 2, 3, 4] #lista
y = [4, 5, 6, 7]

x = np.array([1, 2, 3])
y = np.array([5, 6, 7])
z = np.add(x, y)
z = np.sum([x,y])
z = np.subtract(x, y)
z = np.multiply(x, y)
z = np.divide(x, y)
z = np.power(x,y)
z = np.mod(x,y)
z = np.remainder(x,y)
z = np.divmod(x,y)
z = np.absolute(x,y)
print(z)

x = np.array([-1, 2.555, 3.9])
z = np.trunc(x)
z = np.fix(x)
z = np.around(x)
z = np.floor(x)
z = np.ceil(x)

x = np.arange(1,10)#não inclui o 10
z = np.log2(x)
z = np.log10(x)
z = np.log()


x = np.array([1, 2, 3])
z = np.prod(x)
z = np.prod([x,x])

# Finding LCM (Lowest Common Multiple)
num1 = 4
num2 = 6
x = np.lcm(num1, num2)

#Finding GCD (Greatest Common Denominator)
num1 = 6
num2 = 15
x = np.gcd(num1,num2)
x = np.gcd.reduce([3,5,8,9,24])


x = np.sin(np.pi/2)
arr = np.array([np.pi/2, np.pi/3, np.pi/4, np.pi/5])
x = np.sin(arr)

arr = np.array([90, 180, 270, 360])
x = np.deg2rad(arr) #converte para radianos

x = np.sinh(np.pi/2) #função hiperbólica
x = np.arcsinh(1.0)







