import pandas as pd
import math
import scipy
from scipy.stats import norm
from numpy.random import choice


# Calculate the conversion rates
def conversion_rate(data,variant):
    count = 0
    convert = 0
    for i in range(0,len(data)):
        if data.iloc[i]['Variant'] == variant:
            count += 1
            if data.iloc[i]['purchase_TF'] == True:
                convert += 1
    c_r = round((convert/count),4)
    return convert,count,c_r


# A/B testing significance
# H0: conversion_rate_B <= conversion_rate_A
# H1: conversion_rate_B > conversion_rate_A
def significance_test(a,b):
    p = (a[0] + b[0]) / (a[1] + b[1])
    z_socre = (b[2] - a[2]) / math.sqrt(p*(1-p)*(1/a[1] + 1/b[1]))
    z_5 = norm.ppf(0.95,loc=0,scale=1)
    if math.fabs(z_socre) >= z_5:
        print("Reject null hypothesis. Alternative B improved conversion rates over alternative A.")
    else:
        print("Fail to reject null hypothesis. Alternative B didn't improve conversion rates over alternative A.")


def sample_size(a,b):
    z_25 = norm.ppf(0.975,loc=0,scale=1)
    z_2 = norm.ppf(0.8,loc=0,scale=1)
    P = (a[2]+b[2])/2
    size = ((z_25 * math.sqrt(2*P*(1-P)) + z_2 * math.sqrt(a[2] * (1 - a[2]) + b[2] * (1 - b[2])))
            / (0.03))**2
    print('min sample size: %d' % size)
    return size


def randomly_sample(data,size):
    new_df = data.sample(n=size, replace=True, axis=0)
    return new_df


def lambda_calculation(p0,p1):
    lamb1 = math.log(p1/p0, math.e)
    lamb0 = math.log((1-p1)/(1-p0), math.e)
    return lamb0,lamb1

print('Question1: Hypothesis tests')
data = pd.read_csv("AB_test_data.csv",usecols=[0,1])
conversion_rate_A = conversion_rate(data,'A')
conversion_rate_B = conversion_rate(data,'B')
significance_test(conversion_rate_A,conversion_rate_B)
print('-----------------------------')

print('Question2: Hypothesis tests')
size = int(sample_size(conversion_rate_A,conversion_rate_B))
for i in range(0,10):
    print('Test %d: ' % (i+1))
    new_data = randomly_sample(data,size)
    a = conversion_rate(new_data,'A')
    b = conversion_rate(new_data,'B')
    significance_test(a, b)
    print('-----------------------------')


print('Question3: Sequential Test')
a = 1/0.05
b = 0.2
stop1 = math.log(a, math.e)
stop0 = math.log(b, math.e)
num_sample = 0
for i in range(0,10):
    print('Sequential Test %d: ' % (i+1))
    new_data = randomly_sample(data,size)

    sum0 = 0
    sum1 = 0
    countA = 0
    countB = 0
    A1 = 0
    B1 = 0
    for row in range(0,len(new_data)):
        if row == len(new_data)-1:
            print('Not able to stop the test prior to using the full samples.')
            print('--------------------')
            break
        if new_data.iloc[row]['Variant'] == 'A':
            countA += 1
            if data.iloc[row]['purchase_TF'] == True:
                A1 += 1
        elif new_data.iloc[row]['Variant'] == 'B':
            countB += 1
            if data.iloc[row]['purchase_TF'] == True:
                B1 += 1
        if countA != 0 and countB != 0:
            p0 = A1 / countA
            p1 = B1 / countB
            if p0 != 0 and p1 != 0 and p1 != 1 and p0 != 0:
                lambs = lambda_calculation(p0,p1)
                sum0 += lambs[0]
                sum1 += lambs[1]
                if sum0 <= stop0:
                    print('Sample %d: ' % (row + 1))
                    print('xi = 0')
                    print('sum(ln(lambda)) = %.2f' % sum0)
                    print('ln(B) = %.2f' % stop0)
                    print('Accept H0')
                    print('--------------------')
                    break
                elif sum1 >= stop1:
                    print('Sample %d: ' % (row + 1))
                    print('xi = 1')
                    print('sum(ln(lambda)) = %.2f' % sum1)
                    print('ln(A) = %.2f' % stop1)
                    print('Accept H1')
                    print('--------------------')
                    break
                else:
                    continue
    num_sample += (row + 1)
print('average number of iterations: %.1f' % (num_sample/10))