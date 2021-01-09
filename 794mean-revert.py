#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Apr 19 14:27:12 2020

@author: Chenshuo
"""

import math
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev


def optimal(r):
    b = 0.07
    sigma = 0.2
    zeta1 = 0.006
    zeta2 = 0.004
    c1 = 0.25
    c2 = 0.20
    fxi = lambda x:  b -r - sigma**2 * x + ((c1 * zeta1)/ (1 + x * zeta1)) - (c1 * zeta1) + ((c2 * zeta2) / (1 + x * zeta2))  \
    - c2 * zeta2 
    xi = fsolve(fxi,0.29)
    return xi[0]

def RSmatrix():
    delta =0.0101
    k=1
    alpha=0.05
    b = 0.07
    sigma = 0.2
    zeta1 = 0.006
    zeta2 = 0.004
    c1 = 0.25
    c2 = 0.20
    beta = 0.06
    M = 200
    r_max = 1
    g1 = 1
    h_r = r_max/M
    r = np.array([h_r * i for i in range(M+1)])
    A = np.zeros((M,1))
    #print(A)
    for i in range(M) :
        # flow = float
        opt = float(optimal(r[M-i-1]))
        # if opt <0 :
        #     opt = 0
        # elif opt>1:
        #     opt = 1
        # A[i][0] = (1 - opt) / beta 
        # print(opt)
        y = np.log(beta) -1  +(b * opt) / beta 
        y = y -(((sigma**2) * opt) / (2 * beta)) + c1 * (np.log(1 + opt * zeta1) / beta)
        y = y  - c1*(opt * zeta1) / beta + c2 * (np.log(1 + opt * zeta2) / beta) 
        y = y - c2 * opt * zeta2 / beta
        A[i][0] += y
        
    
    A[0][0] = A[0][0]-k*(alpha-r[M-1])*g1/(2*h_r)-delta**2*g1/(2*h_r**2)
    # A[M-1][0] = A[M-1][0]+k*(alpha)*g1/(2*h_r)-delta**2*g1/(2*h_r**2)
    
    
    
    return A
    
    

def solvePDE():
    k = 1
    alpha = 0.05
    # g1 = 0.05
    delta = 0.0101
    beta = 0.06
    M=200
    h_r = 1/M
    
    r = np.array([h_r * i for i in range(M+1)])
    
    D = np.zeros((M,M))
    
    for j in range(M) :
        
        if j==0 :
            D[j][j] =  -beta-delta**2/h_r**2
            D[j][j+1] = -k*(alpha-r[M-j-1])/(2*h_r)+delta**2/(2*h_r**2)
        
        if j==M-1:
            D[j][j] = -beta-delta**2/h_r**2
            D[j][j-1] = k*(alpha-r[M-j-1])/(2*h_r)+delta**2/(2*h_r**2)
        
        else:
            D[j][j]= -beta-delta**2/h_r**2
            D[j][j-1] = -k*(alpha-r[M-j-1])/(2*h_r)+delta**2/(2*h_r**2)
            D[j][j+1] = k*(alpha-r[M-j-1])/(2*h_r)-delta**2/(2*h_r**2)
    
    A = RSmatrix()
    D = np.matrix(D)
    sol = np.linalg.inv(D) * np.matrix(A)
    
    return D, sol
    
    

def plot_value_function() :
    M = 200
    h_r = 2/M
    
    r = list(h_r * i for i in range(M))
        
        
    D,sol = solvePDE()
    sol = sol.reshape((1,M))
    s = np.zeros(M)
    
    for i in range(M) :
        s[i] = sol[0,M-1-i]

    # return r,s
    tck = splrep(r[10:M-20],s[10:M-20])
    plt.plot(r[10:M-20], s[10:M-20])
    plt.xlabel('r')
    plt.ylabel('f(r)')
    plt.show()
    
    
    
    
          
        
    
    
    
    
    
        
