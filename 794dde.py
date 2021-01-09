#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:11:11 2020

@author: shousakai
"""

import scipy.stats as si
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt




lambda_1 = 0.04
lambda_2 = 0.02
gamma = -0.1

beta = 0.06
b = 0.05

def g(y):
    if y<=0 and y>-2.5 :
        return  -gamma * y
    
    
    else :
        delta_t = 1/100
        g_prime = (g(y-1)-g(y-1-delta_t))/delta_t
        
        return (-b*(y-1)*g_prime+lambda_2*g(y-2)+(beta+lambda_1-lambda_2)*g(y-1))/lambda_1
    


def plot_g():
    
    y = np.linspace(0,5,50)
    
    f = np.zeros(50)
    for i in range(50) :
        f[i] = g(y[i])
        
        
    
    plt.plot(y, f)
    plt.xlabel('Y')
    plt.ylabel('g(Y)')
    plt.show()
    

def first_derivative(y) :
    delta_t = 1/100
    
    g_prime = (g(y)-g(y-delta_t))/delta_t
    
    return g_prime
    

def plot_boundary():
    xi = 0.05
    h= 0.08
    k = np.linspace(0.5, 4.5, 5)

    for i in range(5):
        A_k = float(first_derivative(k[i]))
        if A_k > 0 :
            f_down_y =  (1-xi)/(beta*A_k) * np.ones(10)
            f_up_y = (1+h)/(beta*A_k) * np.ones(10)
            y = np.linspace(k[i]-0.5, k[i]+0.5, 10)
            plt.plot(y, f_down_y)
            plt.plot(y, f_up_y)
    
    plt.xlabel('y')
    plt.ylabel('x')
    
        
        


