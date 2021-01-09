#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:38:33 2020

@author: Yuyang Zhang
"""
import scipy.stats as si
import numpy as np
from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve


r = 0.03
lambda_1 = 0.04
lambda_2 = 0.02

beta = 0.03
b = 0.05


def dde_solver() :
    M=2000
    y_max = 1000
    h_y = y_max / M
    y = np.array([h_y * i for i in range(M+1)])
    A = np.zeros((M+1,M+1))
    for i in range(M-7) :
        A[i][i+4] = lambda_1-lambda_2+beta+b*y[i+4]/h_y
        A[i][i+5] = -b*y[i+4]/h_y
        A[i][i+8] = -lambda_1
        A[i][i] = lambda_2
    for i in range(M-7, M+1) :
        A[i][i] = lambda_2
        if i <= M-4 :
            A[i][i+4] = lambda_1-lambda_2+beta+b*y[i+4]/h_y
        if i <= M-5:
            A[i][i+5] = -b*y[i+4]/h_y
    A = np.matrix(A)
    
    
    B = np.zeros((M+1,1))
    
    for j in range(M-7, M-5):
        B[j][0] = -lambda_1 * 100
    
    B[M-4][0] = (-lambda_1 -b*y[M]/h_y)*100
    
    for p in range(M-3, M) :
        B[p][0] = (-lambda_2+beta)*100
    
    
    
        
    # B[0][0] = 0
    # B[M-2][0] = -2000 * lambda_1
    
    # B = np.matrix(B)
    
    c = (np.log(beta)+(r-beta)/beta) * np.ones((M+1,1))
    
    c = np.matrix(c)
    
    A_inv = np.linalg.pinv(A)
    
    f_y = A_inv * (c-B)
    
    return y, f_y

def plot_dde() :
    y, f_y= dde_solver()
    # plt.plot(y[4:], f_y[4:])
    # plt.xlabel('Y')
    # plt.ylabel('F(Y)')
    # plt.show()
    
    tck = splrep(y[4:], f_y[4:])
    
    return tck

def f_prime(y) :
    tck = plot_dde()
    delta_t = 1/50
    f_prime = (float(splev(y, tck))-float(splev(y+delta_t, tck)))/delta_t
    
    return f_prime


def plot_boundary() :
    k = 0.25
    h = 0.2
    y = np.linspace(6,100, 50)
    y_delta = np.linspace(6+1/50, 100+1/50,50)
    tck = plot_dde()
    f_p = (splev(y_delta,tck)-splev(y,tck)) * 50
    
    y_down_bd = np.zeros(50)
    y_up_bd = np.zeros(50)

    
    for i in range(50):
        f_pr = f_p[i]
        y_down_bd[i] = (1-k)/(beta * f_pr)
        y_up_bd[i] = (1+h)/(beta*f_pr)
        
    plt.plot(y,y_down_bd)
    plt.plot(y,y_up_bd)
    
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.legend(['Sell boundary','Buy boundary'])
    
def plot_value_function() :
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(40, 100, 1)
    Y = np.arange(20, 80, 1)
    X, Y = np.meshgrid(X, Y)
    tck_y = plot_dde()
    f_y = splev(Y, tck_y)
    X_1 = np.log(X)/beta
    Z = X_1+f_y
    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
 
tck = plot_dde()
k = 0.25
h = 0.2
y = np.linspace(6,100, 50)
y_delta = np.linspace(6+1/100, 100+1/100,50)

f_p = (splev(y_delta,tck)-splev(y,tck)) * 100
    
y_down_bd = np.zeros(50)
y_up_bd = np.zeros(50)

    
for i in range(50):
    f_pr = f_p[i]
    y_down_bd[i] = (1-k)/(beta * f_pr)
    y_up_bd[i] = (1+h)/(beta*f_pr)
tck_1 = splrep(y, y_down_bd)
tck_2 = splrep(y,y_up_bd)

   
def g_sell(y_0, x, y) :
    global tck_1
    k=0.2
    
    f_p = splev(y_0,tck_1)
    
    lhs = (1-k)*float(f_p)+y_0
    
    rhs = (1-k)*x+y
    
    return rhs-lhs


def g_buy(y_0,x, y) :
    global tck_2
    h=0.2
    f_p = splev(y_0, tck_2)
    
    lhs = (1+h)*float(f_p)+y_0

    rhs = (1+h)*x+y
    return lhs-rhs


def plot_sell() :
    global tck, tck_1
    X=np.arange(0,40,2)
    Y=np.arange(80,100,1)

    Y_0 = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            x = X[i]
            y = Y[j]
            Y_0[i][j] = float(fsolve(g_sell, 10, args=(x,y)))
            # print(Y_0[i])
    Y_0 = Y_0.reshape((400,))
    
    X_0 = splev(Y_0,tck_1)

    z_1 = splev(Y_0, tck)
    
    Z = z_1 + np.log(X_0)/beta
    Z = np.array(Z).reshape((20,20))
    Z = Z.T
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(X, Y)

    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

def plot_buy() :
    global tck, tck_2

    X=np.arange(100,120,1)
    Y=np.arange(20,40,1)

    Y_0 = np.zeros((20,20))
    for i in range(20):
        for j in range(20):
            x = X[i]
            y = Y[j]
            Y_0[i][j] = float(fsolve(g_buy, 10, args=(x,y)))
            # print(Y_0[i])
    Y_0 = Y_0.reshape((400,))
    
    X_0 = splev(Y_0,tck_2)
    
    
    
        

    z_1 = splev(Y_0, tck)
    
    Z = z_1 + np.log(X_0)/beta
    Z = np.array(Z).reshape((20,20))
    Z = Z.T
    fig = plt.figure()
    ax = Axes3D(fig)
    X, Y = np.meshgrid(X, Y)

    
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')

    
    
            
            
            
    
    
    
    
        
        
        
    
    
    
    
    
    
    
        
    
    
    
    






