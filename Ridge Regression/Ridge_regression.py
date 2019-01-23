# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 20:14:12 2018

@author: User
"""
import numpy as np


def ridgeReg(X,Y,_lambda):
    n=X.shape[1]
    d=X.shape[0]
    Xbar=np.vstack((X,np.ones(n)))
    Ibar=np.identity(d+1)
    Ibar[d][d]=0
    Xbar=np.asmatrix(Xbar)
    Ibar=np.asmatrix(Ibar)
    Y=np.asmatrix(Y)
    C = Xbar * Xbar.T + _lambda * Ibar
    D=Xbar * Y
    C_inv = np.linalg.inv(C)
    wbar= C_inv*D    
    cvErrs=(wbar.T*Xbar - Y.T)/ (1-np.diagonal(Xbar.T*C_inv*Xbar))
    obj=_lambda*wbar[0:d].T*wbar[0:d]+(wbar.T*Xbar-Y.T)*(wbar.T*Xbar-Y.T).T
    b=wbar[d]
    wbar=wbar[:d]
    
    
    return wbar,b, obj, cvErrs


