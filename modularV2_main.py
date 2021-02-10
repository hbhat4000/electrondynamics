# object-oriented Python code
# 1) analytical gradient and Hessian
# 2) single trajectory
# 3) Hamiltonian real depends on real and imag depends on imag
# 4) Hamiltonian has zeros in locations where density matrices have zeros; all other DOFs are active
# 5) least squares training

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as si
import scipy.linalg as sl

import multiprocessing
import time
import os
from sklearn import linear_model
from functools import partial
from modularV2_LearnHam import *

if __name__ == '__main__':
    mymol = 'h2'
    mlham = LearnHam(mymol, 'sto-3g', './temp/')
    mlham.load('./data/')
    mlham.loadfield('./data/')
    mlham.trainsplit()
    mlham.buildmodel()
    
    # function outside LearnHam class that computes the gradient
    mygrad = computegrad(mlham)
    # set the gradient inside the object
    mlham.setgrad(mygrad)

    # function outside LearnHam class that computes the Jacobian
    # myjac = computejac(mlham)
    # set the Jacobian inside the object 
    # mlham.setjac(myjac)

    # if you've already computed the Jacobian, you can obtain the Hessian
    # using the following beautiful formula
    # hess2 = 2.0*np.conj(myjac.T) @ myjac
    
    # function outside LearnHam class that computes the Hessian
    # does not need and does not compute either the gradient or the Jacobian
    hess = computehess(mlham)
    # set the Hessian inside the object
    mlham.sethess(hess)

    # print('difference between two ways to compute Hessian:')
    # print(np.linalg.norm(hess - hess2))
    print('******')

    mlham.trainmodel()
    print("Training loss:")
    print(mlham.trainloss)
    print("Gradient of training loss:")
    print(mlham.gradloss)
    
    mlham.plottrainfits()
    mlham.plotvalidfits()
    mlham.computeMLtrainham()
    mlham.plottrainhamerr()

    # propagate using ML Hamiltonian with no field
    # MLsol = mlham.propagate(mlham.MLham, mlham.denMOflat[mlham.offset,:], mytol=1e-9)
    MLsol = mlham.MMUT_Prop(mlham.MLham, mlham.denMOflat[mlham.offset,:], dilfac=4)
    print(MLsol.shape)

    # propagate using Exact Hamiltonian with no field
    EXsol = mlham.propagate(mlham.EXham, mlham.denMOflat[mlham.offset,:], mytol=1e-9)

    # quantitatively and graphically compare the trajectories we just obtained against denMO
    # bigger figure for LiH
    err = mlham.quantcomparetraj(MLsol, EXsol, mlham.denMO)
    print("Field-off propagation error:")
    print(err)
    if mymol == 'lih':
        fs = (8,16)
    else:
        fs = (8,12)
    # mlham.graphcomparetraj(MLsol, EXsol, mlham.denMO, fs)

    # propagate using ML Hamiltonian with field
    # MLsolWF = mlham.propagate(mlham.MLham, mlham.fielddenMOflat[mlham.offset,:], mytol=1e-9, field=True)
    MLsolWF = mlham.MMUT_Prop(mlham.MLham, mlham.fielddenMOflat[mlham.offset,:], field=True, dilfac=4)
    print(MLsolWF.shape)
    
    # propagate using Exact Hamiltonian with field
    EXsolWF = mlham.propagate(mlham.EXham, mlham.fielddenMOflat[mlham.offset,:], mytol=1e-9, field=True)
    
    # quantitatively and graphically compare the trajectories we just obtained against denMO
    # bigger figure for LiH
    errWF = mlham.quantcomparetraj(MLsolWF, EXsolWF, mlham.fielddenMO, 'tdHamerrWF.npz')
    print("Field-on propagation error:")
    print(errWF)
    if mymol == 'lih':
        fs = (8,16)
        infl = False
    else:
        fs = (8,12)
        infl = True
    # mlham.graphcomparetraj(MLsolWF, EXsolWF, mlham.fielddenMO, fs, infl, 'propWF.pdf')

    # compute true theta
    mlham.computetruetheta()
    print("True theta:")
    # print(mlham.truethetaflat)
    print("Loss when we plug in true theta:")
    print(mlham.newloss(mlham.truethetaflat))

