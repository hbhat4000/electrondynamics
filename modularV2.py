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

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

# environment variables that, depending on the OS, control the number of threads used for numerical linear algebra
threadkeys = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']

class LearnHam:

    # class initializer
    def __init__(self, mol, basis, outpath):
        
        # store the short string that indicates which molecule we're working on
        self.mol = mol

        # store the short string that indicates which basis set we're using
        self.basis = basis
        
        # field sign correction
        if self.mol == 'h2':
            self.fieldsign = -1
        else:
            self.fieldsign = 1
            
        # store the path to output files, i.e., saved research outputs like figures
        self.outpath = outpath

        # need this for the gradient
        self.stars_g = np.array([[1,-1,1,-1]],dtype=np.int32).T
        
        # need this for the Jacobian
        self.stars_j = np.array([1,-1,1,-1],dtype=np.int32)

        # need this for the Hessian
        self.stars_s = np.array([[1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1]],dtype=np.int32).T
        self.stars_a = np.array([[-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1]],dtype=np.int32).T
        self.stars_sa = self.stars_s * self.stars_a
    
    # load and process field-free data
    def load(self,inpath):
        # store the path to input files, i.e., training data, auxiliary matrices, etc
        inpath = inpath
        rawden = np.load(inpath + 'td_dens_re+im_rt-tdexx_delta_s0_'+self.mol+'_'+self.basis+'.npz',allow_pickle=True)
        overlap = np.load(inpath + 'ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+self.mol+'_'+self.basis+'.npz',allow_pickle=True)

        # put things into better variables
        self.kinmat = overlap['ke_data']
        self.enmat = overlap['en_data']
        self.eeten = overlap['ee_twoe_data']

        # need these for orthogonalization below
        s = overlap['overlap_data']
        self.sevals, self.sevecs = np.linalg.eigh(s)
        self.xmat = self.sevecs @ np.diag(self.sevals**(-0.5))
            
        # remove duplicates
        realpt = rawden['td_dens_re_data']
        imagpt = rawden['td_dens_im_data']
        den = realpt + 1j*imagpt
        self.drc = den.shape[1]
        # Read dipole data
        self.didat = [[]]*3
        self.didat[0] = np.zeros(shape=(self.drc,self.drc))
        self.didat[1] = np.zeros(shape=(self.drc,self.drc))
        self.didat[2] = np.zeros(shape=(self.drc,self.drc))
        self.didat[0] = overlap['dipx_data']
        self.didat[1] = overlap['dipy_data']
        self.didat[2] = overlap['dipz_data']
        denflat = den.reshape((-1,self.drc**2))
        dennodupflat = np.array([np.delete(denflat[:,i], np.s_[101::100]) for i in range(self.drc**2)]).T
        self.denAO = dennodupflat.reshape((-1,self.drc,self.drc))

        # transform to MO using canonical orthogonalization
        # in this code, by "MO" we mean the canonical orthogonalization of the AO basis
        self.denMO = np.zeros(self.denAO.shape,dtype=np.complex128)
        self.denMOflat = self.denMO.reshape((-1,self.drc**2))
        npts = self.denAO.shape[0]
        for i in range(1,npts):
            self.denMO[i,:,:] = np.diag(self.sevals**(0.5)) @ self.sevecs.T @ self.denAO[i,:,:] @ self.sevecs @ np.diag(self.sevals**(0.5))

        # find off-diag DOFs of the supplied density matrices that are (sufficiently close to) zero across all time points
        self.realnzs = []
        self.imagnzs = []
        for j in range(realpt.shape[1]):
            for i in range(j+1):
                realnorm = np.linalg.norm(np.real(self.denMO[:,i,j]))
                # print("|| Re[den["+str(i)+","+str(j)+"]] || = " + str(realnorm))
                if not np.isclose(realnorm,0):
                    self.realnzs.append((i,j))

                if i < j:
                    imagnorm = np.linalg.norm(np.imag(self.denMO[:,i,j]))
                    # print("|| Im[den["+str(i)+","+str(j)+"]] || = " + str(imagnorm))
                    if not np.isclose(imagnorm,0):
                        self.imagnzs.append((i,j))

        # these turn out to be super useful for train/validation split below
        # more generally, you iterate through these to extract real DOFs from the complex density matrices
        self.rnzl = [list(t) for t in zip(*self.realnzs)]
        self.inzl = [list(t) for t in zip(*self.imagnzs)]

        # build two dictionaries that help us find the absolute column number given human-readable (i,j) indices
        # for both the real and imaginary non-zero density DOFs
        # also build matrix equivalents for these dictionaries, which are needed by numba jit
        self.nzreals = {}
        self.nzrealm = -np.ones((self.drc,self.drc),dtype=np.int32)
        cnt = 0
        for i in self.realnzs:
            self.nzreals[i] = cnt
            self.nzrealm[i[0],i[1]] = cnt
            cnt += 1

        self.nzimags = {}
        self.nzimagm = -np.ones((self.drc,self.drc),dtype=np.int32)
        for i in self.imagnzs:
            self.nzimags[i] = cnt
            self.nzimagm[i[0],i[1]] = cnt
            cnt += 1

        # need all of the following for our fast Hessian assembler
        self.nnzr = len(self.nzreals)
        self.nnzi = len(self.nzimags)
        self.hesslen = self.nnzr*(self.nnzr+1) + self.nnzi**2
        self.ndof = cnt
        self.allnzs = list(set(self.realnzs + self.imagnzs))
        self.nall = len(self.allnzs)
        self.nzrow = np.zeros(self.nall, dtype=np.int32)
        self.nzcol = np.zeros(self.nall, dtype=np.int32)
        for i in range(self.nall):
            self.nzrow[i] = self.allnzs[i][0]
            self.nzcol[i] = self.allnzs[i][1]
        
        # show that we got here
        return True
    
    # load and process data with field
    def loadfield(self,inpath):
        fielddata = np.load(inpath + 'td_efield+dipole_rt-tdexx_ndlaser1cycs0_'+self.mol+'_'+self.basis+'.npz')
        self.efdat = fielddata['td_efield_data']
        fielddens = np.load(inpath + 'td_dens_re+im_rt-tdexx_ndlaser1cycs0_'+self.mol+'_'+self.basis+'.npz',allow_pickle=True)
        self.fieldden = fielddens['td_dens_re_data'] + 1j*fielddens['td_dens_im_data']

        # change basis from AO to orthogonalization of AO (called MO here)
        fielddenMO = np.zeros(self.fieldden.shape, dtype=np.complex128)
        for i in range(self.fieldden.shape[0]):
            fielddenMO[i,:,:] = np.diag(self.sevals**(0.5)) @ self.sevecs.T @ self.fieldden[i,:,:] @ self.sevecs @ np.diag(self.sevals**(0.5))

        # remove duplicates
        fielddenMOflat = fielddenMO.reshape((-1,self.drc**2))

        # retain in the object only the stuff without duplicates
        self.fielddenMOflat = np.array([np.delete(fielddenMOflat[:,i], np.s_[101::100]) for i in range(self.drc**2)]).T
        self.fielddenMO = self.fielddenMOflat.reshape((-1,self.drc,self.drc))

        # show that we got here
        return True
    
    # convert complex drc x drc density matrices to real ndof density vectors
    def complex2real(self, rawden):
        rawden_real = np.real(rawden)[:,self.rnzl[0], self.rnzl[1]]
        rawden_imag = np.imag(rawden)[:,self.inzl[0], self.inzl[1]]
        realden = np.hstack([rawden_real, rawden_imag])
        return realden
        
    # this function sets up the training and validation split
    # ntrain should be the desired length of the training set
    # tt should be the desired length of the training + validation set
    def trainsplit(self, ntrain=1000, tt=4000):
        self.x_inp = self.complex2real(self.denMO)

        self.offset = 2
        self.tt = tt
        self.ntrain = ntrain
        self.x_inp = self.x_inp[self.offset:(self.tt+self.offset),:]

        self.dt = 0.08268 
        self.tint_whole = np.arange(self.x_inp.shape[0])*self.dt

        # training set
        self.x_inp_train = self.x_inp[:ntrain,:]
        self.denMO_train = self.denMO[self.offset:(self.offset+self.ntrain),:,:]
        self.tint = self.tint_whole[:ntrain]

        # validation set
        self.x_inp_valid = self.x_inp[ntrain:,:]
        self.denMO_valid = self.denMO[(self.offset+self.ntrain):(self.offset+self.tt),:,:]
        self.tint_valid = self.tint_whole[ntrain:]

        # show that we got here
        return True

    # note that this replaces the vmap call in JAX
    # you pass in flattened real density vectors and the corresponding times
    # you get back a batch evaluation of rgm on everything you passed in
    def batchrgm(self, realdens, times):
        return np.stack([ self.rgm(x,t) for x,t in zip(realdens, times) ],axis=0)[:,:,0]
        
    # here we set up building blocks for the ML Hamiltonian model
    # this includes computing xdot and rgmmat on the training set
    def buildmodel(self, maxdeg=1):
        # d = ndof = number of degrees of freedom
        # let's agree not to use "d" anymore as it's not descriptive enough

        # set maximum polynomial degree (1 = linear, 2 = quadratic, etc)
        self.maxdeg = maxdeg

        # compute total number of parameters per dimension
        self.nump = 1
        for j in range(1,self.maxdeg+1):
            self.nump += self.ndof**j
        
        # this can be changed if we don't want the ML Ham to have as many DOF's as the input density matrices
        self.hamreals = self.nzreals.copy()
        self.hamimags = self.nzimags.copy()

        self.nhamreals = len(self.hamreals)
        self.nhamimags = len(self.hamimags)
        self.hamdof = self.nhamreals + self.nhamimags

        self.lentheta = (1+len(self.nzreals))*self.nhamreals + len(self.nzimags)*self.nhamimags

        # set up matrices for initial training
        self.xdot = (self.x_inp_train[2:,:] - self.x_inp_train[:-2,:])/(2*self.dt)
        self.denMO_train_dot = (self.denMO_train[2:,:] - self.denMO_train[:-2,:])/(2*self.dt)
        
        # note that this replaces the vmap call in JAX
        self.rgmmat = self.batchrgm(self.x_inp_train[1:-1,:], self.tint[1:-1])
        
        # need this to compute gradient way down below
        self.qmat = self.computeqmat(np.zeros(self.lentheta))

    def blocktheta(self, thetain):
        thetaUL = thetain[:(1+len(self.nzreals))*self.nhamreals].reshape((1+len(self.nzreals),self.nhamreals))
        thetaLR = thetain[(1+len(self.nzreals))*self.nhamreals:].reshape((len(self.nzimags),self.nhamimags))
        thetaU = np.hstack([ thetaUL, np.zeros((1+len(self.nzreals), self.nhamimags))])
        thetaL = np.hstack([ np.zeros((len(self.nzimags), self.nhamreals)), thetaLR])
        theta = np.vstack([thetaU, thetaL])
        return theta

    # compute regression matrix (rgm) on data set x at time t
    def rgm(self, x, t):
        # form regression matrix
        # start with constants and linears
        regmat0 = np.array([[1.]])
        regmat1 = np.expand_dims(x,1)
        reglist = [regmat0, regmat1]

        # include higher-order terms if needed
        for j in range(2, self.maxdeg+1):
            reglist.append(np.matmul(reglist[j-1], regmat1.T).reshape((self.ndof**j,1))/np.math.factorial(j))

        # concatenate everybody
        regmat = np.concatenate(reglist)
        return regmat
    
    # this function takes as input a theta vector that parameterizes the Hamiltonian, and
    # outputs predicted values of xdot -- time derivatives of the time-dependent density matrix,
    # at each instant of time represented in the training data
    #
    # note that this function has to be run after buildmodel -- can include some error-checking on that...
    #
    def newpred(self, thetain, extrgm=None, extden=None):
        if extrgm is None:
            rgmmat = self.rgmmat
        else:
            rgmmat = extrgm
        if extden is None:
            den = self.denMO_train
        else:
            den = extden
        
        theta = self.blocktheta(thetain)
        hflat = np.matmul(rgmmat, theta)
        h = np.zeros((hflat.shape[0], self.drc, self.drc), dtype=np.complex128)

        for ij in self.hamreals:
            h[:, ij[0], ij[1]] = hflat[:, self.hamreals[ij]]
            h[:, ij[1], ij[0]] = hflat[:, self.hamreals[ij]]
        for ij in self.hamimags:
            h[:, ij[0], ij[1]] += (1J)*hflat[:, self.hamimags[ij]]
            h[:, ij[1], ij[0]] -= (1J)*hflat[:, self.hamimags[ij]]
        
        hp = np.einsum('ijk,ikl->ijl', h, den[1:-1,:])
        ph = np.einsum('ijk,ikl->ijl', den[1:-1,:], h)
        rhs = (hp - ph)/(1j)
        return rhs
    
    # note that the residual is i*\dot{P} - [H,P]
    # this function computes -(i*\dot{P} - [H,P])^\ast = -(-i*\dot{P}^\ast - [H,P]^\ast)
    #                                                  = i*\dot{P}^\ast + [H,P]^\ast
    # hence -np.conj(self.computeqmat)) gives us the residual
    #
    # this function is used to compute the gradient of the loss
    def computeqmat(self, thetain):
        theta = self.blocktheta(thetain)
        hflat = np.matmul(self.rgmmat, theta)
        hl = []
        for i in range(self.drc):
            for j in range(self.drc):
                if i>j:
                    if (j,i) in self.nzreals and (j,i) in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(j,i)]] - 1J*hflat[:,self.nzimags[(j,i)]])
                    elif (j,i) in self.nzreals and (j,i) not in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(j,i)]])
                    elif (j,i) not in self.nzreals and (j,i) in self.nzimags:
                        hl.append(-1J*hflat[:,self.nzimags[(j,i)]])                    
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                if i==j:
                    if (i,j) in self.nzreals:
                        hl.append(hflat[:,self.nzreals[(i,j)]])
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                if i<j:
                    if (i,j) in self.nzreals and (i,j) in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(i,j)]] + 1J*hflat[:,self.nzimags[(i,j)]])
                    elif (i,j) in self.nzreals and (i,j) not in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(i,j)]])
                    elif (i,j) not in self.nzreals and (i,j) in self.nzimags:
                        hl.append(1J*hflat[:,self.nzimags[(i,j)]])
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                    
        h = np.vstack(hl).T.reshape((-1,self.drc,self.drc))
        hp = np.einsum('ijk,ikl->ijl', h, self.denMO_train[1:-1,:])
        ph = np.einsum('ijk,ikl->ijl', self.denMO_train[1:-1,:], h)
        rhs = (hp - ph)
        qmat = -np.conj((1j)*self.denMO_train_dot - rhs)
        return qmat
    
    # this function only exists so that I could use JAX to debug the Jacobian
    # I needed a function that takes in a real vector theta and outputs the residual as a real vector,
    # so that I could take the Jacobian using JAX.
    def computeresid(self, thetain):
        theta = self.blocktheta(thetain)
        hflat = np.matmul(self.rgmmat, theta)
        hl = []
        for i in range(self.ndof):
            for j in range(self.ndof):
                if i>j:
                    if (j,i) in self.nzreals and (j,i) in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(j,i)]] - 1J*hflat[:,self.nzimags[(j,i)]])
                    elif (j,i) in self.nzreals and (j,i) not in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(j,i)]])
                    elif (j,i) not in self.nzreals and (j,i) in self.nzimags:
                        hl.append(-1J*hflat[:,self.nzimags[(j,i)]])                    
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                if i==j:
                    if (i,j) in self.nzreals:
                        hl.append(hflat[:,self.nzreals[(i,j)]])
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                if i<j:
                    if (i,j) in self.nzreals and (i,j) in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(i,j)]] + 1J*hflat[:,self.nzimags[(i,j)]])
                    elif (i,j) in self.nzreals and (i,j) not in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(i,j)]])
                    elif (i,j) not in self.nzreals and (i,j) in self.nzimags:
                        hl.append(1J*hflat[:,self.nzimags[(i,j)]])
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                    
        h = np.vstack(hl).T.reshape((-1,self.ndof))
        hp = np.einsum('ijk,ikl->ijl', h, self.denMO_train[1:-1,:])
        ph = np.einsum('ijk,ikl->ijl', self.denMO_train[1:-1,:], h)
        rhs = (hp - ph)
        complexresid = ((1j)*self.denMO_train_dot - rhs).reshape(-1)
        outresid = np.concatenate([np.real(complexresid), np.imag(complexresid)])
        return outresid

    # this is the sum of squares loss function
    def newloss(self, thetain):
        theta = self.blocktheta(thetain)
        hflat = np.matmul(self.rgmmat, theta)
        hl = []
        for i in range(self.drc):
            for j in range(self.drc):
                if i>j:
                    if (j,i) in self.nzreals and (j,i) in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(j,i)]] - 1J*hflat[:,self.nzimags[(j,i)]])
                    elif (j,i) in self.nzreals and (j,i) not in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(j,i)]])
                    elif (j,i) not in self.nzreals and (j,i) in self.nzimags:
                        hl.append(-1J*hflat[:,self.nzimags[(j,i)]])                    
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                if i==j:
                    if (i,j) in self.nzreals:
                        hl.append(hflat[:,self.nzreals[(i,j)]])
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                if i<j:
                    if (i,j) in self.nzreals and (i,j) in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(i,j)]] + 1J*hflat[:,self.nzimags[(i,j)]])
                    elif (i,j) in self.nzreals and (i,j) not in self.nzimags:
                        hl.append(hflat[:,self.nzreals[(i,j)]])
                    elif (i,j) not in self.nzreals and (i,j) in self.nzimags:
                        hl.append(1J*hflat[:,self.nzimags[(i,j)]])
                    else:
                        hl.append(np.zeros(hflat.shape[0]))
                    
        h = np.vstack(hl).T.reshape((-1,self.drc,self.drc))
        hp = np.einsum('ijk,ikl->ijl', h, self.denMO_train[1:-1,:])
        ph = np.einsum('ijk,ikl->ijl', self.denMO_train[1:-1,:], h)
        rhs = (hp - ph)
        resid = (1j)*self.denMO_train_dot - rhs
        loss = np.sum(np.abs(resid)**2)
        return loss

    # set gradient from external source
    def setgrad(self, ingrad):
        self.grad = ingrad.copy()
        return True
    
    # set Jacobian from external source
    def setjac(self, injac):
        self.jac = injac.copy()
        return True

    # set Hessian from external source
    def sethess(self, inhess):
        self.hess = inhess.copy()
        return True
    
    # TRAIN AND SAVE theta TO DISK
    def trainmodel(self, savetodisk=True):

        # define zero vector in theta space
        theta0 = np.zeros(self.lentheta)        
        
        # solve least squares minimization problem
        self.theta,_,_,_ = sl.lstsq(self.hess, -self.grad, lapack_driver='gelsy')
        #
        # print("Computing regularization path using the LARS ...")
        # resid = (1J)*self.denMO_train_dot - (1J)*self.newpred(theta0)
        # alphas, _, coefs = linear_model.lars_path(self.jac, -resid.reshape(-1,1), method='lasso',verbose= True,alpha_min=-1)
        # self.theta = coefs.T[-1]
        
        # save training loss
        self.trainloss = self.newloss(self.theta)
        
        # update qmat and compute norm of gradient of loss at new theta value
        self.qmat = self.computeqmat(self.theta)
        self.gradloss = np.linalg.norm(computegrad(self))
        
        # check newpred function
        # resid = (1J)*self.denMO_train_dot - (1J)*self.newpred(self.theta)
        # print("checking newpred: " + str(np.linalg.norm(resid)))
        # print("checking newresid: " + str(np.linalg.norm(self.qmat)))

        # save to disk
        if savetodisk:
            fname = self.outpath + 'hamiltoniantheta0.npz'
            np.savez(fname, theta=self.theta)
        
        # show that we made it here
        return True
    
    # plot fit on training data and save to disk
    def plottrainfits(self):
        xdotpred = self.complex2real(self.newpred(self.theta))
        for i in range(self.ndof):
            plt.figure(figsize=(6,6))
            plt.plot(self.tint[1:-1], self.xdot[:,i])
            plt.plot(self.tint[1:-1], xdotpred[:,i])
            plt.xlabel('time')
            plt.ylabel('x[' + str(i) + ']')
            plt.savefig(self.outpath + 'prefitTRAIN1' + str(i) + '.pdf')
            plt.close()
        
        # show that we made it here
        return True
        
    # plot fit on validation data and save to disk
    def plotvalidfits(self):
        xdotvalid = (self.x_inp_valid[2:,:] - self.x_inp_valid[:-2,:])/(2*self.dt)
        rgmmat = self.batchrgm(self.x_inp_valid[1:-1,:], self.tint_valid[1:-1])
        xdotvalidpred = self.complex2real(self.newpred(self.theta, extrgm=rgmmat, extden=self.denMO_valid))
        for i in range(self.ndof):
            plt.figure(figsize=(18,6))
            plt.plot(self.tint_valid[1:-1], xdotvalid[:,i])
            plt.plot(self.tint_valid[1:-1], xdotvalidpred[:,i])
            plt.xlabel('time')
            plt.ylabel('x[' + str(i) + ']')
            plt.savefig(self.outpath + 'prefitVALID1' + str(i) + '.pdf')
            plt.close()
        
        # show that we made it here
        return True
    
    # compute the ML Hamiltonian on the training set
    def computeMLtrainham(self):
        myhamraw = np.array(np.matmul(self.batchrgm(self.x_inp_train, self.tint), self.blocktheta(self.theta)))
        self.myham = np.zeros((myhamraw.shape[0],self.drc,self.drc), dtype=np.complex128)
        for ij in self.hamreals:
            self.myham[:, ij[0], ij[1]] = myhamraw[:, self.hamreals[ij]]
            self.myham[:, ij[1], ij[0]] = myhamraw[:, self.hamreals[ij]]
        for ij in self.hamimags:
            self.myham[:, ij[0], ij[1]] += (1J)*myhamraw[:, self.hamimags[ij]]
            self.myham[:, ij[1], ij[0]] -= (1J)*myhamraw[:, self.hamimags[ij]]
        
        # show that we made it here
        return True
        
    # Karnamohit's function (July 1 version)
    # this computes the Coulomb and exchange parts of the potential
    def get_ee_onee_AO(self, dens, exchange=True):
        assert len(dens.shape) == 2
        assert len(self.eeten.shape) == 4
        assert dens.shape[0] == dens.shape[1], 'Density matrix (problem with axes 0 and 1, all axis-dimensions must be the same!)'
        assert self.eeten.shape[0] == self.eeten.shape[1], 'ERIs (problem with axes 0 and 1, all axis-dimensions must be the same!)'
        assert self.eeten.shape[2] == self.eeten.shape[3], 'ERIs (problem with axes 2 and 3, all axis-dimensions must be the same!)'
        assert self.eeten.shape[0] == self.eeten.shape[2], 'ERIs (problem with axes 0 and 2, all axis-dimensions must be the same!)'
        e = True
        if (dens.shape[0] == self.eeten.shape[0]):
            nbas = dens.shape[0]
            vee_data = np.zeros((nbas, nbas), dtype=np.complex128)
            e = False
            if (exchange == True):
                for u in range(nbas):
                    for v in range(u,nbas):
                        for l in range(nbas):
                            for s in range(nbas):
                                # coulomb - 0.5*exchange
                                vee_data[u,v] += 2*dens[l,s]*(self.eeten[u,v,l,s])
                                vee_data[u,v] -= 2*dens[l,s]*(0.5*self.eeten[u,l,v,s])
                        vee_data[v,u] = np.conjugate(vee_data[u,v])
            elif (exchange == False):
                for u in range(nbas):
                    for v in range(u,nbas):
                        for l in range(nbas):
                            for s in range(nbas):
                                # coulomb
                                vee_data[u,v] += 2*dens[l,s]*(self.eeten[u,v,l,s])
                        vee_data[v,u] = np.conjugate(vee_data[u,v])
            return vee_data
        elif (e == True):
            print('\nError: Shapes of density and ERI tensors are not compatible.')
            return
    
    # compute and plot error between ML and true Hamiltonians on training set
    def plottrainhamerr(self):
        # this calculates the true Hamiltonian in the AO basis
        trueham = np.zeros((self.myham.shape[0],self.drc,self.drc), dtype=np.complex128)
        for i in range(self.myham.shape[0]):
            twoe = self.get_ee_onee_AO(self.denAO[i,:,:])
            tot = self.kinmat - self.enmat + twoe
            trueham[i,:,:] = tot

        truehamMO = np.zeros(trueham.shape,dtype=np.complex128)
        npts = trueham.shape[0]
        for i in range(1,npts):
            truehamMO[i,:,:] =  self.xmat.conj().T @ trueham[i,:,:] @ self.xmat

        ncp = self.myham.shape[0]-self.offset
        hamerr = np.zeros((ncp,self.drc,self.drc), dtype=np.complex128)
        for j in range(ncp):
            hamerr[j,:,:] = self.myham[j,:,:] + truehamMO[j+self.offset,:,:]  

        plt.plot(np.linalg.norm(hamerr, axis=(1,2)))
        plt.xlabel('time')
        plt.ylabel('|| H_true - H_ML ||')
        plt.savefig(self.outpath + 'hamiltonianerror.pdf')
        plt.close()
        return True    
        
    # EXACT deltakick Hamiltonian, NO FIELD
    # this function is defined for propagation purposes
    def EXhamrhs(self, t, pin):
        p = pin.reshape(self.drc,self.drc)
        
        pAO = self.xmat @ p @ self.xmat.conj().T
        twoe = self.get_ee_onee_AO(pAO)
        hAO = np.array(self.kinmat - self.enmat, dtype=np.complex128) + twoe
        h = -self.xmat.conj().T @ hAO @ self.xmat

        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)

    # MACHINE LEARNED deltakick Hamiltonian, NO FIELD
    # this function is defined for propagation purposes
    def MLhamrhs(self, t, pin):
        p = pin.reshape(self.drc,self.drc)
        
        # MACHINE LEARNED deltakick Hamiltonian
        pflat = np.zeros(self.ndof, dtype=np.complex128)
        for ij in self.nzreals:
            pflat[self.nzreals[ij]] = np.real(p[ij[0], ij[1]])
        for ij in self.nzimags:
            pflat[self.nzimags[ij]] = np.imag(p[ij[0], ij[1]])

        hflat = np.matmul(self.rgm(pflat, t)[:,0], self.blocktheta(self.theta))
        h = np.zeros((self.drc,self.drc), dtype=np.complex128)
        for ij in self.hamreals:
            h[ij[0], ij[1]] = hflat[self.hamreals[ij]]
            h[ij[1], ij[0]] = hflat[self.hamreals[ij]]
        for ij in self.hamimags:
            h[ij[0], ij[1]] += (1J)*hflat[self.hamimags[ij]]
            h[ij[1], ij[0]] -= (1J)*hflat[self.hamimags[ij]]
        
        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)

    # EXACT deltakick Hamiltonian, WITH FIELD ON
    # this function is defined for propagation purposes
    def EXhamwfrhs(self, t, pin):
        freq = 0.0428
        if t > 2*np.pi/freq:
            ez = 0
        elif t < 0:
            ez = 0
        else:
            ez = 0.05*np.sin(0.0428*t)

        hfieldAO = np.array(ez*self.didat[2], dtype=np.complex128)

        p = pin.reshape(self.drc,self.drc)
        pAO = self.xmat @ p @ self.xmat.conj().T
        twoe = self.get_ee_onee_AO(pAO)
        
        hAO = (np.array(self.kinmat - self.enmat, dtype=np.complex128) + twoe) + self.fieldsign * hfieldAO
        h = -self.xmat.conj().T @ hAO @ self.xmat

        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)

    # MACHINE LEARNED deltakick Hamiltonian, WITH FIELD ON
    # this function is defined for propagation purposes
    def MLhamwfrhs(self, t, pin):
        freq = 0.0428
        if t > 2*np.pi/freq:
            ez = 0
        elif t < 0:
            ez = 0
        else:
            ez = 0.05*np.sin(0.0428*t)

        hfieldAO = np.array(ez*self.didat[2], dtype=np.complex128)

        p = pin.reshape(self.drc,self.drc)

        # MACHINE LEARNED deltakick Hamiltonian
        pflat = np.zeros(self.ndof, dtype=np.complex128)
        for ij in self.nzreals:
            pflat[self.nzreals[ij]] = np.real(p[ij[0], ij[1]])
        for ij in self.nzimags:
            pflat[self.nzimags[ij]] = np.imag(p[ij[0], ij[1]])

        hflat = np.matmul(self.rgm(pflat, t)[:,0], self.blocktheta(self.theta)) # + thtMOvec
        h = np.zeros((self.drc,self.drc), dtype=np.complex128)
        for ij in self.hamreals:
            h[ij[0], ij[1]] = hflat[self.hamreals[ij]]
            h[ij[1], ij[0]] = hflat[self.hamreals[ij]]
        for ij in self.hamimags:
            h[ij[0], ij[1]] += (1J)*hflat[self.hamimags[ij]]
            h[ij[1], ij[0]] -= (1J)*hflat[self.hamimags[ij]]
        
        h -= self.fieldsign * self.xmat.conj().T @ hfieldAO @ self.xmat
        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)
    
    # propagate one method forward in time from self.offset to intpts = "integration points"
    # use initial condition given by initcond
    # use RK45 integration with relative and absolute tolerances set to mytol
    def propagate(self, rhsfunc, initcond, intpts=2000, mytol=1e-12):
        self.intpts = intpts
        self.tvec = self.dt*np.arange(intpts-self.offset)
        THISsol = si.solve_ivp(rhsfunc, [0, self.tvec[-1]], initcond, 'RK45', t_eval = self.tvec, rtol=mytol, atol=mytol)
        return THISsol.y

    # Andrew's method
    def fock_build(self,pin):
        p = pin.reshape(self.drc,self.drc)
        pAO = self.xmat @ p @ self.xmat.conj().T
        twoe = self.get_ee_onee_AO(pAO)
        hAO = np.array(self.kinmat - self.enmat, dtype=np.complex128) + twoe
        h = -self.xmat.conj().T @ hAO @ self.xmat
        return h
    
    # Andrew's method, rewritten slightly so that it does not rely on anything defined in propagate
    def MMUT_Prop(self, initial_density, intpts=2000):
        self.intpts = intpts
        ntvec = intpts-self.offset
        self.tvec = self.dt*np.arange(ntvec)
        propagated_dens = np.zeros((self.drc**2,ntvec), dtype=np.complex128)
        propagated_dens[:,0] = initial_density
        for i in range(1,ntvec):
            if i == 1:
                P_n_min_12 = initial_density
                P_n = P_n_min_12 + 0.5*self.dt*self.EXhamrhs(i,P_n_min_12)
                P_n = P_n.reshape(self.drc,self.drc)
                F_n = self.fock_build(P_n)
                P_n_min_12 =  P_n_min_12.reshape(self.drc,self.drc)
            else:
                P_n_min_12 = P_n_plus_12
                F_n = self.fock_build(P_n_plus_1)

            P_n_plus_12 = sl.expm(-1j*self.dt*F_n) @ P_n_min_12 @ sl.expm(1j*self.dt*F_n)
            propagated_dens[:,i] = P_n_plus_12.reshape(self.drc**2)
            P_n_plus_1 = sl.expm(-1j*self.dt/2*F_n) @  P_n_plus_12 @ sl.expm(1j*self.dt/2*F_n)

        return propagated_dens

    # think of traj1 and traj2 as two different numerical solutions that we got by running propagate
    # and groundtruth as the ground truth
    # here we compare the two trajectories QUANTITATIVELY
    def quantcomparetraj(self, traj1, traj2, groundtruth, fname='tdHamerr.npz'):

        errors = np.zeros(3)

        # error between propagating machine learned Hamiltonian and Gaussian data
        errors[0] = np.mean(np.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:], axis = (1,2) ))
        
        # error between propagating exact Hamiltonian and Gaussian data
        errors[1] = np.mean(np.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis = (1,2) ))
        
        # error between propagating exact Hamiltonian and propagating machine learned Hamiltonian
        errors[2] = np.mean(np.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - traj1.T.reshape((-1,self.drc,self.drc)), axis = (1,2) ))
        
        # compute and save time-dependent propagation errors 
        tdexHamerr = np.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis=(1,2))
        tdmlHamerr = np.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis=(1,2))
        tdexmlerr = np.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - traj1.T.reshape((-1,self.drc,self.drc)) , axis=(1,2))
        
        np.savez(self.outpath +self.mol+ fname,tdexHamerr=tdexHamerr,tdmlHamerr=tdmlHamerr,tdexmlerr=tdexmlerr)
        return errors

    # think of traj1 and traj2 as two different numerical solutions that we got by running propagate
    # and groundtruth as the ground truth
    # here we compare the two trajectories GRAPHICALLY
    def graphcomparetraj(self, traj1, traj2, groundtruth, myfigsize=(8,16), includeField=False, fname='prop.pdf', mytitle=None):

        fig = plt.figure(figsize=(myfigsize))
        mylabels = []
        if includeField:
            axs = fig.subplots(self.ndof+1)
            trueefield = 0.05*np.sin(0.0428*self.tvec)
            trueefield[1776:] = 0.
            axs[0].plot(self.tvec, trueefield, 'k-')
            thislabel = 'E-field'
            mylabels.append(thislabel)
            ctr = 1
        else:
            axs = fig.subplots(self.ndof)
            ctr = 0
        
        if mytitle == None:
            mytitle = 'Gaussian (black), exact-H (blue), and ML-H (red) propagation results'
        fig.suptitle(mytitle,y=0.9)

        for ij in self.nzreals:
            axs[ctr].plot(self.tvec, np.real(traj2.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'b-')
            axs[ctr].plot(self.tvec, np.real(traj1.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'r-')
            axs[ctr].plot(self.tvec, np.real(groundtruth[self.offset:self.intpts,ij[0],ij[1]]), 'k-')
            ijprime = (ij[0]+1, ij[1]+1)
            thislabel = 'Re(P_' + str(ijprime) + ')'
            mylabels.append(thislabel)
            ctr += 1
        
        for ij in self.nzimags:
            axs[ctr].plot(self.tvec, np.imag(traj2.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'b-')
            axs[ctr].plot(self.tvec, np.imag(traj1.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'r-')
            axs[ctr].plot(self.tvec, np.imag(groundtruth[self.offset:self.intpts,ij[0],ij[1]]), 'k-')
            ijprime = (ij[0]+1, ij[1]+1)
            thislabel = 'Im(P_' + str(ijprime) + ')'
            mylabels.append(thislabel)
            ctr += 1
        
        plt.subplots_adjust(wspace=0, hspace=0)

        cnt = 0
        for ax in axs.flat:
            ax.set(xlabel='time', ylabel=mylabels[cnt])
            if cnt % 2 == 0:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
            cnt += 1
        
        for ax in axs.flat:
            ax.label_outer()
        
        fig.savefig(self.outpath + self.mol + fname)
        plt.close()
        return True
        
    def fastthd(self, imagflag=False):
        thd = np.zeros((self.drc,self.drc,self.drc,self.drc), dtype=np.complex128)
        ssevecs = self.sevecs @ np.diag(self.sevals**(-0.5))
        for a in range(self.drc):
            for b in range(self.drc):
                for r in range(self.drc):
                    for z in range(self.drc):
                        testelt = np.outer( self.xmat[:,[a]] @ ssevecs[:,[r]].T, ssevecs[:,[z]] @ self.xmat[:,[b]].T )
                        testelt = testelt.reshape((self.drc,self.drc,self.drc,self.drc))
                        testelt = np.swapaxes(np.swapaxes(testelt, 1, 3), 2, 3)
                        if imagflag:
                            testelt *= self.myeeten2
                        else:
                            testelt *= self.myeeten
                        thd[a,b,r,z] += np.sum(testelt)
        if imagflag:
            return -thd
        else:
            return thd
    
    def computetruetheta(self):
        self.myeeten = np.zeros((self.drc,self.drc,self.drc,self.drc))
        self.myeeten2 = np.zeros((self.drc,self.drc,self.drc,self.drc))
        for u in range(self.drc):
            for v in range(self.drc):
                for l in range(self.drc):
                    for s in range(self.drc):
                        if u <= v:
                            self.myeeten[u,v,l,s] = 2*self.eeten[u,v,l,s] - self.eeten[u,l,v,s]
                            self.myeeten2[u,v,l,s] = -2*self.eeten[u,v,l,s] + self.eeten[u,l,v,s]
                        else:
                            self.myeeten[u,v,l,s] = 2*self.eeten[v,u,l,s] - self.eeten[v,l,u,s]
                            self.myeeten2[u,v,l,s] = 2*self.eeten[v,u,l,s] - self.eeten[v,l,u,s]
        
        self.truethetaR = np.zeros((self.nhamreals+1, self.nhamreals))
        constpart = self.xmat.conj().T @ (self.kinmat - self.enmat) @ self.xmat
        cnt = 0
        for ij in self.nzreals:
            self.truethetaR[0,cnt] = -np.real(constpart[ij[0], ij[1]])
            cnt += 1
        
        self.truethetaI = np.zeros((self.nhamimags, self.nhamimags))
        realthd = np.real(self.fastthd())
        imagthd = np.real(self.fastthd(True))
        row = 1
        col = 0
        for kl in self.nzreals:
            for ij in self.nzreals:
                if kl[0]==kl[1]:
                    self.truethetaR[row, col] = -realthd[ij[0],ij[1],kl[0],kl[1]]
                else:
                    self.truethetaR[row, col] = -realthd[ij[0],ij[1],kl[0],kl[1]] - realthd[ij[0],ij[1],kl[1],kl[0]]
                col += 1
            col = 0
            row += 1
        
        row = 0
        col = 0
        for kl in self.nzimags:
            for ij in self.nzimags:
                self.truethetaI[row, col] = -imagthd[ij[0],ij[1],kl[0],kl[1]] + imagthd[ij[0],ij[1],kl[1],kl[0]]
                col += 1
            col = 0
            row += 1
        
        self.truethetaflat = np.concatenate([self.truethetaR.reshape((-1)), self.truethetaI.reshape((-1))])
        return True

# this function is intended to be called by either map or a multiprocessing variant of map
def gradone(tu):
    gradelement = var_dict['shared']
    den = var_dict['den']
    xinp = var_dict['xinp']
    lh = var_dict['lh']
    #
    t = lh.nzrow[tu]
    u = lh.nzcol[tu]
    #
    term = np.zeros((4, lh.ntrain-2), dtype=np.complex128)
    #
    term[0,:] = np.sum(den[:,u,:] * lh.qmat[:,t,:],axis=1)
    term[1,:] = np.sum(den[:,t,:] * lh.qmat[:,u,:],axis=1)*(t < u)
    term[2,:] = -np.sum(den[:,:,t] * lh.qmat[:,:,u],axis=1)
    term[3,:] = -np.sum(den[:,:,u] * lh.qmat[:,:,t],axis=1)*(t < u)
    #
    for s in range(lh.ndof):
        # beta_0 derivative
        if s==0 and lh.nzrealm[t,u]>=0:
            row = lh.nzrealm[t,u]
            gradelement[row] = 2*np.real(np.sum(term))
        # beta_s derivative
        if s<lh.nnzr and lh.nzrealm[t,u]>=0:
            term1 = term*xinp[:,s]
            row = (s+1)*lh.nnzr + lh.nzrealm[t,u]
            gradelement[row] = 2*np.real(np.sum(term1))
        # gamma_s derivative
        if s>=lh.nnzr and lh.nzimagm[t,u]>=0:
            term2 = term*lh.stars_g*(1j)*xinp[:,s]
            row = (lh.nnzr)*lh.nnzr + (s-lh.nnzr)*lh.nnzi + lh.nzimagm[t,u]
            gradelement[row] = 2*np.real(np.sum(term2))
    
    return True

# this function is intended to be called by either map or a multiprocessing variant of map
def jacone(tu):
    jacelement = var_dict['shared']
    den = var_dict['den']
    xinp = var_dict['xinp']
    lh = var_dict['lh']
    #drc2 = lh.drc**2
    offset = den.shape[0]*lh.ndof
    #
    j = np.arange((lh.ntrain-2), dtype=np.int32)
    t = lh.nzrow[tu]
    u = lh.nzcol[tu]
    for s in range(lh.ndof):
        for m in range(lh.ndof):
            for n in range(lh.ndof):
                term = np.zeros((4, lh.ntrain-2), dtype=np.complex128)
                term[0,:] = den[:,u,n]*(u>=m)*(t==m)
                term[1,:] = den[:,t,n]*(t<m)*(u==m)
                term[2,:] = -den[:,m,t]*(t<=n)*(u==n)
                term[3,:] = -den[:,m,u]*(u>n)*(t==n)
                if lh.nzrealm[t,u] >= 0:
                    # beta_0 derivative
                    col = lh.nzrealm[t,u]
                    term0sum = np.sum(term,axis=0)
                    jacelement[j*lh.ndof + m*lh.nnzr + n, col] = np.real(term0sum)
                    jacelement[offset + j*lh.ndof + m*lh.nnzi + n, col] = np.imag(term0sum)
                # beta_s derivative
                if s < lh.nnzr and lh.nzrealm[t,u] >= 0:
                    col = (s+1)*lh.nnzr + lh.nzrealm[t,u]
                    term1 = term * xinp[:,s]
                    term1sum = np.sum(term1,axis=0)
                    jacelement[j*lh.ndof + m*lh.nnzr + n, col] = np.real(term1sum)
                    jacelement[offset + j*lh.ndof + m*lh.nnzi + n, col] = np.imag(term1sum)
                # gamma_s derivative
                if s >= lh.nnzr and lh.nzimagm[t,u] >= 0:
                    col = (lh.nnzr)*lh.nnzr + (s-lh.nnzr)*lh.nnzi + lh.nzimagm[t,u]
                    term2 = (1j)*(term*lh.stars_g) * xinp[:,s]
                    term2sum = np.sum(term2, axis=0)
                    jacelement[j*lh.ndof + m*lh.nnzr + n, col] = np.real(term2sum)
                    jacelement[offset + j*lh.ndof + m*lh.nnzi + n, col] = np.imag(term2sum)

    return True

# this function is intended to be called by either map or a multiprocessing variant of map
def hessone(iii): 
    hesselement = var_dict['shared']
    den = var_dict['den']
    xinp = var_dict['xinp']
    lh = var_dict['lh']
    #
    tu = iii // lh.nall
    bc = iii % lh.nall
    t = lh.nzrow[tu]
    u = lh.nzcol[tu]
    b = lh.nzrow[bc]
    c = lh.nzcol[bc]
    #
    term = np.zeros((16,lh.ntrain-2), dtype=np.complex128)
    #
    term[0,:] = np.sum(den[:,u,:]*np.conj(den[:,c,:]),axis=1)*(t==b)
    term[1,:] = np.sum(den[:,u,:]*np.conj(den[:,b,:]),axis=1)*(t==c)*(b<c)
    term[2,:] = -den[:,u,c]*np.conj(den[:,t,b])
    term[3,:] = -den[:,u,b]*np.conj(den[:,t,c])*(b<c)
    #
    term[4,:] = np.sum(den[:,t,:]*np.conj(den[:,c,:]),axis=1)*(u==b)*(t<u)
    term[5,:] = np.sum(den[:,t,:]*np.conj(den[:,b,:]),axis=1)*(u==c)*(t<u)*(b<c)
    term[6,:] = -den[:,t,c]*np.conj(den[:,u,b])*(t<u)
    term[7,:] = -den[:,t,b]*np.conj(den[:,u,c])*(t<u)*(b<c)
    #
    term[8,:] = -den[:,b,t]*np.conj(den[:,c,u])
    term[9,:] = -den[:,c,t]*np.conj(den[:,b,u])*(b<c)
    term[10,:] = np.sum(den[:,:,t]*np.conj(den[:,:,b]),axis=1)*(u==c)
    term[11,:] = np.sum(den[:,:,t]*np.conj(den[:,:,c]),axis=1)*(u==b)*(b<c)
    #
    term[12,:] = -den[:,b,u]*np.conj(den[:,c,t])*(t<u)
    term[13,:] = -den[:,c,u]*np.conj(den[:,b,t])*(b<c)*(t<u)
    term[14,:] = np.sum(den[:,:,u]*np.conj(den[:,:,b]),axis=1)*(t==c)*(t<u)
    term[15,:] = np.sum(den[:,:,u]*np.conj(den[:,:,c]),axis=1)*(t==b)*(t<u)*(b<c)
    #
    for s in range(lh.ndof):
        for a in range(lh.ndof):
            if s==0 and a==0 and lh.nzrealm[t,u]>=0 and lh.nzrealm[b,c]>=0:
                # no extra factor for 00 block
                row00 = lh.nzrealm[t,u]
                col00 = lh.nzrealm[b,c]
                hesselement[row00,col00] = 2*np.real(np.sum(term))
            if s==0 and a<lh.nnzr and lh.nzrealm[t,u]>=0 and lh.nzrealm[b,c]>=0:
                # work on the 01 block
                term01 = term*xinp[:,a]
                row01 = lh.nzrealm[t,u]
                col01 = (a+1)*lh.nnzr + lh.nzrealm[b,c]
                hesselement[row01,col01] = 2*np.real(np.sum(term01))
            if s==0 and a>=lh.nnzr and lh.nzrealm[t,u]>=0 and lh.nzimagm[b,c]>=0:
                # work on the 02 block
                # here we need to use the star pattern for index a
                term02 = term*lh.stars_a
                term02 = term02*(1j)*xinp[:,a]
                row02 = lh.nzrealm[t,u]
                col02 = lh.nnzr*lh.nnzr + (a-lh.nnzr)*lh.nnzi + lh.nzimagm[b,c]
                hesselement[row02,col02] = 2*np.real(np.sum(term02))
            if s<lh.nnzr and a<lh.nnzr and lh.nzrealm[t,u]>=0 and lh.nzrealm[b,c]>=0:
                # overall factor for 11 block
                term11 = term*xinp[:,s]*xinp[:,a]
                row11 = (s+1)*lh.nnzr + lh.nzrealm[t,u]
                col11 = (a+1)*lh.nnzr + lh.nzrealm[b,c]
                hesselement[row11,col11] = 2*np.real(np.sum(term11))
            if s<lh.nnzr and a>=lh.nnzr and lh.nzrealm[t,u]>=0 and lh.nzimagm[b,c]>=0:
                # work on the 12 block
                # here we need to use the star pattern for index a
                term12 = term*lh.stars_a
                term12 = term12*xinp[:,s]*(1j)*xinp[:,a]
                row12 = (s+1)*lh.nnzr + lh.nzrealm[t,u]
                col12 = (lh.nnzr)*lh.nnzr + (a-lh.nnzr)*lh.nnzi + lh.nzimagm[b,c]
                hesselement[row12,col12] = 2*np.real(np.sum(term12))
            if s>=lh.nnzr and a>=lh.nnzr and lh.nzimagm[t,u]>=0 and lh.nzimagm[b,c]>=0:
                # work on the 22 blcok
                # we need star patterns for both s and a indices
                term22 = term*lh.stars_sa
                term22 = term22*(1j)*xinp[:,s]*(1j)*xinp[:,a]
                row22 = (lh.nnzr)*lh.nnzr + (s-lh.nnzr)*lh.nnzi + lh.nzimagm[t,u]
                col22 = (lh.nnzr)*lh.nnzr + (a-lh.nnzr)*lh.nnzi + lh.nzimagm[b,c]
                hesselement[row22,col22] = 2*np.real(np.sum(term22))
    
    return True

def init_worker(X, den, xinp, lh):
    var_dict['shared'] = X
    var_dict['den'] = den
    var_dict['xinp'] = xinp
    var_dict['lh'] = lh
    
def singlethreadnumpy():
    # save the old values
    for k in threadkeys:
        try:
            var_dict[k] = os.environ[k]
        except KeyError:
            continue
            
    # set everything to "1" to be single-threaded
    for k in threadkeys:
        os.environ[k] = "1"
    
def restorethreadnumpy():
    # restore the old values that were saved
    for k in threadkeys:
        try:
            os.environ[k] = var_dict[k]
        except KeyError:
            continue

def computegrad(lh):
    # need to chop one time point off the top and bottom to match size of time-derivative
    denMOtrain = lh.denMO_train[1:(lh.ntrain-1),:,:]
    xinptrain = lh.x_inp_train[1:(lh.ntrain-1),:]

    singlethreadnumpy()

    # shape of gradient = lh.hesslen
    X = multiprocessing.RawArray('d', lh.hesslen)
    # Wrap X as a numpy array
    sharedgrad = np.frombuffer(X)
    # Copy data to our hsared array
    data = np.zeros(lh.hesslen)
    np.copyto(sharedgrad, data)

    start = time.time()
    pool = multiprocessing.Pool(initializer=init_worker,initargs=(sharedgrad, denMOtrain, xinptrain, lh))
    _ = pool.imap_unordered(gradone, range(lh.nall))
    pool.close()
    pool.join()
    end = time.time()
    print('Wall clock time to compute gradient: '+str(end-start))
    
    restorethreadnumpy()
    
    return sharedgrad.copy()

def computejac(lh):
    # need to chop one time point off the top and bottom to match size of time-derivative
    denMOtrain = lh.denMO_train[1:(lh.ntrain-1),:,:]
    xinptrain = lh.x_inp_train[1:(lh.ntrain-1),:]

    singlethreadnumpy()

    # shape of shared Jacobian = 'shsj'
    shsj = (2*denMOtrain.shape[0]*lh.ndof, lh.hesslen)
    X = multiprocessing.RawArray('d', shsj[0]*shsj[1])
    # Wrap X as a numpy array
    sharedjac = np.frombuffer(X).reshape(shsj)
    # Copy data to our hsared array
    data = np.zeros(shsj)
    np.copyto(sharedjac, data)

    start = time.time()
    pool = multiprocessing.Pool(initializer=init_worker,initargs=(sharedjac, denMOtrain, xinptrain, lh))
    _ = pool.imap_unordered(jacone, range(lh.nall))
    pool.close()
    pool.join()
    end = time.time()
    print('Wall clock time to compute Jacobian: '+str(end-start))
    
    restorethreadnumpy()
    
    nn = shsj[0]//2
    jacmat = -sharedjac[:nn,:] -(1j)*sharedjac[nn:,:]
    return jacmat

def computehess(lh):
    denMOtrain = lh.denMO_train[1:(lh.ntrain-1),:,:]
    xinptrain = lh.x_inp_train[1:(lh.ntrain-1),:]

    singlethreadnumpy()

    # shape of shared Hessian = 'shsh'
    shsh = (lh.hesslen, lh.hesslen)
    X = multiprocessing.RawArray('d', shsh[0]*shsh[1])
    # Wrap X as an numpy array
    sharedhess = np.frombuffer(X).reshape(shsh)
    # Copy data to our shared array
    data = np.zeros(shsh)
    np.copyto(sharedhess, data)

    start = time.time()
    pool = multiprocessing.Pool(initializer=init_worker,initargs=(sharedhess, denMOtrain, xinptrain, lh))
    _ = pool.imap_unordered(hessone, range(lh.nall**2))
    pool.close()
    pool.join()
    
    restorethreadnumpy()
    
    hessmat = sharedhess.copy()
    hessmat[lh.nnzr:,0:lh.nnzr] = hessmat[0:lh.nnzr, lh.nnzr:].T
    hessmat[(lh.nnzr*(lh.nnzr+1)):,lh.nnzr:(lh.nnzr*(lh.nnzr+1))] = hessmat[lh.nnzr:(lh.nnzr*(lh.nnzr+1)),(lh.nnzr*(lh.nnzr+1)):].T
    end = time.time()
    
    print('Wall clock time to compute Hessian: '+str(end-start))
    return hessmat

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
    # MLsol = mlham.propagate(mlham.MLhamrhs, mlham.denMOflat[mlham.offset,:], mytol=1e-6)
    MLsol = mlham.MMUT_Prop(mlham.denMOflat[mlham.offset,:])
    print(MLsol.shape)

    # propagate using Exact Hamiltonian with no field
    EXsol = mlham.propagate(mlham.EXhamrhs, mlham.denMOflat[mlham.offset,:], mytol=1e-6)

    # quantitatively and graphically compare the trajectories we just obtained against denMO
    # bigger figure for LiH
    err = mlham.quantcomparetraj(MLsol, EXsol, mlham.denMO)
    print("Field-off propagation error:")
    print(err)
    if mymol == 'lih':
        fs = (8,16)
    else:
        fs = (8,12)
    mlham.graphcomparetraj(MLsol, EXsol, mlham.denMO, fs)

    # propagate using ML Hamiltonian with field
    MLsolWF = mlham.propagate(mlham.MLhamwfrhs, mlham.fielddenMOflat[mlham.offset,:], mytol=1e-6)

    # propagate using Exact Hamiltonian with field
    EXsolWF = mlham.propagate(mlham.EXhamwfrhs, mlham.fielddenMOflat[mlham.offset,:], mytol=1e-6)
    
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
    mlham.graphcomparetraj(MLsolWF, EXsolWF, mlham.fielddenMO, fs, infl, 'propWF.pdf')

    # compute true theta
    mlham.computetruetheta()
    print("True theta:")
    print(mlham.truethetaflat)
    print("Loss when we plug in true theta:")
    print(mlham.newloss(mlham.truethetaflat))

