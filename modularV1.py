# object-oriented Python code
# 1) JAX gradient and Hessian
# 2) single trajectory
# 3) Hamiltonian real/imag parts depend on both real/imag parts of density
# 4) Hamiltonian has zeros in locations where density matrices have zeros; all other DOFs are active
# 5) least squares training

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import grad, jit, jacobian, vmap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as onp
import scipy.integrate as si
import scipy.linalg as sl

from functools import partial

class LearnHam:

    # class initializer
    def __init__(self, mol, outpath):
        
        # store the short string that indicates which molecule we're working on
        self.mol = mol
        
        # field sign correction
        if self.mol == 'h2':
            self.fieldsign = -1
        else:
            self.fieldsign = 1
            
        # store the path to output files, i.e., saved research outputs like figures
        self.outpath = outpath
    # load and process field-free data
    def load(self,inpath):
        # store the path to input files, i.e., training data, auxiliary matrices, etc
        inpath = inpath
        rawden = onp.load(inpath + 'td_dens_re+im_rt-tdexx_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)
        overlap = onp.load(inpath + 'ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)

        # put things into better variables
        self.kinmat = overlap['ke_data']
        self.enmat = overlap['en_data']
        self.eeten = overlap['ee_twoe_data']

        # need these for orthogonalization below
        s = overlap['overlap_data']
        self.sevals, self.sevecs = onp.linalg.eigh(s)
        self.xmat = self.sevecs @ onp.diag(self.sevals**(-0.5))
        
        
        # remove duplicates
        realpt = rawden['td_dens_re_data']
        imagpt = rawden['td_dens_im_data']
        den = realpt + 1j*imagpt
        self.drc = den.shape[1]
        # Read dipole data
        self.didat = [[]]*3
        self.didat[0] = onp.zeros(shape=(self.drc,self.drc))
        self.didat[1] = onp.zeros(shape=(self.drc,self.drc))
        self.didat[2] = onp.zeros(shape=(self.drc,self.drc))
        self.didat[0] = overlap['dipx_data']
        self.didat[1] = overlap['dipy_data']
        self.didat[2] = overlap['dipz_data']
        print('dipole',self.didat[2])
        denflat = den.reshape((-1,self.drc**2))
        dennodupflat = onp.array([onp.delete(denflat[:,i], onp.s_[101::100]) for i in range(self.drc**2)]).T
        self.denAO = dennodupflat.reshape((-1,self.drc,self.drc))

        # transform to MO using canonical orthogonalization
        # in this code, by "MO" we mean the canonical orthogonalization of the AO basis
        self.denMO = onp.zeros(self.denAO.shape,dtype=onp.complex128)
        self.denMOflat = self.denMO.reshape((-1,self.drc**2))
        onpts = self.denAO.shape[0]
        for i in range(1,onpts):
            self.denMO[i,:,:] = onp.diag(self.sevals**(0.5)) @ self.sevecs.T @ self.denAO[i,:,:] @ self.sevecs @ onp.diag(self.sevals**(0.5))

        # find off-diag DOFs of the supplied density matrices that are (sufficiently close to) zero across all time points
        self.realnzs = []
        self.imagnzs = []
        for j in range(realpt.shape[1]):
            for i in range(j+1):
                realnorm = onp.linalg.norm(onp.real(self.denMO[:,i,j]))
                # print("|| Re[den["+str(i)+","+str(j)+"]] || = " + str(realnorm))
                if not onp.isclose(realnorm,0):
                    self.realnzs.append((i,j))

                if i < j:
                    imagnorm = onp.linalg.norm(onp.imag(self.denMO[:,i,j]))
                    # print("|| Im[den["+str(i)+","+str(j)+"]] || = " + str(imagnorm))
                    if not onp.isclose(imagnorm,0):
                        self.imagnzs.append((i,j))

        # these turn out to be super useful when we build the ML Hamiltonian matrices much further down
        self.rnzl = [list(t) for t in zip(*self.realnzs)]
        self.inzl = [list(t) for t in zip(*self.imagnzs)]

        # build two dictionaries that help us find the absolute column number given human-readable (i,j) indices
        # for both the real and imaginary non-zero density DOFs
        # also build matrix equivalents for these dictionaries, which are needed by numba jit
        self.nzreals = {}
        self.nzrealm = -onp.ones((self.drc,self.drc),dtype=onp.int32)
        cnt = 0
        for i in self.realnzs:
            self.nzreals[i] = cnt
            self.nzrealm[i[0],i[1]] = cnt
            cnt += 1

        self.nzimags = {}
        self.nzimagm = -onp.ones((self.drc,self.drc),dtype=onp.int32)
        for i in self.imagnzs:
            self.nzimags[i] = cnt
            self.nzimagm[i[0],i[1]] = cnt
            cnt += 1

        # need all of the following for our fast Hessian assembler
        self.ndof = cnt
        self.allnzs = list(set(self.realnzs + self.imagnzs))
        self.nall = len(self.allnzs)
        self.nzrow = onp.zeros(self.nall, dtype=onp.int32)
        self.nzcol = onp.zeros(self.nall, dtype=onp.int32)
        for i in range(self.nall):
            self.nzrow[i] = self.allnzs[i][0]
            self.nzcol[i] = self.allnzs[i][1]
        
        # show that we got here
        return True
    
    # load and process data with field
    def loadfield(self,inpath):
        fielddata = onp.load(inpath + 'td_efield+dipole_rt-tdexx_ndlaser1cycs0_'+mol+'_sto-3g.npz')
        self.efdat = fielddata['td_efield_data']
        fielddens = onp.load(inpath + 'td_dens_re+im_rt-tdexx_ndlaser1cyc_s0_'+mol+'_sto-3g.npz',allow_pickle=True)
        self.fieldden = fielddens['td_dens_re_data'] + 1j*fielddens['td_dens_im_data']

        # change basis from AO to orthogonalization of AO (called MO here)
        fielddenMO = onp.zeros(self.fieldden.shape, dtype=onp.complex128)
        for i in range(self.fieldden.shape[0]):
            fielddenMO[i,:,:] = onp.diag(self.sevals**(0.5)) @ self.sevecs.T @ self.fieldden[i,:,:] @ self.sevecs @ onp.diag(self.sevals**(0.5))

        # remove duplicates
        fielddenMOflat = fielddenMO.reshape((-1,self.drc**2))

        # retain in the object only the stuff without duplicates
        self.fielddenMOflat = onp.array([onp.delete(fielddenMOflat[:,i], onp.s_[101::100]) for i in range(self.drc**2)]).T
        self.fielddenMO = self.fielddenMOflat.reshape((-1,self.drc,self.drc))

        # show that we got here
        return True
    
    # this function sets up the training and validation split
    # ntrain should be the desired length of the training set
    # tt should be the desired length of the training + validation set
    def trainsplit(self, ntrain=1000, tt=4000):
        x_inp_real = np.real(self.denMO)[:,self.rnzl[0], self.rnzl[1]]
        x_inp_imag = np.imag(self.denMO)[:,self.inzl[0], self.inzl[1]]
        self.x_inp = np.hstack([x_inp_real, x_inp_imag])

        self.offset = 2
        self.tt = tt
        self.ntrain = ntrain
        self.x_inp = self.x_inp[self.offset:(self.tt+self.offset),:]

        self.dt = 0.08268 
        self.tint_whole = np.arange(self.x_inp.shape[0])*self.dt

        # training set
        self.x_inp_train = self.x_inp[:ntrain,:]
        self.tint = self.tint_whole[:ntrain]

        # validation set
        self.x_inp_valid = self.x_inp[ntrain:,:]
        self.tint_valid = self.tint_whole[ntrain:]

        # show that we got here
        return True

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
        self.hamdof = self.ndof

        # set up matrices for initial training
        self.xdot = (self.x_inp_train[2:,:] - self.x_inp_train[:-2,:])/(2*self.dt)
        self.rgmmat = vmap(self.rgm)(self.x_inp_train[1:-1,:], self.tint[1:-1])[:,:,0]
    
    # compute regression matrix (rgm) on data set x at time t
    def rgm(self, x, t):
        # form regression matrix
        # start with constants and linears
        regmat0 = np.array([[1.]])
        regmat1 = np.expand_dims(x,1)
        reglist = [regmat0, regmat1]

        # include higher-order terms if needed
        for j in range(2, self.maxdeg+1):
            reglist.append(np.matmul(reglist[j-1], regmat1.T).reshape((self.ndof**j,1))/onp.math.factorial(j))

        # concatenate everybody
        regmat = np.concatenate(reglist)
        return regmat
    
    # this function takes as input a theta vector that parameterizes the Hamiltonian, and
    # outputs predicted values of xdot -- time derivatives of the time-dependent density matrix,
    # at each instant of time represented in the training data
    #
    # note that this function has to be run after buildmodel -- can include some error-checking on that...
    #
    def mypred(self, theta, extrgm=None):
        # each column of h represents one hamiltonian DOF, i.e.,
        # one element of either real(H_{kl}) or imag(H_{kl})
        if extrgm is None:
            rgmmat = self.rgmmat
        else:
            rgmmat = extrgm
        
        nc = np.reshape(theta,((self.nump, self.hamdof)))
        h = np.matmul(rgmmat, nc) # + thtMOvec
        rmodlist = []
        imodlist = []

        # we're interested here in the real density components \dot{P}_{km}^R
        for km in self.nzreals:
            k = km[0]
            m = km[1]
            rmod = np.zeros(rgmmat.shape[0])
            for l in range(self.drc):
                if l < k:
                    if (l,k) in self.hamreals and (l,m) in self.nzimags:  # outer term +
                        rmod += h[:, self.hamreals[(l,k)]]*rgmmat[:, 1+self.nzimags[(l,m)]]
                    if (l,k) in self.nzreals and (l,m) in self.hamimags:  # outer term -
                        rmod -= rgmmat[:, 1+self.nzreals[(l,k)]]*h[:, self.hamimags[(l,m)]]
                    if (l,k) in self.hamimags and (l,m) in self.nzreals:  # inner term +
                        rmod -= h[:, self.hamimags[(l,k)]]*rgmmat[:, 1+self.nzreals[(l,m)]]
                    if (l,k) in self.nzimags and (l,m) in self.hamreals:  # inner term -
                        rmod += rgmmat[:, 1+self.nzimags[(l,k)]]*h[:, self.hamreals[(l,m)]]
                elif k <= l and l <= m:
                    if (k,l) in self.hamreals and (l,m) in self.nzimags:  # outer term +
                        rmod += h[:, self.hamreals[(k,l)]]*rgmmat[:, 1+self.nzimags[(l,m)]]
                    if (k,l) in self.nzreals and (l,m) in self.hamimags:  # outer term -
                        rmod -= rgmmat[:, 1+self.nzreals[(k,l)]]*h[:, self.hamimags[(l,m)]]
                    if (k,l) in self.hamimags and (l,m) in self.nzreals:  # inner term +
                        rmod += h[:, self.hamimags[(k,l)]]*rgmmat[:, 1+self.nzreals[(l,m)]]
                    if (k,l) in self.nzimags and (l,m) in self.hamreals:  # inner term -
                        rmod -= rgmmat[:, 1+self.nzimags[(k,l)]]*h[:, self.hamreals[(l,m)]]
                else:
                    if (k,l) in self.hamreals and (m,l) in self.nzimags:  # outer term +
                        rmod -= h[:, self.hamreals[(k,l)]]*rgmmat[:, 1+self.nzimags[(m,l)]]
                    if (k,l) in self.nzreals and (m,l) in self.hamimags:  # outer term -
                        rmod += rgmmat[:, 1+self.nzreals[(k,l)]]*h[:, self.hamimags[(m,l)]]
                    if (k,l) in self.hamimags and (m,l) in self.nzreals:  # inner term +
                        rmod += h[:, self.hamimags[(k,l)]]*rgmmat[:, 1+self.nzreals[(m,l)]]
                    if (k,l) in self.nzimags and (m,l) in self.hamreals:  # inner term -
                        rmod -= rgmmat[:, 1+self.nzimags[(k,l)]]*h[:, self.hamreals[(m,l)]]

            rmodlist.append(rmod)

        # we're interested here in the imag density components \dot{P}_{km}^I
        for km in self.nzimags:
            k = km[0]
            m = km[1]
            imod = np.zeros(rgmmat.shape[0])  # remember to multiply by -1 at the end
            for l in range(self.drc):
                if l < k:
                    if (l,k) in self.hamreals and (l,m) in self.nzreals:  # first terms
                        imod += h[:, self.hamreals[(l,k)]]*rgmmat[:, 1+self.nzreals[(l,m)]]
                    if (l,k) in self.nzreals and (l,m) in self.hamreals:
                        imod -= rgmmat[:, 1+self.nzreals[(l,k)]]*h[:, self.hamreals[(l,m)]]
                    if (l,k) in self.hamimags and (l,m) in self.nzimags:  # last terms
                        imod += h[:, self.hamimags[(l,k)]]*rgmmat[:, 1+self.nzimags[(l,m)]]
                    if (l,k) in self.nzimags and (l,m) in self.hamimags:
                        imod -= rgmmat[:, 1+self.nzimags[(l,k)]]*h[:, self.hamimags[(l,m)]]
                elif k <= l and l <= m:
                    if (k,l) in self.hamreals and (l,m) in self.nzreals:  # first terms
                        imod += h[:, self.hamreals[(k,l)]]*rgmmat[:, 1+self.nzreals[(l,m)]]
                    if (k,l) in self.nzreals and (l,m) in self.hamreals:
                        imod -= rgmmat[:, 1+self.nzreals[(k,l)]]*h[:, self.hamreals[(l,m)]]
                    if (k,l) in self.hamimags and (l,m) in self.nzimags:  # last terms
                        imod -= h[:, self.hamimags[(k,l)]]*rgmmat[:, 1+self.nzimags[(l,m)]]
                    if (k,l) in self.nzimags and (l,m) in self.hamimags:
                        imod += rgmmat[:, 1+self.nzimags[(k,l)]]*h[:, self.hamimags[(l,m)]]
                else:
                    if (k,l) in self.hamreals and (m,l) in self.nzreals:  # first terms
                        imod += h[:, self.hamreals[(k,l)]]*rgmmat[:, 1+self.nzreals[(m,l)]]
                    if (k,l) in self.nzreals and (m,l) in self.hamreals:
                        imod -= rgmmat[:, 1+self.nzreals[(k,l)]]*h[:, self.hamreals[(m,l)]]
                    if (k,l) in self.hamimags and (m,l) in self.nzimags:  # last terms
                        imod += h[:, self.hamimags[(k,l)]]*rgmmat[:, 1+self.nzimags[(m,l)]]
                    if (k,l) in self.nzimags and (m,l) in self.hamimags:
                        imod -= rgmmat[:, 1+self.nzimags[(k,l)]]*h[:, self.hamimags[(m,l)]]

            imodlist.append(-imod)

        pred = np.vstack([rmodlist, imodlist]).T
        return pred

    # define the sum of squared errors loss function
    @partial(jit, static_argnums=(0,))
    def myloss(self, theta):
        pred = self.mypred(theta)
        loss = np.sum(np.square(self.xdot - pred))
        return loss
    
    # TRAIN AND SAVE theta TO DISK
    def trainmodel(self, savetodisk=True):
        # define the gradient and Hessian using JAX automatic differentiation
        jmyloss = jit(self.myloss)
        gradmyloss = jit(grad(self.myloss))
        hess = jit(jacobian(grad(self.myloss)))

        # compute the gradient and hessian at zero
        theta0 = np.zeros(self.nump*self.hamdof)
        rhs = onp.array(gradmyloss(theta0))
        hessmat = onp.array(hess(theta0))

        # solve least squares minimization problem
        self.theta,_,_,_ = sl.lstsq(hessmat,-rhs,lapack_driver='gelsy')

        # compute metrics
        self.trainloss = jmyloss(self.theta)
        self.gradloss = onp.linalg.norm(gradmyloss(self.theta))

        # save to disk
        if savetodisk:
            fname = self.outpath + 'hamiltoniantheta0.npz'
            onp.savez(fname, theta=self.theta)
        
        # show that we made it here
        return True
    
    # plot fit on training data and save to disk
    def plottrainfits(self):
        xdotpred = self.mypred(self.theta)
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
        rgmmat = vmap(self.rgm)(self.x_inp_valid[1:-1,:], self.tint_valid[1:-1])[:,:,0]
        xdotvalidpred = self.mypred(self.theta, extrgm=rgmmat)
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
        myhamraw = onp.array(onp.matmul(vmap(self.rgm)(self.x_inp_train, self.tint)[:,:,0], onp.reshape(self.theta,((self.nump, self.hamdof)))))
        self.myham = onp.zeros((myhamraw.shape[0],self.drc,self.drc), dtype=np.complex128)
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
            vee_data = onp.zeros((nbas, nbas), dtype=onp.complex128)
            e = False
            if (exchange == True):
                for u in range(nbas):
                    for v in range(u,nbas):
                        for l in range(nbas):
                            for s in range(nbas):
                                # coulomb - 0.5*exchange
                                vee_data[u,v] += 2*dens[l,s]*(self.eeten[u,v,l,s])
                                vee_data[u,v] -= 2*dens[l,s]*(0.5*self.eeten[u,l,v,s])
                        vee_data[v,u] = onp.conjugate(vee_data[u,v])
            elif (exchange == False):
                for u in range(nbas):
                    for v in range(u,nbas):
                        for l in range(nbas):
                            for s in range(nbas):
                                # coulomb
                                vee_data[u,v] += 2*dens[l,s]*(self.eeten[u,v,l,s])
                        vee_data[v,u] = onp.conjugate(vee_data[u,v])
            return vee_data
        elif (e == True):
            print('\nError: Shapes of density and ERI tensors are not compatible.')
            return
    
    # compute and plot error between ML and true Hamiltonians on training set
    def plottrainhamerr(self):
        # this calculates the true Hamiltonian in the AO basis
        trueham = onp.zeros((self.myham.shape[0],self.drc,self.drc), dtype=onp.complex128)
        for i in range(self.myham.shape[0]):
            twoe = self.get_ee_onee_AO(self.denAO[i,:,:])
            tot = self.kinmat - self.enmat + twoe
            trueham[i,:,:] = tot

        truehamMO = onp.zeros(trueham.shape,dtype=onp.complex128)
        npts = trueham.shape[0]
        for i in range(1,npts):
            truehamMO[i,:,:] =  self.xmat.conj().T @ trueham[i,:,:] @ self.xmat

        ncp = self.myham.shape[0]-self.offset
        hamerr = onp.zeros((ncp,self.drc,self.drc), dtype=onp.complex128)
        for j in range(ncp):
            hamerr[j,:,:] = self.myham[j,:,:] + truehamMO[j+self.offset,:,:]  

        plt.plot(onp.linalg.norm(hamerr, axis=(1,2)))
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
        hAO = onp.array(self.kinmat - self.enmat, dtype=onp.complex128) + twoe
        h = -self.xmat.conj().T @ hAO @ self.xmat

        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)

    # MACHINE LEARNED deltakick Hamiltonian, NO FIELD
    # this function is defined for propagation purposes
    def MLhamrhs(self, t, pin):
        p = pin.reshape(self.drc,self.drc)
        
        # MACHINE LEARNED deltakick Hamiltonian
        pflat = onp.zeros(self.ndof, dtype=np.complex128)
        for ij in self.nzreals:
            pflat[self.nzreals[ij]] = onp.real(p[ij[0], ij[1]])
        for ij in self.nzimags:
            pflat[self.nzimags[ij]] = onp.imag(p[ij[0], ij[1]])

        hflat = onp.matmul(self.rgm(pflat, t)[:,0], self.theta.reshape((self.nump,self.hamdof))) # + thtMOvec
        h = onp.zeros((self.drc,self.drc), dtype=np.complex128)
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
        if t > 2*onp.pi/freq:
            ez = 0
        elif t < 0:
            ez = 0
        else:
            ez = 0.05*onp.sin(0.0428*t)

        hfieldAO = onp.array(ez*self.didat[2], dtype=onp.complex128)

        p = pin.reshape(self.drc,self.drc)
        pAO = self.xmat @ p @ self.xmat.conj().T
        twoe = self.get_ee_onee_AO(pAO)
        
        hAO = (onp.array(self.kinmat - self.enmat, dtype=onp.complex128) + twoe) + self.fieldsign * hfieldAO
        h = -self.xmat.conj().T @ hAO @ self.xmat

        rhs = (h @ p - p @ h)/(1j)
        return rhs.reshape(self.drc**2)

    # MACHINE LEARNED deltakick Hamiltonian, WITH FIELD ON
    # this function is defined for propagation purposes
    def MLhamwfrhs(self, t, pin):
        freq = 0.0428
        if t > 2*onp.pi/freq:
            ez = 0
        elif t < 0:
            ez = 0
        else:
            ez = 0.05*onp.sin(0.0428*t)

        hfieldAO = onp.array(ez*self.didat[2], dtype=onp.complex128)

        p = pin.reshape(self.drc,self.drc)

        # MACHINE LEARNED deltakick Hamiltonian
        pflat = onp.zeros(self.ndof, dtype=np.complex128)
        for ij in self.nzreals:
            pflat[self.nzreals[ij]] = onp.real(p[ij[0], ij[1]])
        for ij in self.nzimags:
            pflat[self.nzimags[ij]] = onp.imag(p[ij[0], ij[1]])

        hflat = onp.matmul(self.rgm(pflat, t)[:,0], self.theta.reshape((self.nump,self.hamdof))) # + thtMOvec
        h = onp.zeros((self.drc,self.drc), dtype=np.complex128)
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
        self.tvec = self.dt*onp.arange(intpts-self.offset)
        THISsol = si.solve_ivp(rhsfunc, [0, self.tvec[-1]], initcond, 'RK45', t_eval = self.tvec, rtol=mytol, atol=mytol)
        return THISsol.y

    # think of traj1 and traj2 as two different numerical solutions that we got by running propagate
    # and groundtruth as the ground truth
    # here we compare the two trajectories QUANTITATIVELY
    def quantcomparetraj(self, traj1, traj2, groundtruth, fname='tdHamerr.npz'):

        errors = onp.zeros(3)

        # error between propagating machine learned Hamiltonian and Gaussian data
        errors[0] = onp.mean(onp.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:], axis = (1,2) ))
        
        # error between propagating exact Hamiltonian and Gaussian data
        errors[1] = onp.mean(onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis = (1,2) ))
        
        # error between propagating exact Hamiltonian and propagating machine learned Hamiltonian
        errors[2] = onp.mean(onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - traj1.T.reshape((-1,self.drc,self.drc)), axis = (1,2) ))
        
        # compute and save time-dependent propagation errors 
        tdexHamerr = onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis=(1,2))
        tdmlHamerr = onp.linalg.norm( traj1.T.reshape((-1,self.drc,self.drc)) - groundtruth[self.offset:self.intpts,:,:] , axis=(1,2))
        tdexmlerr = onp.linalg.norm( traj2.T.reshape((-1,self.drc,self.drc)) - traj1.T.reshape((-1,self.drc,self.drc)) , axis=(1,2))
        
        onp.savez(self.outpath +mol+ fname,tdexHamerr=tdexHamerr,tdmlHamerr=tdmlHamerr,tdexmlerr=tdexmlerr)
        return errors

    # think of traj1 and traj2 as two different numerical solutions that we got by running propagate
    # and groundtruth as the ground truth
    # here we compare the two trajectories GRAPHICALLY
    def graphcomparetraj(self, traj1, traj2, groundtruth, myfigsize=(8,16), includeField=False, fname='prop.pdf', mytitle=None):

        fig = plt.figure(figsize=(myfigsize))
        mylabels = []
        if includeField:
            axs = fig.subplots(self.ndof+1)
            trueefield = 0.05*onp.sin(0.0428*self.tvec)
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
            axs[ctr].plot(self.tvec, onp.real(traj2.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'b-')
            axs[ctr].plot(self.tvec, onp.real(traj1.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'r-')
            axs[ctr].plot(self.tvec, onp.real(groundtruth[self.offset:self.intpts,ij[0],ij[1]]), 'k-')
            ijprime = (ij[0]+1, ij[1]+1)
            thislabel = 'Re(P_' + str(ijprime) + ')'
            mylabels.append(thislabel)
            ctr += 1
        
        for ij in self.nzimags:
            axs[ctr].plot(self.tvec, onp.imag(traj2.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'b-')
            axs[ctr].plot(self.tvec, onp.imag(traj1.T.reshape((-1,self.drc,self.drc))[:,ij[0],ij[1]]), 'r-')
            axs[ctr].plot(self.tvec, onp.imag(groundtruth[self.offset:self.intpts,ij[0],ij[1]]), 'k-')
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
        
        fig.savefig(self.outpath + mol + fname)
        plt.close()
        return True


if __name__ == '__main__':
    mol = 'heh+'
    mlham = LearnHam(mol,'./'+mol+'LINEAR/')
    mlham.load('/Users/prachigupta/ml_tddft/scripts/data/heh+/sto-3g/extracted_data/')
    mlham.loadfield('/Users/prachigupta/ml_tddft/scripts/data/heh+/sto-3g/extracted_data/')
    mlham.trainsplit()
    mlham.buildmodel()
    print(mlham.rgmmat.shape)
    print(mlham.xdot.shape)

    mlham.trainmodel()
    print(mlham.trainloss)
    print(mlham.gradloss)
    
    mlham.plottrainfits()
    mlham.plotvalidfits()
    mlham.computeMLtrainham()
    mlham.plottrainhamerr()

    # propagate using ML Hamiltonian with no field
    MLsol = mlham.propagate(mlham.MLhamrhs, mlham.denMOflat[mlham.offset,:], mytol=1e-6)

    # propagate using Exact Hamiltonian with no field
    EXsol = mlham.propagate(mlham.EXhamrhs, mlham.denMOflat[mlham.offset,:], mytol=1e-6)

    # quantitatively and graphically compare the trajectories we just obtained against denMO
    # bigger figure for LiH
    err = mlham.quantcomparetraj(MLsol, EXsol, mlham.denMO)
    print(err)
    if mol == 'lih':
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
    print(errWF)
    if mol == 'lih':
        fs = (8,16)
        infl = False
    else:
        fs = (8,12)
        infl = True
    mlham.graphcomparetraj(MLsolWF, EXsolWF, mlham.fielddenMO, fs, infl, 'propWF.pdf')

