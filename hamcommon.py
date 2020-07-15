from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
import jax.nn
from jax import grad, jit, jacobian, random, vmap, lax
from jax.ops import index, index_update
from jax.experimental import ode

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as onp
import scipy.integrate as si

mol = 'lih'
savepath = './'+mol+'LINEAR/'
rawden = onp.load('./td_dens_re+im_rt-tdexx_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)

if mol=='heh+' or mol=='c2h4':
    overlap = onp.load('./ke+en+overlap+ee_twoe+dip_hf_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)
else:
    overlap = onp.load('./ke+en+overlap+ee_twoe_hf_delta_s0_'+mol+'_sto-3g.npz',allow_pickle=True)

# putting things into better variables
kinmat = overlap['ke_data']
enmat = overlap['en_data']
eeten = overlap['ee_twoe_data']

# transform to MO using canonical orthogonalization
s = overlap['overlap_data']
realpt = rawden['td_dens_re_data']
imagpt = rawden['td_dens_im_data']
den = realpt + 1j*imagpt

npts = den.shape[0]
hermit = onp.zeros(npts)
for i in range(1,npts):
    hermit[i] = onp.linalg.norm(den[i,:,:] - den[i,:,:].conj().T)

print('|| P - P.conj().T || = ', onp.mean(hermit))
print('')

# delete duplicated entries in time series
denAOnorms = onp.linalg.norm(np.diff(den,axis=0),axis=(1,2))
plt.plot(denAOnorms[1:1000])
plt.savefig('denAOnorms.pdf')
plt.close()

drc = den.shape[1]
denflat = den.reshape((-1,drc**2))
dennodupflat = onp.array([onp.delete(denflat[:,i], onp.s_[101::100]) for i in range(drc**2)]).T
denAOnodup = dennodupflat.reshape((-1,drc,drc))
print('shape of AO densities after removing duplicates = ', denAOnodup.shape)
denAOnodupnorms = onp.linalg.norm(np.diff(denAOnodup,axis=0),axis=(1,2))
plt.plot(denAOnodupnorms[1:1000])
plt.savefig('denAOnodupnorms.pdf')
plt.close()

sevals, sevecs = onp.linalg.eigh(s)

print('sevecs.T @ sevecs = ')
print(sevecs.T @ sevecs)
print('')
print('sevecs.conj() - sevecs = ')
print(sevecs.conj() - sevecs)
print('')

denMO = onp.zeros(denAOnodup.shape,dtype=np.complex128)
denMOflat = denMO.reshape((-1,drc**2))
npts = denAOnodup.shape[0]
idempot = onp.zeros(npts)
traces = onp.zeros(npts)
hermit = onp.zeros(npts)
for i in range(1,npts):
    denMO[i,:,:] = onp.diag(sevals**(0.5)) @ sevecs.T @ denAOnodup[i,:,:] @ sevecs @ onp.diag(sevals**(0.5))
    
    # check if MO density is idempotent
    idempot[i] = onp.linalg.norm(denMO[i,:,:] @ denMO[i,:,:] - denMO[i,:,:])
    
    # compute trace of MO density
    traces[i] = onp.real(onp.trace(denMO[i,:,:]))
    
    # to check if MO density is hermitian
    hermit[i] = onp.linalg.norm(denMO[i,:,:] - denMO[i,:,:].conj().T)

# this should be close to 0 if denMO is Hermitian
print('|| P_{MO} - P_{MO}.conj().T || = ', onp.mean(hermit))
print('')

# this should be close to 1, half the number of electrons in the system
print('mean(trace(P_{MO})) = ', onp.mean(traces))
print('')

# this should be close to 0 if denMO is idempotent
print('|| P_{MO} . P_{MO} - P_{MO} || = ', onp.mean(idempot))

denMOnorms = onp.linalg.norm(np.diff(denMO,axis=0),axis=(1,2))
plt.plot(denMOnorms[1:1000])
plt.savefig('denMOnorms.pdf')
plt.close()

# find off-diag DOFs that are zero
realnzs = []
imagnzs = []

for j in range(realpt.shape[1]):
    for i in range(j+1):
        realnorm = onp.linalg.norm(onp.real(denMO[:,i,j]))
        print("|| Re[den["+str(i)+","+str(j)+"]] || = " + str(realnorm))
        if not onp.isclose(realnorm,0):
            realnzs.append((i,j))

        if i < j:
            imagnorm = onp.linalg.norm(onp.imag(denMO[:,i,j]))
            print("|| Im[den["+str(i)+","+str(j)+"]] || = " + str(imagnorm))
            if not onp.isclose(imagnorm,0):
                imagnzs.append((i,j))

rnzl = [list(t) for t in zip(*realnzs)]
inzl = [list(t) for t in zip(*imagnzs)]
print('Shape of denMO on real non-zero degrees of freedom:')
print(onp.real(denMO)[:,rnzl[0], rnzl[1]].shape)
print('Shape of denMO on imag non-zero degrees of freedom:')
print(onp.imag(denMO)[:,inzl[0], inzl[1]].shape)

# build two dictionaries that help us find the absolute column number given human-readable (i,j) indices
# for both the real and imaginary non-zero density DOFs
nzreals = {}
cnt = 0
for i in realnzs:
    nzreals[i] = cnt
    cnt += 1

nzimags = {}
for i in imagnzs:
    nzimags[i] = cnt
    cnt += 1

print('nzreals:')
print(nzreals)
print('nzimags:')
print(nzimags)
ndof = cnt
print('ndof: ', ndof)

# build two dictionaries that help us find the absolute column number given human-readable (i,j) indices
# for both the real and imaginary non-zero Hamiltonian DOFs
"""
hamreals = {}
hamimags = {}
cnt = 0
for i in range(denMO.shape[1]):
    for j in range(i, denMO.shape[1]):
       hamreals[(i,j)] = cnt
       cnt += 1

for i in range(denMO.shape[1]):
    for j in range(i+1, denMO.shape[1]):
       hamimags[(i,j)] = cnt
       cnt += 1
"""
hamreals = nzreals.copy()
hamimags = nzimags.copy()

print('hamreals:')
print(hamreals)
print('hamimags:')
print(hamimags)
hamdof = cnt
print('hamdof: ', hamdof)

# set up training data
x_inp_real = np.real(denMO)[:,rnzl[0], rnzl[1]]
x_inp_imag = np.imag(denMO)[:,inzl[0], inzl[1]]
x_inp = np.hstack([x_inp_real, x_inp_imag])

offset = 2
tt = 4000
x_inp = x_inp[offset:(tt+offset),:]

dt = 0.08268 
npts = x_inp.shape[0]
tint_whole = np.arange(npts)*dt

if mol == 'c2h4':
    ntrain = 2000
else:
    ntrain = 1000

nvalid = npts - ntrain

x_inp_train = x_inp[:ntrain,:]
tint = tint_whole[:ntrain]

x_inp_valid = x_inp[ntrain:,:]
tint_valid = tint_whole[ntrain:]

# d = ndof = number of degrees of freedom
d = ndof

# set maximum polynomial degree (1 = linear, 2 = quadratic, etc)
maxdeg = 1

# compute total number of parameters per dimension
nump = 1
for j in range(1,maxdeg+1):
    nump += d**j

print('nump: ', nump)

def rgm(x, t):
    # form regression matrix
    # start with constants and linears
    regmat0 = np.array([[1.]])
    regmat1 = np.expand_dims(x,1)
    reglist = [regmat0, regmat1]

    # include higher-order terms if needed
    for j in range(2, maxdeg+1):
        reglist.append(np.matmul(reglist[j-1], regmat1.T).reshape((d**j,1))/onp.math.factorial(j))

    # concatenate everybody
    regmat = np.concatenate(reglist)
    return regmat

# set up matrices for initial training
xdot = (x_inp_train[2:,:] - x_inp_train[:-2,:])/(2*dt)
rgmmat = vmap(rgm)(x_inp_train[1:-1,:], tint[1:-1])[:,:,0]

def mypred(theta):
    # each column of h represents one hamiltonian DOF, i.e.,
    # one element of either real(H_{kl}) or imag(H_{kl})
    nc = np.reshape(theta,((nump, hamdof)))
    h = np.matmul(rgmmat, nc) # + thtMOvec
    rmodlist = []
    imodlist = []

    # we're interested here in the real density components \dot{P}_{km}^R
    for km in nzreals:
        k = km[0]
        m = km[1]
        rmod = np.zeros(rgmmat.shape[0])
        for l in range(drc):
            if l < k:
                if (l,k) in hamreals and (l,m) in nzimags:  # outer term +
                    rmod += h[:, hamreals[(l,k)]]*rgmmat[:, 1+nzimags[(l,m)]]
                if (l,k) in nzreals and (l,m) in hamimags:  # outer term -
                    rmod -= rgmmat[:, 1+nzreals[(l,k)]]*h[:, hamimags[(l,m)]]
                if (l,k) in hamimags and (l,m) in nzreals:  # inner term +
                    rmod -= h[:, hamimags[(l,k)]]*rgmmat[:, 1+nzreals[(l,m)]]
                if (l,k) in nzimags and (l,m) in hamreals:  # inner term -
                    rmod += rgmmat[:, 1+nzimags[(l,k)]]*h[:, hamreals[(l,m)]]
            elif k <= l and l <= m:
                if (k,l) in hamreals and (l,m) in nzimags:  # outer term +
                    rmod += h[:, hamreals[(k,l)]]*rgmmat[:, 1+nzimags[(l,m)]]
                if (k,l) in nzreals and (l,m) in hamimags:  # outer term -
                    rmod -= rgmmat[:, 1+nzreals[(k,l)]]*h[:, hamimags[(l,m)]]
                if (k,l) in hamimags and (l,m) in nzreals:  # inner term +
                    rmod += h[:, hamimags[(k,l)]]*rgmmat[:, 1+nzreals[(l,m)]]
                if (k,l) in nzimags and (l,m) in hamreals:  # inner term -
                    rmod -= rgmmat[:, 1+nzimags[(k,l)]]*h[:, hamreals[(l,m)]]
            else:
                if (k,l) in hamreals and (m,l) in nzimags:  # outer term +
                    rmod -= h[:, hamreals[(k,l)]]*rgmmat[:, 1+nzimags[(m,l)]]
                if (k,l) in nzreals and (m,l) in hamimags:  # outer term -
                    rmod += rgmmat[:, 1+nzreals[(k,l)]]*h[:, hamimags[(m,l)]]
                if (k,l) in hamimags and (m,l) in nzreals:  # inner term +
                    rmod += h[:, hamimags[(k,l)]]*rgmmat[:, 1+nzreals[(m,l)]]
                if (k,l) in nzimags and (m,l) in hamreals:  # inner term -
                    rmod -= rgmmat[:, 1+nzimags[(k,l)]]*h[:, hamreals[(m,l)]]

        rmodlist.append(rmod)

    # we're interested here in the imag density components \dot{P}_{km}^I
    for km in nzimags:
        k = km[0]
        m = km[1]
        imod = np.zeros(rgmmat.shape[0])  # remember to multiply by -1 at the end
        for l in range(drc):
            if l < k:
                if (l,k) in hamreals and (l,m) in nzreals:  # first terms
                    imod += h[:, hamreals[(l,k)]]*rgmmat[:, 1+nzreals[(l,m)]]
                if (l,k) in nzreals and (l,m) in hamreals:
                    imod -= rgmmat[:, 1+nzreals[(l,k)]]*h[:, hamreals[(l,m)]]
                if (l,k) in hamimags and (l,m) in nzimags:  # last terms
                    imod += h[:, hamimags[(l,k)]]*rgmmat[:, 1+nzimags[(l,m)]]
                if (l,k) in nzimags and (l,m) in hamimags:
                    imod -= rgmmat[:, 1+nzimags[(l,k)]]*h[:, hamimags[(l,m)]]
            elif k <= l and l <= m:
                if (k,l) in hamreals and (l,m) in nzreals:  # first terms
                    imod += h[:, hamreals[(k,l)]]*rgmmat[:, 1+nzreals[(l,m)]]
                if (k,l) in nzreals and (l,m) in hamreals:
                    imod -= rgmmat[:, 1+nzreals[(k,l)]]*h[:, hamreals[(l,m)]]
                if (k,l) in hamimags and (l,m) in nzimags:  # last terms
                    imod -= h[:, hamimags[(k,l)]]*rgmmat[:, 1+nzimags[(l,m)]]
                if (k,l) in nzimags and (l,m) in hamimags:
                    imod += rgmmat[:, 1+nzimags[(k,l)]]*h[:, hamimags[(l,m)]]
            else:
                if (k,l) in hamreals and (m,l) in nzreals:  # first terms
                    imod += h[:, hamreals[(k,l)]]*rgmmat[:, 1+nzreals[(m,l)]]
                if (k,l) in nzreals and (m,l) in hamreals:
                    imod -= rgmmat[:, 1+nzreals[(k,l)]]*h[:, hamreals[(m,l)]]
                if (k,l) in hamimags and (m,l) in nzimags:  # last terms
                    imod += h[:, hamimags[(k,l)]]*rgmmat[:, 1+nzimags[(m,l)]]
                if (k,l) in nzimags and (m,l) in hamimags:
                    imod -= rgmmat[:, 1+nzimags[(k,l)]]*h[:, hamimags[(m,l)]]

        imodlist.append(-imod)

    pred = np.vstack([rmodlist, imodlist]).T
    return pred

# note that the loss function is the SUM of squared errors here,
# this is intentional to magnify the error
# MSE looks artificially low because of the large denominator
def myloss(theta):
    pred = mypred(theta)
    loss = np.sum(np.square(xdot - pred))
    return loss

jmyloss = jit(myloss)
gradmyloss = jit(grad(myloss))


