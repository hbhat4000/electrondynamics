from hamcommon import *

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
# weights = onp.linalg.norm(x_inp_train,axis=0)**(-2)
# weights = onp.var(x_inp_train,axis=0)**(-1.0)
weights = onp.ones(d)
print("weights:")
print(weights)
def myloss(theta):
    pred = mypred(theta)
    loss = np.sum(weights*np.square(xdot - pred))
    return loss

jmyloss = jit(myloss)
gradmyloss = jit(grad(myloss))
hess = jit(jacobian(grad(myloss)))

# TRAIN AND SAVE theta TO DISK
theta0 = np.zeros(nump*hamdof)

# pseudoinverse
# rhs = onp.array(gradmyloss(theta0)/-2)
# hessmat = onp.array(hess(theta0))
# theta = onp.linalg.pinv(0.5*hessmat) @ rhs

# least squares solve
# note we could try doing this with scipy.linalg as well
rhs = onp.array(gradmyloss(theta0))
hessmat = onp.array(hess(theta0))
theta,_,_,_ = onp.linalg.lstsq(hessmat,-rhs,rcond=-1)
print('Training loss: ', myloss(theta))
print('|| Grad(loss) || at theta: ', onp.linalg.norm(gradmyloss(theta)))

if mol == 'h2' or mol == 'heh+':
    test = (onp.eye(hessmat.shape[0]) - hessmat @ onp.linalg.pinv(hessmat)) @ rhs
    print(test)
    print(onp.linalg.norm(test))

fname = savepath + 'hamiltoniantheta0.npz'
onp.savez(fname, theta=theta)

xdot = (x_inp_train[2:,:] - x_inp_train[:-2,:])/(2*dt)
rgmmat = vmap(rgm)(x_inp_train[1:-1,:], tint[1:-1])[:,:,0]
xdotpred = mypred(theta)
print("New training fits:")
for i in range(ndof):
    plt.figure(figsize=(6,6))
    plt.plot(tint[1:-1], xdot[:,i])
    plt.plot(tint[1:-1], xdotpred[:,i])
    plt.xlabel('time')
    plt.ylabel('x[' + str(i) + ']')
    plt.savefig(savepath + 'prefitTRAIN1' + str(i) + '.pdf')
    plt.close()

xdotvalid = (x_inp_valid[2:,:] - x_inp_valid[:-2,:])/(2*dt)
rgmmat = vmap(rgm)(x_inp_valid[1:-1,:], tint_valid[1:-1])[:,:,0]
xdotvalidpred = mypred(theta)
print("New validation fits:")
for i in range(ndof):
    plt.figure(figsize=(18,6))
    plt.plot(tint_valid[1:-1], xdotvalid[:,i])
    plt.plot(tint_valid[1:-1], xdotvalidpred[:,i])
    plt.xlabel('time')
    plt.ylabel('x[' + str(i) + ']')
    plt.savefig(savepath + 'prefitVALID1' + str(i) + '.pdf')
    plt.close()




