from hamcommon import *

# INSTEAD OF RETRAINING, LOAD SAVED theta FROM DISK
fname = savepath + 'hamiltoniantheta0.npz'
theta = onp.load(fname)['theta']
print(theta.shape)

# total number of parameters to represent ML Hamiltonian
lentheta = nump*hamdof

# t can be scalar
# x should be of length d
# beta should be of length lentheta
def oscG(x, t, beta):

    regmat = rgm(x, t).T

    # each column of h represents one hamiltonian DOF, i.e.,
    # one element of either real(H_{kl}) or imag(H_{kl})
    nc = np.reshape(beta,((nump, hamdof)))
    h = np.matmul(regmat, nc)
    rmodlist = []
    imodlist = []

    # we're interested here in the real density components \dot{P}_{km}^R
    for km in nzreals:
        k = km[0]
        m = km[1]
        rmod = np.zeros(regmat.shape[0])
        for l in range(drc):
            if l < k:
                if (l,k) in hamreals and (l,m) in nzimags:  # outer term +
                    rmod += h[:, hamreals[(l,k)]]*regmat[:, 1+nzimags[(l,m)]]
                if (l,k) in nzreals and (l,m) in hamimags:  # outer term -
                    rmod -= regmat[:, 1+nzreals[(l,k)]]*h[:, hamimags[(l,m)]]
                if (l,k) in hamimags and (l,m) in nzreals:  # inner term +
                    rmod -= h[:, hamimags[(l,k)]]*regmat[:, 1+nzreals[(l,m)]]
                if (l,k) in nzimags and (l,m) in hamreals:  # inner term -
                    rmod += regmat[:, 1+nzimags[(l,k)]]*h[:, hamreals[(l,m)]]
            elif k <= l and l <= m:
                if (k,l) in hamreals and (l,m) in nzimags:  # outer term +
                    rmod += h[:, hamreals[(k,l)]]*regmat[:, 1+nzimags[(l,m)]]
                if (k,l) in nzreals and (l,m) in hamimags:  # outer term -
                    rmod -= regmat[:, 1+nzreals[(k,l)]]*h[:, hamimags[(l,m)]]
                if (k,l) in hamimags and (l,m) in nzreals:  # inner term +
                    rmod += h[:, hamimags[(k,l)]]*regmat[:, 1+nzreals[(l,m)]]
                if (k,l) in nzimags and (l,m) in hamreals:  # inner term -
                    rmod -= regmat[:, 1+nzimags[(k,l)]]*h[:, hamreals[(l,m)]]
            else:
                if (k,l) in hamreals and (m,l) in nzimags:  # outer term +
                    rmod -= h[:, hamreals[(k,l)]]*regmat[:, 1+nzimags[(m,l)]]
                if (k,l) in nzreals and (m,l) in hamimags:  # outer term -
                    rmod += regmat[:, 1+nzreals[(k,l)]]*h[:, hamimags[(m,l)]]
                if (k,l) in hamimags and (m,l) in nzreals:  # inner term +
                    rmod += h[:, hamimags[(k,l)]]*regmat[:, 1+nzreals[(m,l)]]
                if (k,l) in nzimags and (m,l) in hamreals:  # inner term -
                    rmod -= regmat[:, 1+nzimags[(k,l)]]*h[:, hamreals[(m,l)]]

        rmodlist.append(rmod)

    # we're interested here in the imag density components \dot{P}_{km}^I
    for km in nzimags:
        k = km[0]
        m = km[1]
        imod = np.zeros(regmat.shape[0])  # remember to multiply by -1 at the end
        for l in range(drc):
            if l < k:
                if (l,k) in hamreals and (l,m) in nzreals:  # first terms
                    imod += h[:, hamreals[(l,k)]]*regmat[:, 1+nzreals[(l,m)]]
                if (l,k) in nzreals and (l,m) in hamreals:
                    imod -= regmat[:, 1+nzreals[(l,k)]]*h[:, hamreals[(l,m)]]
                if (l,k) in hamimags and (l,m) in nzimags:  # last terms
                    imod += h[:, hamimags[(l,k)]]*regmat[:, 1+nzimags[(l,m)]]
                if (l,k) in nzimags and (l,m) in hamimags:
                    imod -= regmat[:, 1+nzimags[(l,k)]]*h[:, hamimags[(l,m)]]
            elif k <= l and l <= m:
                if (k,l) in hamreals and (l,m) in nzreals:  # first terms
                    imod += h[:, hamreals[(k,l)]]*regmat[:, 1+nzreals[(l,m)]]
                if (k,l) in nzreals and (l,m) in hamreals:
                    imod -= regmat[:, 1+nzreals[(k,l)]]*h[:, hamreals[(l,m)]]
                if (k,l) in hamimags and (l,m) in nzimags:  # last terms
                    imod -= h[:, hamimags[(k,l)]]*regmat[:, 1+nzimags[(l,m)]]
                if (k,l) in nzimags and (l,m) in hamimags:
                    imod += regmat[:, 1+nzimags[(k,l)]]*h[:, hamimags[(l,m)]]
            else:
                if (k,l) in hamreals and (m,l) in nzreals:  # first terms
                    imod += h[:, hamreals[(k,l)]]*regmat[:, 1+nzreals[(m,l)]]
                if (k,l) in nzreals and (m,l) in hamreals:
                    imod -= regmat[:, 1+nzreals[(k,l)]]*h[:, hamreals[(m,l)]]
                if (k,l) in hamimags and (m,l) in nzimags:  # last terms
                    imod += h[:, hamimags[(k,l)]]*regmat[:, 1+nzimags[(m,l)]]
                if (k,l) in nzimags and (m,l) in hamimags:
                    imod -= regmat[:, 1+nzimags[(k,l)]]*h[:, hamimags[(m,l)]]

        imodlist.append(-imod)

    pred = np.vstack([rmodlist, imodlist]).T
    return pred[0,:]

# just-in-time (JIT) compiled version
foscG = jit(oscG)

# use automatic differentiation and JIT together
mygradoscG = jacobian(oscG, 0)
fmygradoscG = jit(mygradoscG)

# use automatic differentiation and JIT together
mygradoscGtheta = jacobian(oscG, 2)
fmygradoscGtheta = jit(mygradoscGtheta)

# for scipy.optimize
# z should have (xinit, curtheta)
# note that the t variable is **not** passed in as z[0]
def newlagwithgrad(xinit, curtheta):
    # solves the forward ODE using our current estimates of xinit and curtheta
    foscI = lambda y, t: foscG(y, t, curtheta)
    x = lax.stop_gradient(ode.odeint(foscI, t=tint, rtol=1e-9, atol=1e-9, y0=xinit)).T

    # set up adjoint ODE
    fadj = lambda y, t: -np.matmul(y, fmygradoscG(y, t, curtheta))
    icmat = np.eye(d)
    adjtint = np.array([0, dt])

    # function that solves the adjoint ODE once for one initial condition
    @jit
    def solonce(y0):
        adjsol = lax.stop_gradient(ode.odeint(fadj, t=adjtint, rtol=1e-9, atol=1e-9, y0=y0))
        return adjsol[1,:]

    # this is to solve the adjoint ODE for all initial conditions in the icmat **at once**
    propagator = vmap(solonce, in_axes=(0))(icmat) # + (1e-6)*np.eye(d)
    backprop = lax.stop_gradient(np.linalg.inv(propagator))

    yminusx = lax.stop_gradient(y - x)

    @jit
    def growlamb(i, lamb):
        lambplus = np.matmul(lamb[i,:], backprop)
        outlamb = index_update(lamb, index[i+1, :], lambplus + yminusx[:,(npts-2-i)])
        return outlamb

    initlamb = np.vstack([np.expand_dims(yminusx[:,(npts-1)],0), np.zeros((npts-1, d))])
    lambminus = lax.fori_loop(0, npts-1, growlamb, initlamb)

    # compute current value of lagrangian
    allxdot = np.hstack([(x[:,[1]]-x[:,[0]]), (x[:,2:] - x[:,:-2])/2, (x[:,[npts-2]]-x[:,[npts-3]])])/dt

    @jit
    def goodfun(i, lag):
        f = foscG(x[:, i], tint[i], curtheta)
        lag1 = lag + np.dot(lambminus[npts-1-i], allxdot[:,i]-f)*dt
        return lag1

    lag = lax.fori_loop(0, npts-1, goodfun, 0.0)
    lag += np.sum(np.square(x - y))/2.0

    # compute gradients using lamb (solution of adjoint ODE)
    # gradient of L with respect to parameters theta
    initgradtheta = np.zeros(lentheta)

    @jit
    def gt1i(i, gt):
        g = fmygradoscGtheta(x[:, i], tint[i], curtheta).reshape((d, lentheta))  # nabla_theta f
        gradtheta = gt - np.matmul(lambminus[npts-1-i],g)*dt
        return gradtheta

    gradtheta = lax.fori_loop(0, npts-1, gt1i, initgradtheta)
    gradx0 = -lambminus[npts-2]

    return lag, gradx0, gradtheta, x

lagwithgrad = jit(newlagwithgrad)

# adjoint solver with Nesterov accelerated gradient
y = x_inp_train.T

# take as initial guess x = y
key = random.PRNGKey(0)
theta0 = theta.copy()
x0 = y[:,0] # + (1e-8)*random.normal(key,(d,))

maxiters = 101
step = 1e-7

x = x0.copy()
theta = theta0.copy()

lag, gradx0, gradtheta, xest = lagwithgrad(x, theta)
print(onp.linalg.norm(gradtheta))

# train/optimize using Nesterov accelerated gradient
lambs = 0
nys = theta.copy()
for i in range(maxiters):
    # Nesterov
    lambsplus1 = 0.5*(1.0 + onp.sqrt(1 + 4*lambs**2))
    gammas = (1.0 - lambs)/lambsplus1

    lag, gradx0, gradtheta, xest = lagwithgrad(x, theta)
    if i % 100 == 0:
        fname = savepath + 'trainvalidITER' + str(i) + '.npz'
        onp.savez(fname,x=x,theta=theta)
        trainerr = np.mean(np.abs(xest.T - x_inp_train)**2, axis=1)
        print("iteration " + str(i) + " ; lag = " + str(lag) + " ; trainerr = " + str(np.mean(trainerr)))
        plt.figure(figsize=(6,6))
        plt.plot(tint,trainerr)
        plt.xlabel('time')
        plt.ylabel('training MSE')
        plt.savefig(savepath +'adjointTRAINMSE_' + str(i) + '.pdf')
        plt.close()
        xpred = lax.stop_gradient(ode.odeint(lambda y, t: foscG(y, t, theta), t=tint_whole, rtol=1e-12, atol=1e-12, y0=x)).T
        validerr = np.mean(np.square(x_inp_valid - xpred.T[ntrain:,]))
        print("validerr = " + str(validerr))
        fig, axs = plt.subplots(d)
        fig.tight_layout()
        cnt = 0
        for ij in nzreals:
            axs[cnt].plot(tint_valid, x_inp_valid[:,cnt], 'k-', linewidth=0.5)
            axs[cnt].plot(tint_valid, xpred.T[ntrain:,cnt], 'r-', linewidth=0.5)
            ijprime = (ij[0]+1, ij[1]+1)
            thistitle = 'Re(P_' + str(ijprime) + ')'
            axs[cnt].set_title(thistitle)
            cnt += 1

        for ij in nzimags:
            axs[cnt].plot(tint_valid, x_inp_valid[:,cnt], 'k-', linewidth=0.5)
            axs[cnt].plot(tint_valid, xpred.T[ntrain:,cnt], 'r-', linewidth=0.5)
            ijprime = (ij[0]+1, ij[1]+1)
            thistitle = 'Im(P_' + str(ijprime) + ')'
            axs[cnt].set_title(thistitle)
            cnt += 1

        for ax in axs.flat:
            ax.set(xlabel='time')

        for ax in axs.flat:
            ax.label_outer()

        fig.savefig(savepath + 'adjointVALID_' + str(i) + '.pdf')
        plt.close()

        plt.figure(figsize=(12,6))
        plt.plot(tint_valid, np.mean(np.abs(xpred.T[ntrain:,] - x_inp_valid)**2, axis=1))
        plt.xlabel('time')
        plt.ylabel('validation MSE')
        plt.savefig(savepath +'adjointVALIDMSE_' + str(i) + '.pdf')
        plt.close()

    # Nesterov
    nysp1 = theta - step*gradtheta # .reshape((3, nump))
    theta = (1-gammas)*nysp1 + gammas*nys
    nys = nysp1
    lambs = lambsplus1

    # SINDY-style hard-thresholding
    # mys = 1e-2
    # theta = onp.array(theta)
    # theta[np.abs(theta) <= mys] = 0.0
    # theta = np.array(theta)

fname = savepath + 'trainvalidFINAL.npz'
onp.savez(fname,x=x,theta=theta)

fname = savepath + 'hamiltoniantheta1.npz'
onp.savez(fname,theta=theta)


