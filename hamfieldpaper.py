from hamcommon import *

mytol = 1e-12

# INSTEAD OF RETRAINING, LOAD SAVED theta FROM DISK
fname = savepath + 'hamiltoniantheta0.npz'
theta = onp.load(fname)['theta']
print(theta.shape)

myhamraw = onp.array(np.matmul(vmap(rgm)(x_inp_train, tint)[:,:,0], np.reshape(theta,((nump, hamdof))))) # + onp.array(thtMOvec)
myham = onp.zeros((myhamraw.shape[0],drc,drc), dtype=np.complex128)
for ij in hamreals:
    myham[:, ij[0], ij[1]] = myhamraw[:, hamreals[ij]]
    myham[:, ij[1], ij[0]] = myhamraw[:, hamreals[ij]]
for ij in hamimags:
    myham[:, ij[0], ij[1]] += (1J)*myhamraw[:, hamimags[ij]]
    myham[:, ij[1], ij[0]] -= (1J)*myhamraw[:, hamimags[ij]]

# Karnamohit's function (July 1 version)
def get_ee_onee_AO(dens, ee_twoe, exchange=True):
    assert len(dens.shape) == 2
    assert len(ee_twoe.shape) == 4
    assert dens.shape[0] == dens.shape[1], 'Density matrix (problem with axes 0 and 1, all axis-dimensions must be the same!)'
    assert ee_twoe.shape[0] == ee_twoe.shape[1], 'ERIs (problem with axes 0 and 1, all axis-dimensions must be the same!)'
    assert ee_twoe.shape[2] == ee_twoe.shape[3], 'ERIs (problem with axes 2 and 3, all axis-dimensions must be the same!)'
    assert ee_twoe.shape[0] == ee_twoe.shape[2], 'ERIs (problem with axes 0 and 2, all axis-dimensions must be the same!)'
    e = True
    if (dens.shape[0] == ee_twoe.shape[0]):
        nbas = dens.shape[0]
        vee_data = onp.zeros((nbas, nbas), dtype=onp.complex128)
        e = False
        if (exchange == True):
            for u in range(nbas):
                for v in range(u,nbas):
                    for l in range(nbas):
                        for s in range(nbas):
                            # coulomb - 0.5*exchange
                            vee_data[u,v] += 2*dens[l,s]*(ee_twoe[u,v,l,s])
                            vee_data[u,v] -= 2*dens[l,s]*(0.5*ee_twoe[u,l,v,s])
                    vee_data[v,u] = onp.conjugate(vee_data[u,v])
        elif (exchange == False):
            for u in range(nbas):
                for v in range(u,nbas):
                    for l in range(nbas):
                        for s in range(nbas):
                            # coulomb
                            vee_data[u,v] += 2*dens[l,s]*(ee_twoe[u,v,l,s])
                    vee_data[v,u] = onp.conjugate(vee_data[u,v])
        return vee_data
    elif (e == True):
        print('\nError: Shapes of density and ERI tensors are not compatible.')
        return

# this calculates the true Hamiltonian in the AO basis
trueham = onp.zeros((myham.shape[0],drc,drc), dtype=onp.complex128)
for i in range(myham.shape[0]):
    twoe = get_ee_onee_AO(denAOnodup[i,:,:], eeten)
    tot = kinmat - enmat + twoe
    trueham[i,:,:] = tot

truehamMO = onp.zeros(trueham.shape,dtype=onp.complex128)
xmat = sevecs @ onp.diag(sevals**(-0.5))
npts = trueham.shape[0]
for i in range(1,npts):
    truehamMO[i,:,:] =  xmat.conj().T @ trueham[i,:,:] @ xmat

trueabserr = onp.zeros(truehamMO.shape[0])
truerelerr = onp.zeros(truehamMO.shape[0])
for j in range(truehamMO.shape[0]):
    rhs = truehamMO[j,:,:] @ denMO[j,:,:] - denMO[j,:,:] @ truehamMO[j,:,:] 
    lhs = -1j*(denMO[j+1,:,:] - denMO[j-1,:,:])/(2*dt)
    trueabserr[j] = onp.linalg.norm(rhs-lhs)
    truerelerr[j] = onp.linalg.norm(rhs-lhs)/onp.linalg.norm(lhs)

learnabserr = onp.zeros(myham.shape[0])
learnrelerr = onp.zeros(myham.shape[0])
for j in range(myham.shape[0]):
    rhs = myham[j,:,:] @ denMO[j+offset,:,:] - denMO[j+offset,:,:] @ myham[j,:,:]
    lhs = 1j*(denMO[j+1+offset,:,:] - denMO[j-1+offset,:,:])/(2*dt)
    learnabserr[j] = onp.linalg.norm(rhs-lhs)
    learnrelerr[j] = onp.linalg.norm(rhs-lhs)/onp.linalg.norm(lhs)

plt.plot(trueabserr[2:])
plt.xlabel('time')
plt.ylabel('true H ODE abs. residual')
plt.savefig(savepath + 'trueabserr.pdf')
plt.close()

plt.plot(truerelerr[2:])
plt.xlabel('time')
plt.ylabel('true H ODE rel. residual')
plt.savefig(savepath + 'truerelerr.pdf')
plt.close()

plt.plot(learnabserr[2:])
plt.xlabel('time')
plt.ylabel('ML H ODE abs. residual')
plt.savefig(savepath + 'learnabserr.pdf')
plt.close()

plt.plot(learnrelerr[2:])
plt.xlabel('time')
plt.ylabel('ML H ODE rel. residual')
plt.savefig(savepath + 'learnrelerr.pdf')
plt.close()

ncp = myham.shape[0]-offset
hamerr = onp.zeros((ncp,drc,drc), dtype=onp.complex128)
for j in range(ncp):
    hamerr[j,:,:] = myham[j,:,:] + truehamMO[j+offset,:,:]  

plt.plot(onp.linalg.norm(hamerr, axis=(1,2)))
plt.xlabel('time')
plt.ylabel('|| H_true - H_ML ||')
plt.savefig(savepath + 'hamiltonianerror.pdf')
plt.close()

auxdata = onp.load('./ke+en+overlap+ee_twoe+dip_hf_ndlaser1cyc_s0_'+mol+'_sto-3g.npz', allow_pickle=True)

print(list(auxdata.keys()))
print(onp.linalg.norm(auxdata['ke_data'] - kinmat))
print(onp.linalg.norm(auxdata['en_data'] - enmat))
print(onp.linalg.norm(auxdata['ee_twoe_data'] - eeten))

fielddata = onp.load('./td_efield+dipole_rt-tdexx_ndlaser1cycs0_'+mol+'_sto-3g.npz')

print(fielddata['td_efield_data'].shape)
print(fielddata['td_dipole_data'].shape)
efdat = fielddata['td_efield_data']

if mol == 'heh+':
    didat = [onp.zeros((2,2)), onp.zeros((2,2)), onp.array([[-0.729434, 0.0734846], [ 0.0734846, 0.729434 ]])]

elif mol == 'h2':
    didat = [onp.zeros((2,2)), onp.zeros((2,2)), onp.array([[-0.699199, 0.0], [0.0, 0.699199]])]

elif mol == 'lih':
    didat = [[]]*3
    didat[0] = onp.zeros((6,6))
    didat[0] = onp.zeros((6,6))
    didat[2] =  onp.array([[-0.144564e1, 0., 0., 0., 0., 0.],
                [0.698659e-01, 0.144564e+01, 0., 0., 0., 0.],
                [-0.341601, 0.348597, 0.144564e01, 0., 0., 0.],
                [0., 0., 0., 0.144564e+01, 0., 0.],
                [0., 0., 0., 0., 0.144564e+01, 0.],
                [0.667143, 0.147240, 0.180330e+01, 0., 0., 0.144564e+01]])
    didat[2] += onp.tril(didat[2], k=-1).T

elif mol == 'c2h4':
    didat = [[]]*3
    didat[0] = onp.zeros((drc,drc))
    didat[1] = onp.zeros((drc,drc))
    didat[2] = onp.array([[0.126517e1,0,0,0,0,0,0,0,0,0,0,0,0,0],
       [0.314221, 0.126517e1, 0,0,0,0,0,0,0,0,0,0,0,0],
       [0.,0.,0.126517e1 ,0,0,0,0,0,0,0,0,0,0,0],
       [0.,0.,0.,0.126517e1 ,0,0,0,0,0,0,0,0,0,0],
       [0.718555e-01,  0.838744,  0., 0., 0.126517e1, 0,0,0,0,0,0,0,0,0],
       [0., -0.490823e-01, 0., 0., 0.813421e-01, -0.126517e1, 0,0,0,0,0,0,0,0],
       [0.490823e-01, 0., 0., 0., 0.244692, -0.314221, -0.126517e1, 0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0, -0.126517e1, 0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,-0.126517e1, 0,0,0,0,0],
       [0.813421e-01,  0.244692,  0,0,0,  0.718555e-01,  0.838744, 0., 0., -0.126517e1, 0,0,0,0],
       [0.838104e-01,  0.898824,  0, 0.724971,  0.794395, -0.681175e-02, 0.463892e-01,0.,0.247053e-01, 0.125735, 0.232833e+01, 0.,0.,0.],
       [0.838104e-01,  0.898824,  0., -0.724971,  0.794395, -0.681175e-02,  0.463892e-01, 0., -0.247053e-01, 0.125735, 0.348982,  0.232833e1, 0.,0.],
       [0.681175e-02, -0.463892e-01, 0., -0.247053e-01,  0.125735, -0.838104e-01, -0.898824,  0., -0.724971,  0.794395, 0.,  0., -0.232833e1, 0.],
       [0.681175e-02, -0.463892e-01, 0., 0.247053e-01,  0.125735, -0.838104e-01, -0.898824,  0.,  0.724971,  0.794395, 0.,  0., -0.348982, -0.232833e1]])
    didat[2] += onp.tril(didat[2], k=-1).T

print(didat[2])

# EXACT deltakick Hamiltonian, NO FIELD
def EXhamrhs(t, pin):
    p = pin.reshape(drc,drc)
    
    pAO = xmat @ p @ xmat.conj().T
    twoe = get_ee_onee_AO(pAO, eeten)
    hAO = onp.array(kinmat - enmat, dtype=onp.complex128) + twoe
    h = -xmat.conj().T @ hAO @ xmat

    rhs = (h @ p - p @ h)/(1j)
    return rhs.reshape(drc**2)

# MACHINE LEARNED deltakick Hamiltonian, NO FIELD
def MLhamrhs(t, pin):
    p = pin.reshape(drc,drc)
    
    # MACHINE LEARNED deltakick Hamiltonian
    pflat = onp.zeros(d, dtype=np.complex128)
    for ij in nzreals:
        pflat[nzreals[ij]] = onp.real(p[ij[0], ij[1]])
    for ij in nzimags:
        pflat[nzimags[ij]] = onp.imag(p[ij[0], ij[1]])

    hflat = onp.matmul(rgm(pflat, t)[:,0], theta.reshape((nump,hamdof))) # + thtMOvec
    h = onp.zeros((drc,drc), dtype=np.complex128)
    for ij in hamreals:
        h[ij[0], ij[1]] = hflat[hamreals[ij]]
        h[ij[1], ij[0]] = hflat[hamreals[ij]]
    for ij in hamimags:
        h[ij[0], ij[1]] += (1J)*hflat[hamimags[ij]]
        h[ij[1], ij[0]] -= (1J)*hflat[hamimags[ij]]
    
    rhs = (h @ p - p @ h)/(1j)
    return rhs.reshape(drc**2)

# EXACT deltakick Hamiltonian, WITH FIELD ON
def EXhamwfrhs(t, pin):
    """
    i = int(t//dt)
    if i+offset > 1775:
        ez = 0
    elif i < 0:
        ez = 0
    else:
        frac = t/dt - i
        ez = (1-frac)*efdat[2,i+offset] + frac*efdat[2,i+1+offset]
    """
    freq = 0.0428
    if t > 2*onp.pi/freq:
        ez = 0
    elif t < 0:
        ez = 0
    else:
        ez = 0.05*onp.sin(0.0428*t)

    hfieldAO = onp.array(ez*didat[2], dtype=onp.complex128)

    p = pin.reshape(drc,drc)
    pAO = xmat @ p @ xmat.conj().T
    twoe = get_ee_onee_AO(pAO, eeten)
    
    if mol == 'heh+':
        hAO = (onp.array(kinmat - enmat, dtype=onp.complex128) + twoe) + hfieldAO
    elif mol == 'lih':
        hAO = (onp.array(kinmat - enmat, dtype=onp.complex128) + twoe) + hfieldAO
    elif mol == 'c2h4':
        hAO = (onp.array(kinmat - enmat, dtype=onp.complex128) + twoe) + hfieldAO
    elif mol == 'h2':
        hAO = (onp.array(kinmat - enmat, dtype=onp.complex128) + twoe) - hfieldAO

    h = -xmat.conj().T @ hAO @ xmat

    rhs = (h @ p - p @ h)/(1j)
    return rhs.reshape(drc**2)

# MACHINE LEARNED deltakick Hamiltonian, WITH FIELD ON
def MLhamwfrhs(t, pin):
    """
    i = int(t//dt)
    if i+offset > 1775:
        ez = 0
    elif i < 0:
        ez = 0
    else:
        frac = t/dt - i
        ez = (1-frac)*efdat[2,i+offset] + frac*efdat[2,i+1+offset]
    """

    freq = 0.0428
    if t > 2*onp.pi/freq:
        ez = 0
    elif t < 0:
        ez = 0
    else:
        ez = 0.05*onp.sin(0.0428*t)

    hfieldAO = onp.array(ez*didat[2], dtype=onp.complex128)
    hfieldAO = onp.array(ez*didat[2], dtype=onp.complex128)

    p = pin.reshape(drc,drc)

    # MACHINE LEARNED deltakick Hamiltonian
    pflat = onp.zeros(d, dtype=np.complex128)
    for ij in nzreals:
        pflat[nzreals[ij]] = onp.real(p[ij[0], ij[1]])
    for ij in nzimags:
        pflat[nzimags[ij]] = onp.imag(p[ij[0], ij[1]])

    hflat = onp.matmul(rgm(pflat, t)[:,0], theta.reshape((nump,hamdof))) # + thtMOvec
    h = onp.zeros((drc,drc), dtype=np.complex128)
    for ij in hamreals:
        h[ij[0], ij[1]] = hflat[hamreals[ij]]
        h[ij[1], ij[0]] = hflat[hamreals[ij]]
    for ij in hamimags:
        h[ij[0], ij[1]] += (1J)*hflat[hamimags[ij]]
        h[ij[1], ij[0]] -= (1J)*hflat[hamimags[ij]]
    
    if mol == 'heh+':
        h -= xmat.conj().T @ hfieldAO @ xmat
    elif mol == 'lih':
        h -= xmat.conj().T @ hfieldAO @ xmat
    elif mol == 'c2h4':
        h -= xmat.conj().T @ hfieldAO @ xmat
    elif mol == 'h2':
        h += xmat.conj().T @ hfieldAO @ xmat
    
    rhs = (h @ p - p @ h)/(1j)
    return rhs.reshape(drc**2)

# propagate forward in time using ML ham
intpts = 2000
tvec = dt*onp.arange(intpts-offset)
print(tvec[-1])
initcond = onp.array(denMOflat[offset,:])

EXsol = si.solve_ivp(EXhamrhs, [0, tvec[-1]], initcond, 'RK45', t_eval = tvec, rtol=mytol, atol=mytol)
MLsol = si.solve_ivp(MLhamrhs, [0, tvec[-1]], initcond, 'RK45', t_eval = tvec, rtol=mytol, atol=mytol)

# error between propagating machine learned Hamiltonian and Gaussian data
print(onp.mean(onp.linalg.norm( MLsol.y.T.reshape((-1,drc,drc)) - denMO[offset:intpts,:,:], axis = (1,2) )))

# error between propagating exact Hamiltonian and Gaussian data
print(onp.mean(onp.linalg.norm( EXsol.y.T.reshape((-1,drc,drc)) - denMO[offset:intpts,:,:] , axis = (1,2) )))

# error between propagating exact Hamiltonian and propagating machine learned Hamiltonian
print(onp.mean(onp.linalg.norm( EXsol.y.T.reshape((-1,drc,drc)) - MLsol.y.T.reshape((-1,drc,drc)), axis = (1,2) )))

# compute and save time-dependent propagation errors 
tdexHamerr = onp.linalg.norm( EXsol.y.T.reshape((-1,drc,drc)) - denMO[offset:intpts,:,:] , axis=(1,2))
tdmlHamerr = onp.linalg.norm( MLsol.y.T.reshape((-1,drc,drc)) - denMO[offset:intpts,:,:] , axis=(1,2))
tdexmlerr = onp.linalg.norm( EXsol.y.T.reshape((-1,drc,drc)) - MLsol.y.T.reshape((-1,drc,drc)) , axis=(1,2))

onp.savez(savepath+mol+'tdHamerr.npz',tdexHamerr=tdexHamerr,tdmlHamerr=tdmlHamerr,tdexmlerr=tdexmlerr)

"""
fig = plt.figure(figsize=((8,8)))
fig.suptitle('Gaussian (black), exact-H (blue), and ML-H (red) propagation results')
fig, ax1 = plt.subplots()
ij = realnzs[0]
ax1.plot(onp.real(EXsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'b-')
ax1.plot(onp.real(MLsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'r-')
ax1.plot(onp.real(denMO[offset:intpts,ij[0],ij[1]]), 'k-')
ax1.set_xlabel('time')
ax1.set_ylabel('P_'+str(ij))
ax1.tick_params(axis='y')

ij = realnzs[1]
ax2 = ax1.twinx()
ax2.plot(onp.real(EXsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'b-')
ax2.plot(onp.real(MLsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'r-')
ax2.plot(onp.real(denMO[offset:intpts,ij[0],ij[1]]), 'k-')
ax2.set_ylabel('P_'+str(ij))
ax2.tick_params(axis='y')

fig.tight_layout()
fig.savefig('./' + mol + 'prop.pdf')
plt.close()
"""

if mol == 'lih':
    fig = plt.figure(figsize=((8,16)))
else:
    fig = plt.figure(figsize=((8,12)))

axs = fig.subplots(d)
fig.suptitle('Gaussian (black), exact-H (blue), and ML-H (red) propagation results',y=0.9)
ctr = 0
mylabels = []
for ij in nzreals:
    axs[ctr].plot(tvec, onp.real(EXsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'b-')
    axs[ctr].plot(tvec, onp.real(MLsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'r-')
    axs[ctr].plot(tvec, onp.real(denMO[offset:intpts,ij[0],ij[1]]), 'k-')
    ijprime = (ij[0]+1, ij[1]+1)
    thislabel = 'Re(P_' + str(ijprime) + ')'
    mylabels.append(thislabel)
    ctr += 1

for ij in nzimags:
    axs[ctr].plot(tvec, onp.imag(EXsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'b-')
    axs[ctr].plot(tvec, onp.imag(MLsol.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'r-')
    axs[ctr].plot(tvec, onp.imag(denMO[offset:intpts,ij[0],ij[1]]), 'k-')
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

fig.savefig(savepath + mol + 'prop.pdf')
plt.close()

fielddens = onp.load('./td_dens_re+im_rt-tdexx_ndlaser1cycs0_'+mol+'_sto-3g.npz',allow_pickle=True)
fieldden = fielddens['td_dens_re_data'] + 1j*fielddens['td_dens_im_data']

# change basis
fielddenMO = onp.zeros(fieldden.shape, dtype=onp.complex128)
for i in range(fieldden.shape[0]):
    fielddenMO[i,:,:] = onp.diag(sevals**(0.5)) @ sevecs.T @ fieldden[i,:,:] @ sevecs @ onp.diag(sevals**(0.5))

# remove duplicates
fielddenMOflat = fielddenMO.reshape((-1,drc**2))
fielddenMOnodupflat = onp.array([onp.delete(fielddenMOflat[:,i], onp.s_[101::100]) for i in range(drc**2)]).T
fielddenMOnodup = fielddenMOnodupflat.reshape((-1,drc,drc))
print(fielddenMOnodup.shape)

# propagate forward in time using ML ham with field
intpts = 2000
tvec = dt*onp.arange(intpts-offset)
print(tvec[-1])
initcond = onp.array(fielddenMOnodupflat[offset,:])

EXsolwf = si.solve_ivp(EXhamwfrhs, onp.array([0, tvec[-1]]), initcond, t_eval = tvec, rtol=mytol, atol=mytol)
MLsolwf = si.solve_ivp(MLhamwfrhs, onp.array([0, tvec[-1]]), initcond, t_eval = tvec, rtol=mytol, atol=mytol)

# error between propagating machine learned Hamiltonian and Gaussian data
print(onp.mean(onp.linalg.norm( MLsolwf.y.T.reshape((-1,drc,drc)) - fielddenMOnodup[offset:intpts,:,:], axis = (1,2) )))

# error between propagating exact Hamiltonian and Gaussian data
print(onp.mean(onp.linalg.norm( EXsolwf.y.T.reshape((-1,drc,drc)) - fielddenMOnodup[offset:intpts,:,:] , axis = (1,2) )))

# error between propagating exact Hamiltonian and propagating machine learned Hamiltonian
print(onp.mean(onp.linalg.norm( EXsolwf.y.T.reshape((-1,drc,drc)) - MLsolwf.y.T.reshape((-1,drc,drc)), axis = (1,2) )))

# compute and save time-dependent propagation errors 
tdexHamerr = onp.linalg.norm( EXsolwf.y.T.reshape((-1,drc,drc)) - fielddenMOnodup[offset:intpts,:,:] , axis=(1,2))
tdmlHamerr = onp.linalg.norm( MLsolwf.y.T.reshape((-1,drc,drc)) - fielddenMOnodup[offset:intpts,:,:] , axis=(1,2))
tdexmlerr = onp.linalg.norm( EXsolwf.y.T.reshape((-1,drc,drc)) - MLsolwf.y.T.reshape((-1,drc,drc)) , axis=(1,2))
onp.savez(savepath+mol+'tdHamerrWF.npz',tdexHamerr=tdexHamerr,tdmlHamerr=tdmlHamerr,tdexmlerr=tdexmlerr)

mylabels = []
if mol == 'lih':
    fig = plt.figure(figsize=((8,16)))
    fig.suptitle('Gaussian (black), exact-H (blue), and ML-H (red) propagation results',y=0.9)
    axs = fig.subplots(d)
    ctr = 0
else:
    fig = plt.figure(figsize=((8,12)))
    fig.suptitle('Gaussian (black), exact-H (blue), and ML-H (red) propagation results',y=0.9)
    axs = fig.subplots(d+1)
    trueefield = 0.05*onp.sin(0.0428*tvec)
    trueefield[1776:] = 0.
    axs[0].plot(tvec, trueefield, 'k-')
    thislabel = 'E-field'
    mylabels.append(thislabel)
    ctr = 1

for ij in nzreals:
    axs[ctr].plot(tvec, onp.real(EXsolwf.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'b-')
    axs[ctr].plot(tvec, onp.real(MLsolwf.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'r-')
    axs[ctr].plot(tvec, onp.real(fielddenMOnodup[offset:intpts,ij[0],ij[1]]), 'k-')
    ijprime = (ij[0]+1, ij[1]+1)
    thislabel = 'Re(P_' + str(ijprime) + ')'
    mylabels.append(thislabel)
    ctr += 1

for ij in nzimags:
    axs[ctr].plot(tvec, onp.imag(EXsolwf.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'b-')
    axs[ctr].plot(tvec, onp.imag(MLsolwf.y.T.reshape((-1,drc,drc))[:,ij[0],ij[1]]), 'r-')
    axs[ctr].plot(tvec, onp.imag(fielddenMOnodup[offset:intpts,ij[0],ij[1]]), 'k-')
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

fig.savefig(savepath + mol + 'propWF.pdf')
plt.close()





