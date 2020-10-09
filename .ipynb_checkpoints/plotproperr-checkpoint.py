import numpy as onp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mols = 'heh+','h2','lih'
tdexHamerr = {}
tdmlHamerr = {}
tdexmlerr = {}
for mol in mols:
    fname = './results/'+mol+'tdHamerr.npz'
    fil = onp.load(fname,allow_pickle=True)
    tdexHamerr[mol] = fil['tdexHamerr']
    tdmlHamerr[mol] = fil['tdmlHamerr']
    tdexmlerr[mol] = fil['tdexmlerr']

tdexHamerrWF = {}
tdmlHamerrWF = {}
tdexmlerrWF = {}
for mol in mols:
    fname = './results/'+mol+'tdHamerrWF.npz'
    fil = onp.load(fname,allow_pickle=True)
    tdexHamerrWF[mol] = fil['tdexHamerr']
    tdmlHamerrWF[mol] = fil['tdmlHamerr']
    tdexmlerrWF[mol] = fil['tdexmlerr']

tvec = 0.08268*onp.arange(1998)

fig = plt.figure(figsize=((8,8)))
fig, ax = plt.subplots()
myms = 0.75
plt.plot(tvec,tdexHamerr['heh+'],'k.',ms=myms,label='HeH+ exact Ham')
plt.plot(tvec,tdmlHamerr['heh+'],'g.',ms=myms,label='HeH+ ML Ham')
plt.plot(tvec,tdexHamerr['h2'],'rs',ms=myms,label='H2 exact Ham')
plt.plot(tvec,tdmlHamerr['h2'],'ms',ms=myms,label='H2 ML Ham')
plt.plot(tvec,tdexHamerr['lih'],'bd',ms=myms,label='LiH exact Ham')
plt.plot(tvec,tdmlHamerr['lih'],'cd',ms=myms,label='LiH ML Ham')
ax.legend(loc='upper left',markerscale=4.0)
ax.set_xlabel('time')
ax.set_ylabel(r'Propagation errors $\mathcal{P}$ and $\mathcal{P}_{\mathrm{Sch}}$')
fig.tight_layout()
fig.savefig('./hamerr.pdf')
plt.close()

fig = plt.figure(figsize=((8,8)))
fig, ax = plt.subplots()
myms = 0.75
plt.plot(tvec,tdexHamerrWF['heh+'],'k.',ms=myms,label='HeH+ exact Ham with field')
plt.plot(tvec,tdmlHamerrWF['heh+'],'g.',ms=myms,label='HeH+ ML Ham with field')
plt.plot(tvec,tdexHamerrWF['h2'],'rs',ms=myms,label='H2 exact Ham with field')
plt.plot(tvec,tdmlHamerrWF['h2'],'ms',ms=myms,label='H2 ML Ham with field')
plt.plot(tvec,tdexHamerrWF['lih'],'bd',ms=myms,label='LiH exact Ham with field')
plt.plot(tvec,tdmlHamerrWF['lih'],'cd',ms=myms,label='LiH ML Ham with field')
ax.legend(loc='upper left',markerscale=4.0)
ax.set_xlabel('time')
ax.set_ylabel(r'Propagation errors $\mathcal{P}$ and $\mathcal{P}_{\mathrm{Sch}}$')
fig.tight_layout()
fig.savefig('./hamerrWF.pdf')
plt.close()

fig = plt.figure(figsize=((8,8)))
fig, ax = plt.subplots()
myms = 0.75
plt.plot(tvec,tdexmlerr['heh+'],'k.',ms=myms,label='HeH+')
plt.plot(tvec,tdexmlerrWF['heh+'],'g.',ms=myms,label='HeH+ with field')
plt.plot(tvec,tdexmlerr['h2'],'rs',ms=myms,label='H2')
plt.plot(tvec,tdexmlerrWF['h2'],'ms',ms=myms,label='H2 with field')
plt.plot(tvec,tdexmlerr['lih'],'bd',ms=myms,label='LiH')
plt.plot(tvec,tdexmlerrWF['lih'],'cd',ms=myms,label='LiH with field')
plt.yscale('log')
ax.legend(loc='lower right',markerscale=4.0)
ax.set_xlabel('time')
ax.set_ylabel(r'Propagation error $\mathcal{P}_{\mathrm{Ham}}$')
fig.tight_layout()
fig.savefig('./exmlproperr.pdf')
plt.close()

