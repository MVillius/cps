import numpy as np
import qutip as qt
import numpy as np
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt

sm = qt.sigmam()
sz = qt.sigmaz()
I = qt.identity(2)
orbital_number = 1
ops = {}

aLeft, aRight = I,I
III = qt.qeye(3)

for i_ops in range(4*orbital_number): # 8=2(spin)*2(K,Kp)*2(left/right)
    ops[i_ops] = sm  #initialize
for i_ops in range(4*orbital_number):
    kid = 4*orbital_number-1-i_ops
    ksz = i_ops
    while kid > 0:
        ops[i_ops] = qt.tensor(ops[i_ops], I) # pad with I to the right
        kid -= 1
    while ksz > 0:
        ops[i_ops] = qt.tensor(-sz, ops[i_ops]) # pad with -sz to the left
        ksz -= 1
    ops[i_ops] = qt.tensor(ops[i_ops],III)  #pad 2 I for photons
    ops[i_ops] = qt.tensor(ops[i_ops],III)
for k in range(4*orbital_number-1):
    aLeft = qt.tensor(aLeft,I)
    aRight = qt.tensor(aRight,I)
## 1st orbital
#LUpK = ops[0]
#LDoK = ops[1]
#RUpK = ops[2]
#RDoK = ops[3]

# 2nd orbital
LUpKp = ops[0]
LDoKp = ops[1]
RUpKp = ops[2]
RDoKp = ops[3]


aLeft = qt.tensor(aLeft, qt.destroy(3))
aRight = qt.tensor(aRight, III)
aLeft = qt.tensor(aLeft, III)
aRight = qt.tensor(aRight, qt.destroy(3))
II = I
zero = qt.tensor(qt.fock(2,1)) 
zero_photons = qt.tensor(qt.fock(3,0))
vac = zero
for i_ops in range(4*orbital_number+1):
    if i_ops<4*orbital_number-1:
        vac = qt.tensor(vac, zero)
        II = qt.tensor(II,I)
    else:
        vac = qt.tensor(vac, zero_photons)
        II = qt.tensor(II,III)

nL = LUpKp.dag()*LUpKp + LDoKp.dag()*LDoKp
nR = RUpKp.dag()*RUpKp + RDoKp.dag()*RDoKp
nUp = LUpKp.dag()*LUpKp + RUpKp.dag()*RUpKp
nDo = LDoKp.dag()*LDoKp + RDoKp.dag()*RDoKp
nPhoton = aLeft.dag()*aLeft+aRight.dag()*aRight
def comm(A,B):
    return A*B-B*A

def Hamiltonian(e_sum, e_delta, bs, bd, g=0.4):
        e_LUp = (e_sum+e_delta)/2 - (bs+bd)/2
        e_LDo = (e_sum+e_delta)/2 + (bs+bd)/2
        e_RUp = (e_sum-e_delta)/2 - (bs-bd)/2
        e_RDo = (e_sum-e_delta)/2 + (bs-bd)/2
        omega0L = (bs+bd)
        omega0R = (bs-bd)
        print(omega0L)
        print(omega0R)
        Hchem = e_LUp*(LUpKp.dag()*LUpKp) +\
                e_LDo*(LDoKp.dag()*LDoKp) + \
                e_RUp*(RUpKp.dag()*RUpKp) + \
                e_RDo*(RDoKp.dag()*RDoKp)

        Hint = (U/2)*(nL*(nL-1)+nR*(nR-1)) + Um*nL*nR

#        HKKp = DeltaKKp * (LUpK.dag()*LUpK+LDoK.dag()*LDoK+\
#                           RUpK.dag()*RUpK+RDoK.dag()*RDoK-\
#                           LUpKp.dag()*LUpKp-LDoKp.dag()*LDoKp-\
#                           RUpKp.dag()*RUpKp-RDoKp.dag()*RDoKp)

        Hteh = teh*np.cos(theta/2)*(LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag()) +\
               teh*np.sin(theta/2)*(LUpKp.dag()*RUpKp.dag()+LDoKp.dag()*RDoKp.dag())
        Hteh += Hteh.dag()
        
        Htee = tee*np.cos(theta/2)*(LUpKp.dag()*RUpKp + LDoKp.dag()*RDoKp) +\
               tee*np.sin(theta/2)*(-LUpKp.dag()*RDoKp + LDoKp.dag()*RUpKp)
        Htee += Htee.dag()
        H_nocavity = Hchem+Hint+Htee+Hteh
        Hcavite = g*nL*(aLeft+aLeft.dag()) + g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)
        return H_nocavity+Hcavite+Hphoton


U = 250
Um = 0
teh = 1
theta = np.pi/4
e_sum = 0
e_delta = 0
tee = teh*e_sum/40
bd = 1
bs = 10
omega0L = 2*(bs+bd)
omega0R = 2*(bs-bd)
#DeltaKKp = 500 
delta = bd

singlet_op = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())
triplet_0_op = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())    
singlet = singlet_op*vac
singlet = singlet/singlet.norm()    
triplet_0 = (LUpKp.dag()*RDoKp.dag()+LDoKp.dag()*RUpKp.dag())*vac
triplet_0 = triplet_0/triplet_0.norm()

delta_max, delta_tmax = [],[]

teh_s = np.linspace(0.1,5, 50)
sweep_teh = True
#If not sweeping and plotting for only one value of tee turn off sweep_teh
data = []
for teh in teh_s:
    omega = np.sqrt(delta**2+2*teh**2*np.cos(theta/2)**2)

    theoretical_tmax = np.pi/(2*omega)
    times = np.linspace(0,theoretical_tmax+0.4,600)
    psi0 = vac
    theory = np.array(([[(delta**2+2*teh**2*np.cos(theta/2)**2*np.cos(omega*t))/omega**2, -1j*np.sqrt(2)*teh*np.cos(theta/2)*np.sin(omega*t)/omega, delta*np.sqrt(2)*teh*np.cos(theta/2)*(np.cos(omega*t)-1)/omega**2] 
        for t in times]))
    theoretical_max = (2*teh**2*np.cos(theta/2)**2)/(2*teh**2*np.cos(theta/2)**2+delta**2)

    H = Hamiltonian(e_sum, e_delta, bs, bd)
    result = qt.mesolve(H, psi0, times, [], [])
    num_max, num_tmax = 0,0
    for ii,st in enumerate(reversed(result.states)):
        if np.abs(singlet.overlap(st))**2>num_max:
            num_max = np.abs(singlet.overlap(st))**2
            num_tmax = times[len(times)-ii-1]
    data.append((theoretical_max, 1/theoretical_tmax, num_max, 1/num_tmax))



if not sweep_teh:
    r  = [[],[],[],[]]
    times2 = []
    for i in range(len(result.states)):
        if i%1==0:
            times2.append(times[i])
            st = result.states[i]
            r[0].append(np.abs(singlet.overlap(st))**2)
            r[1].append(np.abs(triplet_0.overlap(st))**2)
            r[2].append(np.abs(vac.overlap(st))**2)

    plt.scatter(times2, r[0], marker="+",label="Singlet")
    plt.scatter(times2, r[1], marker="+",label="Triplet")
    plt.scatter(times2, r[2], marker="+",label="Vac")
    plt.plot(times, np.abs(theory[:,1])**2,label="Singlet (theory)")
    plt.plot(times, np.abs(theory[:,2])**2,label="Triplet (theory)")
    plt.plot(times, np.abs(theory[:,0])**2,label="Vac (theory)")
    plt.xlabel("Time (ns)")
    plt.legend()
    plt.title("Time evolution starting from vac")
    plt.show()
else:
    data = np.array(data)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.square(teh_s), data[:,0],label="theory")
    ax[0].scatter(np.square(teh_s), data[:,2],marker="+", label="numeric")

    ax[1].plot(np.square(teh_s), data[:,1],label="theory")
    ax[1].scatter(np.square(teh_s), data[:,3],marker="+", label="numeric")

    ax[0].set_xlabel(r"$t_{ee}^2$")
    ax[1].set_xlabel(r"$t_{ee}^2$")
    ax[0].set_title(r"Maximum of $P\left(|S>\right)$")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_title("Frequency")
    ax[0].fill_between(np.square(teh_s), 0, 1, where=(data[:,0]-data[:,2]) <=0.01, facecolor='green', alpha=0.2)
    ax[0].fill_between(np.square(teh_s), 0, 1, where=(data[:,0]-data[:,2]) >=0.01, facecolor='red', alpha=0.2)

    plt.show()

