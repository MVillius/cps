# Objectif du programme
# Déterminer l'évolution temporelle à partir de l'état vide
# à l'aide de simulations temporelles sur QuTip

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

def Hamiltonian(e_sum, e_delta, e_mag, e_asym, g=0.4):
        omega0L = 2*e_mag*(1+e_asym)
        omega0R = 2*e_mag*(1-e_asym)
        e_LUp = (e_sum+e_delta)/2 - e_mag*(1+e_asym)
        e_LDo = (e_sum+e_delta)/2 + e_mag*(1+e_asym)
        e_RUp = (e_sum-e_delta)/2 - e_mag*(1-e_asym)
        e_RDo = (e_sum-e_delta)/2 + e_mag*(1-e_asym)    
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


# Parameters 
U = 250
Um = 0
teh = 1
theta = np.pi/4
e_sum = 0
e_delta = 0
tee = teh*e_sum/40
asym = 0.1
e_mag = 5
b_l = e_mag*(1+asym)
b_r = e_mag*(1-asym)
omega0L = 2*b_l
omega0R = 2*b_r
#DeltaKKp = 500 

delta = b_l-b_r 

singlet_op = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())
triplet_0_op = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())    
singlet = singlet_op*vac
singlet = singlet/singlet.norm()    
triplet_0 = (LUpKp.dag()*RDoKp.dag()+LDoKp.dag()*RUpKp.dag())*vac
triplet_0 = triplet_0/triplet_0.norm()

times = np.linspace(0,5,1000)
psi0 = vac
omega = np.sqrt(delta**2+2*teh**2*np.cos(theta/2)**2)
theory = np.array(([[(delta**2+2*teh**2*np.cos(theta/2)**2*np.cos(omega*t))/omega**2, -1j*np.sqrt(2)*teh*np.cos(theta/2)*np.sin(omega*t)/omega, delta*np.sqrt(2)*teh*np.cos(theta/2)*(np.cos(omega*t)-1)/omega**2] 
    for t in times]))
#Données par la résolution analytique de l'équation de Schrodinger

H = Hamiltonian(e_sum, e_delta, (b_l+b_r)/2, asym)
result = qt.mesolve(H, psi0, times, [], [])


singlet_op = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())
triplet_0_op = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())
singlet = singlet_op*vac
singlet = singlet/singlet.norm()    
triplet_0 = (LUpKp.dag()*RDoKp.dag()+LDoKp.dag()*RUpKp.dag())*vac
triplet_0 = triplet_0/triplet_0.norm()
result = qt.mesolve(H, psi0, times, [], [])
r  = [[],[],[],[]]
for i in range(len(result.states)):
    st = result.states[i]
    r[0].append(np.abs(singlet.overlap(st))**2)
    r[1].append(np.abs(triplet_0.overlap(st))**2)
    r[2].append(np.abs(vac.overlap(st))**2)

plt.plot(times, r[0], label="Singlet")
plt.plot(times, r[1], label="Triplet")
plt.plot(times, r[2], label="Vac")
plt.plot(times, np.abs(theory[:,1])**2, "--",label="Singlet (théorique)")
plt.plot(times, np.abs(theory[:,2])**2, "--",label="Triplet (théorique)")
plt.plot(times, np.abs(theory[:,0])**2, "--",label="Vac (théorique)")

plt.xlabel("Time (ns)")
plt.legend()
plt.title("Time evolution starting from vac")
plt.show()
