"""
Objectif du programme : 
Balayage en fréquence pour différentes valeurs de ed/U afin de déterminer le max de (a^dag*a)
à l'aide de simulations temporelles QuTip
"""

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

#Définition d'état
s_r = (RUpKp.dag()*RDoKp.dag())*vac
s_r /= s_r.norm()
s_l = (LUpKp.dag()*LDoKp.dag())*vac
s_l /= s_l.norm()
singlet = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())*vac
singlet /= singlet.norm()
triplet = (LUpKp.dag()*RDoKp.dag()+LDoKp.dag()*RUpKp.dag())*vac
triplet /= triplet.norm()
triplet_p = LUpKp.dag()*RUpKp.dag()*vac
triplet_m = LDoKp.dag()*RDoKp.dag()*vac

#Etats d'intérêt
st_1 = singlet+triplet+s_r
st_2 = singlet+s_r-triplet
st_3 = singlet-s_r+triplet

etats = [LUpKp.dag()*RDoKp.dag()*vac, LDoKp.dag()*RUpKp.dag()*vac, triplet_m, triplet_p] 

etats_txt = [r'$\uparrow_{K^p}$'+","r'$\downarrow_{K^p}$', r'$\downarrow_{K^p}$'+","r'$\uparrow_{K^p}$',r'$\uparrow_{K^p}$'+","r'$\uparrow_{K^p}$',
r'$\downarrow_{K^p}$'+","r'$\downarrow_{K^p}$',r'$\uparrow_{K^p}\downarrow_{K^p}$'+',o','o,'+r'$\uparrow_{K^p}\downarrow_{K^p}$']

class Params:
    def __init__(self, e_s, e_d, b_l, b_r,U, Um,theta, g,tee, teh):
        self.e_sum = e_s
        self.e_delta = e_d
        self.b_l = b_l
        self.b_r = b_r
        self.U = U
        self.Um = Um
        self.theta = theta
        self.g = g
        self.tee = tee
        self.teh = teh
def Hamiltonian(p,e_sum, e_delta, b_l, b_r, theta, shift_l,shift_r,g=1):
        omega0L = 2*b_l - shift_l
        omega0R = 2*b_r - shift_r
        e_LUp = (e_sum+p.e_delta)/2 - b_l
        e_LDo = (e_sum+p.e_delta)/2 + b_l
        e_RUp = (e_sum-p.e_delta)/2 - b_r
        e_RDo = (e_sum-p.e_delta)/2 + b_r
        print(omega0L)
        print(omega0R)
        Hchem = e_LUp*(LUpKp.dag()*LUpKp) +\
                e_LDo*(LDoKp.dag()*LDoKp) + \
                e_RUp*(RUpKp.dag()*RUpKp) + \
                e_RDo*(RDoKp.dag()*RDoKp)

        Hint = (p.U/2)*(nL*(nL-1)+nR*(nR-1)) + p.Um*nL*nR

#        HKKp = DeltaKKp * (LUpK.dag()*LUpK+LDoK.dag()*LDoK+\
#                           RUpK.dag()*RUpK+RDoK.dag()*RDoK-\
#                           LUpKp.dag()*LUpKp-LDoKp.dag()*LDoKp-\
#                           RUpKp.dag()*RUpKp-RDoKp.dag()*RDoKp)

        Hteh = p.teh*np.cos(p.theta/2)*(LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag()) +\
               p.teh*np.sin(p.theta/2)*(LUpKp.dag()*RUpKp.dag()+LDoKp.dag()*RDoKp.dag())
        Hteh += Hteh.dag()
        
        Htee = p.tee*np.cos(p.theta/2)*(LUpKp.dag()*RUpKp + LDoKp.dag()*RDoKp) +\
               p.tee*np.sin(p.theta/2)*(-LUpKp.dag()*RDoKp + LDoKp.dag()*RUpKp)
        Htee += Htee.dag()
        Hcavite = p.g*nL*(aLeft+aLeft.dag()) + p.g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)
        H = Hchem + Hint + Htee + Hteh + Hcavite + Hphoton
        H_p = Hchem + Hint + Hphoton
        H_i = Htee + Hteh + Hcavite
        #print(Htee)
        H_nocavity = Hchem+Hint+Htee+Hteh
        [n,s] = H_nocavity.eigenstates()
        return H,H_i,H_p


psi0 = singlet
observ = [nL, nR, aLeft.dag()*aLeft, aRight.dag()*aRight] 
arrondi = 10
U = 25
tees = np.linspace(2,2,1)
FLEX_TMAX = False #Allocation dynamique du nombre de points
MAX_T = 25000 #Nombre de points maximum utilisé pour la simulation temporelle QuTip
MAX_POINTS = 15  #Nombre de points pour la recherche dichotomique
INCREASE = 1.5 #facteur de multiplication du nombre de points

for tee in tees:
    X = Params(40, 0, 6.5,5.5, U, 0, np.pi/2, 1, tee,1) #e_s, e_d, b_l, b_r,U, Um,theta, g,te, teh)
    history =  {}
    #Initialisation des valeurs min/max
    hint = tee**2/U

    shift_freq_high = hint + 0.1
    shift_freq_low = hint - 0.1

    if FLEX_TMAX:
        Tmax = 200
    else:
        Tmax = 25000 
    #Recherche du maximum par recherche dichotomique
    for i in range(0,MAX_POINTS):
        print("frequence low : {}".format(shift_freq_low))
        print("frequence high : {}".format(shift_freq_high))

        if np.round(shift_freq_low,arrondi) not in history:
            H_low,_,_ = Hamiltonian(X, X.e_sum, X.e_delta, X.b_l, X.b_r,X.theta,shift_freq_low,0, 1)
            times = np.linspace(0,Tmax,Tmax*10)
            result_low = qt.mesolve(H_low, psi0, times, [], observ)
            iMax = np.argmax(result_low.expect[2]) 
            while FLEX_TMAX and np.abs(times[iMax]-Tmax) < 20 and Tmax<MAX_T: #Cette boucle vérifie qu'il s'agit bien d'un "vrai" maximum. Si ce n'est pas le cas on augmente Tmax jusqu'à atteindre un max
                Tmax = INCREASE*Tmax
                print("Increasing Tmax up to {}".format(Tmax))  
                times = np.linspace(0,Tmax,Tmax*20) 
                result_low = qt.mesolve(H_low, psi0, times, [], observ)
                iMax = np.argmax(result_low.expect[2])
                print(times[iMax])
                print(Tmax)
            maxi_low = np.max(result_low.expect[2])
            history[np.round(shift_freq_low,arrondi)] = maxi_low
        else: #Si la valeur a déjà été calculée précédemment 
            maxi_low = history[np.round(shift_freq_low, arrondi)]

        if np.round(shift_freq_high,arrondi) not in history:
            H_high,_,_ = Hamiltonian(X, X.e_sum, X.e_delta, X.b_l, X.b_r,X.theta,shift_freq_high,0, 1) 
            times = np.linspace(0,Tmax,Tmax*10)
            result_high = qt.mesolve(H_high, psi0, times, [], observ)
            iMax = np.argmax(result_high.expect[2])
            while FLEX_TMAX and np.abs(times[iMax]-Tmax) < 20 and Tmax<2000:
                Tmax = INCREASE*Tmax 
                print("Increasing Tmax up to {}".format(Tmax))+
                times = np.linspace(0,Tmax,Tmax*20)
                result_high = qt.mesolve(H_high, psi0, times, [], observ)
                iMax = np.argmax(result_high.expect[2])

            maxi_high =  np.max(result_high.expect[2])
            history[np.round(shift_freq_high,arrondi)] = maxi_high
        else: #Si la valeur a déjà été calculée
            maxi_high = history[np.round(shift_freq_high, arrondi)]

        if maxi_low < maxi_high:
            shift_freq_low = (shift_freq_high+shift_freq_low)/2     
        else:
            shift_freq_high = (shift_freq_high+shift_freq_low)/2

    print(np.argmax(result_low.expect[2]))
    x,y = [],[]
    x_b,y_b = [],[]
    maxi = 0
    for key in sorted(history):
        x.append(key)
        y.append(history[key])
        if history[key]>maxi:
            maxi = history[key]
    for key in sorted(history):
        if np.abs(history[key]-maxi)<0.1*maxi:# On ne conserve que les points près du maximum 
            x_b.append(key)
            y_b.append(history[key])
    plt.plot(x_b,y_b, label=str(np.round(tee, 4)))
plt.legend()
plt.show()
