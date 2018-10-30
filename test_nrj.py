import numpy as np
import qutip as qt
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
sm = qt.sigmam()
sz = qt.sigmaz()
I = qt.identity(2)
orbital_number = 1
ops = {}

aLeft, aRight = I,I
III = qt.qeye(3)

for i_ops in range(8*orbital_number): # 8=2(spin)*2(K,Kp)*2(left/right)
    ops[i_ops] = sm  #initialize
for i_ops in range(8*orbital_number):
    kid = 8*orbital_number-1-i_ops
    ksz = i_ops
    while kid > 0:
        ops[i_ops] = qt.tensor(ops[i_ops], I) # pad with I to the right
        kid -= 1
    while ksz > 0:
        ops[i_ops] = qt.tensor(-sz, ops[i_ops]) # pad with -sz to the left
        ksz -= 1
    ops[i_ops] = qt.tensor(ops[i_ops],III)  #pad 2 I for photons
    ops[i_ops] = qt.tensor(ops[i_ops],III)
for k in range(8*orbital_number-1):
    aLeft = qt.tensor(aLeft,I)
    aRight = qt.tensor(aRight,I)
# 1st orbital
LUpK = ops[0]
LDoK = ops[1]
RUpK = ops[2]
RDoK = ops[3]

# 2nd orbital
LUpKp = ops[4]
LDoKp = ops[5]
RUpKp = ops[6]
RDoKp = ops[7]


aLeft = qt.tensor(aLeft, qt.destroy(3))
aRight = qt.tensor(aRight, III)
aLeft = qt.tensor(aLeft, III)
aRight = qt.tensor(aRight, qt.destroy(3))

II = I
zero = qt.tensor(qt.fock(2,1)) 
zero_photons = qt.tensor(qt.fock(3,0))
vac = zero
for i_ops in range(8*orbital_number+1):
    if i_ops<8*orbital_number-1:
        vac = qt.tensor(vac, zero)
        II = qt.tensor(II,I)
    else:
        vac = qt.tensor(vac, zero_photons)
        II = qt.tensor(II,III)

nL = LUpK.dag()*LUpK + LDoK.dag()*LDoK + LUpKp.dag()*LUpKp + LDoKp.dag()*LDoKp
nR = RUpK.dag()*RUpK + RDoK.dag()*RDoK + RUpKp.dag()*RUpKp + RDoKp.dag()*RDoKp

nUp = LUpK.dag()*LUpK + LUpKp.dag()*LUpKp+RUpK.dag()*RUpK + RUpKp.dag()*RUpKp
nDo = LDoK.dag()*LDoK +LDoKp.dag()*LDoKp+ RDoK.dag()*RDoK + RDoKp.dag()*RDoKp
nPhoton = aLeft.dag()*aLeft+aRight.dag()*aRight

#Définition d'états
s_r = (-RUpKp.dag()*RDoKp.dag()+LUpKp.dag()*LDoKp.dag())*vac
s_r /= s_r.norm()
s_l = (RUpKp.dag()*RDoKp.dag()+LUpKp.dag()*LDoKp.dag())*vac
s_l /= s_l.norm()
singlet = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())*vac
singlet /= singlet.norm()
triplet = (LUpKp.dag()*RDoKp.dag()+LDoKp.dag()*RUpKp.dag())*vac
triplet /= triplet.norm()
singlet_1 = triplet + RUpKp.dag()*RDoKp.dag()*vac
singlet_1 /= singlet_1.norm()
singlet_2 =    triplet - RUpKp.dag()*RDoKp.dag()*vac
singlet_2 /= singlet_2.norm()
singlet_3 = (singlet+s_r)
singlet_3 /= singlet_3.norm()
singlet_4 = (singlet-s_r)
singlet_4 /= singlet_4.norm()
triplet_p = LUpKp.dag()*RUpKp.dag()*vac
triplet_m = LDoKp.dag()*RDoKp.dag()*vac

#Etats d'intérêt
etats = [LUpKp.dag()*RDoKp.dag()*vac, LDoKp.dag()*RUpKp.dag()*vac, triplet_m, triplet_p]

etats_txt = [r'$\uparrow_{K^p}$'+","r'$\downarrow_{K^p}$', r'$\downarrow_{K^p}$'+","r'$\uparrow_{K^p}$',r'$\uparrow_{K^p}$'+","r'$\uparrow_{K^p}$',
r'$\downarrow_{K^p}$'+","r'$\downarrow_{K^p}$',r'$\uparrow_{K^p}\downarrow_{K^p}$'+',o','o,'+r'$\uparrow_{K^p}\downarrow_{K^p}$']



def Hamiltonian(e_sum, e_delta, e_mag, e_asym):
        omega0L = 2*e_mag*(1+e_asym)
        omega0R = 2*e_mag*(1-e_asym)
        e_LUp = (e_sum+e_delta)/2 - e_mag*(1+e_asym)
        e_LDo = (e_sum+e_delta)/2 + e_mag*(1+e_asym)
        e_RUp = (e_sum-e_delta)/2 - e_mag*(1-e_asym)
        e_RDo = (e_sum-e_delta)/2 + e_mag*(1-e_asym)    
        print(omega0L)
        Hchem = e_LUp*(LUpK.dag()*LUpK + LUpKp.dag()*LUpKp) +\
                e_LDo*(LDoK.dag()*LDoK + LDoKp.dag()*LDoKp) + \
                e_RUp*(RUpK.dag()*RUpK + RUpKp.dag()*RUpKp) + \
                e_RDo*(RDoK.dag()*RDoK + RDoKp.dag()*RDoKp)

        Hint = (U/2)*(nL*(nL-1)+nR*(nR-1)) + Um*nL*nR

        HKKp = DeltaKKp * (LUpK.dag()*LUpK+LDoK.dag()*LDoK+\
                           RUpK.dag()*RUpK+RDoK.dag()*RDoK-\
                           LUpKp.dag()*LUpKp-LDoKp.dag()*LDoKp-\
                           RUpKp.dag()*RUpKp-RDoKp.dag()*RDoKp)

        Hteh = teh*np.cos(theta/2)*(LUpKp.dag()*RDoKp.dag() - LDoKp.dag()*RUpK.dag())+\
               teh*np.sin(theta/2)*(LUpKp.dag()*RUpK.dag() + LDoKp.dag()*RDoKp.dag())
        Hteh += Hteh.dag()
        
        Htee = tee*np.cos(theta/2)*(LUpKp.dag()*RUpKp + LDoKp.dag()*RDoKp)+\
               tee*np.sin(theta/2)*(-LUpKp.dag()*RDoKp + LDoKp.dag()*RUpKp)
        Htee += Htee.dag()
        g = 0
        Hcavite = g*nL*(aLeft+aLeft.dag()) + g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)


        H = Hchem + Hint + HKKp+Htee+Hteh+Hcavite+Hphoton
        H_p = Hchem + Hint +HKKp+Hphoton
        H_i = Htee + Hteh + Hcavite
        return H,H_i,H_p
def proj(H0, p_states, np_states):
    """
    S = real_states[0]-real_states[1]
    S /= S.norm()
    T0 = real_states[0]+real_states[1]
    T0 /= T0.norm()
    Tp = real_states[2]
    Tp /= Tp.norm()
    Tm = real_states[3]
    V0 = real_states[4]
    V1 = real_states[5]

    basis = [S, T0, V0, V1,Tp, Tm]
    """
    """
    a = p_states[0]
    b = p_states[1]
    p_states[0] = (a-b)/np.sqrt(2)
    p_states[1] = (a+b)/np.sqrt(2)
    a = np_states[0]
    b = np_states[1]
    np_states[0] = (a-b)/np.sqrt(2)
    np_states[1] = (a+b)/np.sqrt(2)
    """ 
    N = len(np_states)
    mat = np.zeros((N,N))
    for i,b1 in enumerate(np_states):
        for j, b2 in enumerate(np_states):
            mat[i,j] =np.abs(b1.overlap(H0*b2))
    """
    passage = np.zeros((6,6))
    for i,b1 in enumerate(p_states):
        for j, b2 in enumerate(np_states):
            passage[i,j] = (b2.dag()*b1)[0][0][0]

    tot = 0 
    print(np.round(passage,5))
    print()
    mat2 = np.zeros((N,N))
    for i in range(len(np_states)):
        for j in range(len(np_states)):
            for k,s in enumerate(p_states):
                for l,s2 in enumerate(p_states):
                    mat2[i,j] += passage[k,i]*passage[l,j]
            mat2[i,j] = np.round(mat2[i,j],6)
    print(mat2)

    """
    mat4 = np.zeros((N,N))
    mat5 = np.zeros((N,N))
    for i in range(len(p_states)):
        for j in range(len(p_states)):
            mat4[i,j] = np.real(np.round(p_states[i].overlap(nL*p_states[j]),7))
            mat5[i,j] = np.real(np.round(p_states[i].overlap(nR*p_states[j]),7))
    print(mat4)
    print(mat5)

def evaluate(e_sum, e_delta,e_mag, e_asym):
    H,H_i,H_no = Hamiltonian(e_sum, e_delta, e_mag, e_asym)
    #Calcul des ep non perturbés
    [nrjs, states] = H_no.eigenstates()
    LEFT, CENTER, RIGHT = -10,0, 10
    delta=  1
    positions = []
    real_states = []
    i_s = []
    for j in range(len(etats)):
        maxi = 0
        iMaxi = 0
        for i in range(len(states)):
            nrj = qt.expect(H, states[i])
            if np.abs(etats[j].overlap(states[i]))>maxi:
                maxi = np.abs(etats[j].overlap(states[i]))
                iMaxi = i

        st = states[iMaxi]
        print(str(iMaxi)+" overlap : "+str(maxi)+" @"+str(np.round(nrjs[iMaxi],4))+" "+str(qt.expect(nL, st))+" "+str(qt.expect(nR,st)))
        if qt.expect(nL, st)>0.8 and qt.expect(nL, st)<1.2 and qt.expect(nR, st)>0.8 and qt.expect(nR,st)<1.2:
            pos = CENTER
        if qt.expect(nL, st)>1.2:
            pos = LEFT
        if qt.expect(nR, st)>1.2:
            pos = RIGHT
        nrj = nrjs[iMaxi]
        real_states.append(states[iMaxi])
        i_s.append(iMaxi)
        positions.append((pos+delta*j, nrj))
        ax.scatter(pos+delta*j, nrj,marker='_', color='b', linewidths='10')
        ax.text(pos+delta*j, nrj,etats_txt[j],fontsize='14')
    #Calcul des ep perturbés 
    [nrjs, states2] = H.eigenstates()
    r_i_states = []
    print("--------------------")
    for j in range(len(etats)):
        maxi = 0
        iMaxi = 0
        for i in range(len(states2)):
            if np.abs(etats[j].overlap(states2[i]))>maxi: #Calcul de <phi|phi_perturbe>
                maxi = np.abs(etats[j].overlap(states2[i]))
                iMaxi = i
        print(str(iMaxi)+" overlap : "+str(maxi)+" @"+str(np.round(nrjs[iMaxi],4))+" "+str(qt.expect(nL, states2[iMaxi]))+" "+str(qt.expect(nR,states2[iMaxi])))
        r_i_states.append(states2[iMaxi])
        
    proj(H, r_i_states,etats)
   
    #affichage des couplages graphique
    for i,st in enumerate(etats):
        for j,st2 in enumerate(etats):
            if i!=j:
                ax.plot([positions[i][0],positions[j][0]],[positions[i][1],positions[j][1]],linewidth=10*np.abs(st.overlap(H_i*st2)))

U = 250
Um = 0
DeltaKKp = 500
tee = 1
teh = 0.1
theta = np.pi/2
fig, ax = plt.subplots()
e_sum = 1000
for i in range(0,1):
    e_delta = 0 #ou -U ou +U
    evaluate(e_sum, e_delta, 5,0.3)
    ax.set_title("e_sigma:  "+str(e_sum)+" e_delta : "+str(e_delta))
    plt.pause(0.1) # pause avec duree en secondes
plt.show()
