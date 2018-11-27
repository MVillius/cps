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
print(comm(LDoKp.dag()*RDoKp, nL)==-LDoKp.dag()*RDoKp)
print(comm(LDoKp.dag()*RUpKp.dag(), nL)==-LDoKp.dag()*RUpKp.dag())

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

def label(ii, state):
    pos = 0
    label = '('+str(ii)+" "
    nLUp = qt.expect(LUpKp.dag()*LUpKp, state)
    nLDo = qt.expect(LDoKp.dag()*LDoKp, state)
    check1 = False
    if nLUp+nLDo<0.8:
        label+='o'      
        pos = 0
    else:
        if nLUp<1.8 and nLDo<0.8:
            label+=r'$\uparrow$'
            pos = -1
        elif nLUp > 1.8 and nLDo<0.8:
            label += r"$\uparrow\uparrow$"
            pos = -2
        elif nLDo<1.8 and nLUp<0.8:
            label+=r'$\downarrow$'
            check1 = True
            pos = -1
        elif nLDo > 1.8 and nLUp<0.8:
            label += r"$\downarrow\downarrow$"
            pos = -2
        elif nLDo>0.8 and nLDo<1.6 and nLUp>0.8 and nLUp<1.6:
            label += r"$\uparrow\downarrow$"
            pos = -2
    
    label += ';'
    
    nRUp = qt.expect(RUpKp.dag()*RUpKp, state)
    nRDo = qt.expect(RDoKp.dag()*RDoKp, state)
    check2 = False
    if nRUp+nRDo<0.8:
        label+='o'
        pos += 0
    else:
        if nRUp<1.8 and nRDo<0.8:
            label+=r'$\uparrow$'
            pos += 1
        elif nRUp > 1.8 and nRDo<0.8:
            label += r"$\uparrow\uparrow$"
            pos += 2
        elif nRDo<1.8 and nRUp<0.8:
            label+=r'$\downarrow$'
            pos += 1
            check2 = True

        elif nRDo > 1.8 and nRUp<0.8:
            label += r"$\downarrow\downarrow$"
            pos += 2
        elif nRDo>0.8 and nRDo<1.6 and nRUp>0.8 and nRUp<1.6:
            label += r"$\uparrow\downarrow$"
            pos += 2

    label+="|"
    nPL = qt.expect(aLeft.dag()*aLeft, state)
    nPR = qt.expect(aRight.dag()*aRight, state)
    if nPL+nPR<0.8:
        label+='0;0'
        pos += 0
    else:
        pos+=3
        if nPL<1.8 and nPR<0.8:
            label+=r'$1;0$'
        elif nPL > 0.8 and nPR>0.8:
            label += r"$1;1$"
        elif nPL<0.8 and nPR<1.8:
            label+=r'0;1'
    label += ')'
    return label, pos

def analyse(states,energies):
    fig, ax = plt.subplots()
    for i,nrj in enumerate(energies):
        red = False
        left = qt.expect(nL, states[i])
        right = qt.expect(nR, states[i])
        photL = qt.expect(aLeft.dag()*aLeft, states[i])
        photR = qt.expect(aRight.dag()*aRight, states[i])
        if(left+right>2.2 or (left+right>0.5 and left+right<1.2)):
            continue
        if left+right<0.2 and photL+photR>0.2:
            red = True 
        st_label, pos =label(i,states[i])
        if red:
            ax.scatter(pos, nrj,marker='_', color='r', linewidths='10')
        else:
            ax.scatter(pos, nrj,marker='_', color='b', linewidths='10')
        ax.text(pos, nrj,
                        st_label,
                        fontsize='14')
    plt.show()
def Hamiltonian(e_sum, e_delta, e_mag, e_asym, g=0):
        print("------")
        print(e_sum)
        print(e_delta)
        print(e_mag*(1+e_asym))
        print(e_mag*(1-e_asym))
        print("------")
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

        Hteh = teh*np.sin(theta/2)*(LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag()) +\
               teh*np.sin(theta/2)*(LUpKp.dag()*RUpKp.dag()+LDoKp.dag()*RDoKp.dag())
        Hteh += Hteh.dag()
        
        Htee = tee*np.sin(theta/2)*(LUpKp.dag()*RUpKp + LDoKp.dag()*RDoKp) +\
               tee*np.sin(theta/2)*(-LUpKp.dag()*RDoKp + LDoKp.dag()*RUpKp)
        Htee += Htee.dag()
        Hcavite = g*nL*(aLeft+aLeft.dag()) + g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)

        H = Hchem + Hint + Htee + Hteh + Hcavite + Hphoton
        H_p = Hchem + Hint + Hphoton
        H_i = Htee + Hteh + Hcavite
        #print(Htee)
        return H,H_i,H_p
    
def info_st(s,nrj):
    print("Informations sur l'état ")
    print("Energie : "+str(round(nrj,3)))
    print("Gauche : "+str(round(qt.expect(nL, s),2))+" Droite : "+str(round(qt.expect(nR, s),2)))
    print("Up : "+str(round(qt.expect(nUp, s),2))+ " Down : "+str(round(qt.expect(nDo, s),2)))
    print("Photons g. : "+str(round(qt.expect(aLeft.dag()*aLeft, s),2)))
    print("Photons d. : "+str(round(qt.expect(aRight.dag()*aRight, s),2)))
    print("----------------------")

def proj(H0, p_states, np_states):
    
    N = len(p_states)
    J = np.zeros((6,6))
    base = [singlet, triplet, s_r, s_l,triplet_p, triplet_m]
    for i in range(6):
        for j in range(6):
            J[i,j] = np.real(base[i].overlap(H0*base[j]))
    passage = np.array([[1/np.sqrt(3),1/np.sqrt(3), 1/np.sqrt(3), 0,0,0],[1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3),0,0,0], [1/np.sqrt(3),-1/np.sqrt(3), -1/np.sqrt(3),0,0,0],[0,0,0,1,0,0], [0,0,0,0,1,0],[0,0,0,0,0,1]])
    transfo = np.linalg.inv(passage)*J*passage   
#    print(np.round(J,4))
#    print(np.linalg.inv(passage))
#    print(np.round(transfo,7))
    mat4 = np.zeros((N,N))
    mat5 = np.zeros((N,N))
    sgn = [np.sign(np.real(p_states[i].overlap(etats[i]))) for i in range(4)]

    for i in range(len(p_states)):
        for j in range(len(p_states)):
            mat4[i,j] = np.real(sgn[i]*p_states[i].overlap(nL*sgn[j]*p_states[j]))
            mat5[i,j] = np.real(sgn[i]*p_states[i].overlap(nR*sgn[j]*p_states[j]))

    #Changement de base de la matrice mat4 vers la base {S,T,Sr,Sl,T+,T-}
    passage2 = np.array([[1,1,0,0],[-1, 1,0,0],[0,0,1,0],[0,0,0,1]])
    mat6 =np.linalg.inv(passage2)*mat4*passage2

    alphaL = (mat4[0,3]-mat4[1,3])*np.sqrt(2)
    alphaR = (mat5[0,3]-mat5[1,3])*np.sqrt(2)
    return alphaL, alphaR  #mat4 = Left /mat5 = R

def evaluate(e_sum, e_delta, e_mag, e_asym):
    
    H,H_i,H_no = Hamiltonian(e_sum, e_delta, e_mag, e_asym)
#    for l in range(len(etats)):
#        for m in range(len(etats)):
#            print(np.round(np.real(etats[l].overlap(H*etats[m])),5), end = " ")
#        print()        
        
    #Calcul des ep non perturbés
    [nrjs, states] = H_no.eigenstates()
    LEFT, CENTER, RIGHT = -10,0, 10
    delta=  1
    positions = []
    real_states = []
    i_s = []
    for j in range(len(etats)):
        maxi = 0
        for i in range(len(states)):
            if np.abs(etats[j].overlap(states[i]))>maxi:
                maxi = np.abs(etats[j].overlap(states[i]))
                iMaxi = i

        st = states[iMaxi]
        #print(str(iMaxi)+" overlap : "+str(maxi)+" @"+str(np.round(nrjs[iMaxi],4))+" "+str(qt.expect(nL, st))+" "+str(qt.expect(nR,st)))
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
        #ax.scatter(pos+delta*j, nrj,marker='_', color='b', linewidths='10')
        #ax.text(pos+delta*j, nrj,etats_txt[j],fontsize='14')
    #Calcul des ep perturbés 
    [nrjs2, states2] = H.eigenstates()
    r_i_states = []
    print("--------------------")
    for j in range(len(real_states)):
        maxi = 0
        iMaxi = 0
        for i in range(len(states2)):
            if np.abs(real_states[j].overlap(states2[i]))>maxi: #Calcul de <phi|phi_perturbe>
                maxi = np.abs(real_states[j].overlap(states2[i]))
                iMaxi = i
        #print(str(iMaxi)+" overlap : "+str(maxi)+" @"+str(np.round(nrjs2[iMaxi],4))+" "+str(qt.expect(nL, states2[iMaxi]))+" "+str(qt.expect(nR,states2[iMaxi])))
        r_i_states.append(states2[iMaxi])
    
    #analyse(states2, nrjs)
    #print(states2[48])
    alphaL, alphaR = proj(H, r_i_states, etats)
    return alphaL, alphaR,nrjs


U = 250
Um = 0
#DeltaKKp = 500
tee = 3
teh = 0
theta = np.pi/12
e_sum = 1000
b_l = 5.5
b_r = 4.5

epsilons = np.linspace(-U/2, U, 2)
maxis_L = []
maxis_R = []
alphas_L = []
alphas_R = []
g = 0.4
for e_delta in epsilons:
    psi0 = singlet
    times = np.linspace(0,200,2000)
    H,_,_ = Hamiltonian(e_sum, e_delta,5, 0.1, 0.4)
    observ = [nL, nR, aLeft.dag()*aLeft, aRight.dag()*aRight]
    result = qt.mesolve(H, psi0, times, [], observ)
    maxi_left = np.max(result.expect[2])
    maxi_right = np.max(result.expect[3])
    maxis_L.append(maxi_left)
    maxis_R.append(maxi_right)
    aR, aL,_ = evaluate(e_sum, e_delta, 5,0.1)
    alphas_L.append(aL)
    alphas_R.append(aR)

plt.close('all')
fig, ax = plt.subplots(2,1)
ax[0].set_title("Sweep e_delta")
ax[0].plot(epsilons, maxis_L, label='maxi')
ax[0].plot(epsilons, alphas_L, label='alpha_l')
ax[0].set_xlabel("Time")
ax[0].set_ylabel("<n>")
ax[0].legend()
ax[1].plot(epsilons, maxis_R, label="maxi")
ax[1].plot(epsilons, alphas_R, label='alpha_r')
ax[1].set_ylabel("<a^\daggera>")
ax[1].set_xlabel("Time")
ax[1].legend()
plt.show()