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
s_ = (LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag())/np.sqrt(2)
tp =LUpKp.dag()*RUpKp.dag()
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
def HamiltonianRWA(p, e_sum, e_delta, bl, br, g):
    omega0R = 2*br
    e_LUp = (e_sum+e_delta)/2 - bl
    e_LDo = (e_sum+e_delta)/2 + bl
    e_RUp = (e_sum-e_delta)/2 - br
    e_RDo = (e_sum-e_delta)/2 + br

    sigma_plus_L = LUpKp.dag()*RUpKp.dag()*LDoKp*RUpKp
    sigma_plus_R = LUpKp.dag()*RUpKp.dag()*LUpKp*RDoKp
    sigma_moins_L =LDoKp.dag()*RUpKp.dag()*LUpKp*RUpKp
    sigma_moins_R =LUpKp.dag()*RDoKp.dag()*LUpKp*RUpKp

    sigma_z_L = LUpKp.dag()*RUpKp.dag()*LUpKp*RUpKp-LDoKp.dag()*RUpKp.dag()*LDoKp*RUpKp

    alpha_R = 1
    #alpha_R =p.tee*2*np.sin(p.theta/2)/(np.sqrt(2)*(p.U-e_delta)**2)
    Hcoup = g*alpha_R*(sigma_moins_R*aRight.dag()-sigma_moins_L*aRight.dag()+sigma_plus_R*aRight-sigma_plus_L*aRight)
    Hcoup+=Hcoup.dag()
    H = ((e_LUp+e_RDo)*LUpKp.dag()*RDoKp.dag()*LUpKp*RDoKp+(e_LDo+e_RUp)*LDoKp.dag()*RUpKp.dag()*LDoKp*RUpKp+(e_LUp+e_RUp)*LUpKp.dag()*RUpKp.dag()*LUpKp*RUpKp) + omega0R*aLeft.dag()*aLeft -omega0R*sigma_z_L
    return H+Hcoup

def HamiltonianRWA2(p, e_sum, e_delta, bl, br, g):
    omega0R = 2*br
    omega0L = 2*bl

    e_LUp = (e_sum+e_delta)/2 - bl
    e_LDo = (e_sum+e_delta)/2 + bl
    e_RUp = (e_sum-e_delta)/2 - br
    e_RDo = (e_sum-e_delta)/2 + br

    sigma_plus_L = LUpKp.dag()*RUpKp.dag()*LDoKp*RUpKp
    sigma_plus_R = LUpKp.dag()*RUpKp.dag()*LUpKp*RDoKp
    sigma_moins_L =LDoKp.dag()*RUpKp.dag()*LUpKp*RUpKp
    sigma_moins_R =LUpKp.dag()*RDoKp.dag()*LUpKp*RUpKp

    sigma_z_L = LUpKp.dag()*RUpKp.dag()*LUpKp*RUpKp-LDoKp.dag()*RUpKp.dag()*LDoKp*RUpKp
    sigma_z_R = LUpKp.dag()*RUpKp.dag()*LUpKp*RUpKp-LUpKp.dag()*RDoKp.dag()*LUpKp*RDoKp

    alpha_L, alpha_R = 1,1
    Delta_L, Delta_R = (2*omega0L-omega0R)/3, (2*omega0R-omega0L)/3
    Omega = (omega0L+omega0R)/2
    Hc = Delta_L*sigma_z_L+Delta_R*sigma_z_R
    Hcoupl = g**2*alpha_L*alpha_R/2*(1/(Omega-omega0L)+1/(Omega-omega0R))*sigma_moins_R*aLeft.dag()*sigma_moins_L*aRight.dag()
    Hcoupl += Hcoupl.dag()
    H = Hc+Hcoupl
    return H
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
        Hcavite = p.g*singlet.overlap(nL*triplet_p)*(s_.dag()*tp+tp.dag()*s_)*(aLeft+aLeft.dag()) + p.g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)
        H = Hchem + Hint + Htee + Hteh + Hcavite + Hphoton
        H_p = Hchem + Hint + Hphoton
        H_i = Htee + Hteh + Hcavite
        H_nocavity = Hchem+Hint+Htee+Hteh
        [n,s] = H_nocavity.eigenstates()
        return H,H_i,H_p
    
def info_st(s,nrj,label=-1):
    print("Informations sur l'état "+str(label))
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
        print(str(iMaxi)+" overlap : "+str(maxi)+" @"+str(np.round(nrjs2[iMaxi],4))+" "+str(qt.expect(nL, states2[iMaxi]))+" "+str(qt.expect(nR,states2[iMaxi])))
        r_i_states.append(states2[iMaxi])
    alphaL, alphaR = proj(H, r_i_states, etats)
    return alphaL, alphaR,nrjs


## Parameters
X = Params(50, 0, 6.5,5.5 ,50, 0, np.pi/2, 0.2, 1,1) #e_s, e_d, b_l, b_r,U, Um,theta, g,te, teh)
n = 7000   
###

times = np.linspace(0,1000,n)
psi0 = singlet
H,_,_ = Hamiltonian(X, X.e_sum, X.e_delta, X.b_l, X.b_r,X.theta,0.0059,0, 1) 
[nn, ss] = H.eigenstates()
observ = [nL, nR, aLeft.dag()*aLeft, aRight.dag()*aRight]
result = qt.mesolve(H, psi0, times, [], observ)

### Plot
plt.show()
plt.close('all')
fig, ax = plt.subplots(2,1)
ax[0].set_title("Evolution for e_delta = "+str(np.round(X.e_delta/X.U,3))+"U")
ax[0].plot(times, result.expect[2], label='singlet')
ax[0].plot(times,result.expect[3], label='triplet')
ax[0].set_xlabel("Time")
ax[0].set_ylabel("<âa>")
ax[0].legend()
plt.show()
###