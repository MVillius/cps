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

#Définition d'états
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



def Hamiltonian(e_sum, e_delta, bL, bR, g=0.2):
        print("------")
        print('e_sum =',e_sum)
        print('e_delta =',e_delta)
        print('bL =',bL)
        print('bR =',bR)
        omega0L = 2*bL
        omega0R = 2*bR
        e_LUp = (e_sum+e_delta)/2 - bL
        e_LDo = (e_sum+e_delta)/2 + bL
        e_RUp = (e_sum-e_delta)/2 - bR
        e_RDo = (e_sum-e_delta)/2 + bR   
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
        
        Hcavite = g*nL*(aLeft+aLeft.dag()) + g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)


        H = Hchem + Hint + Htee + Hteh + Hcavite + Hphoton
        H_p = Hchem + Hint + Hphoton
        H_i = Htee + Hteh + Hcavite

        return H,H_i,H_p
    
    
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
#    print(mat6)
#    print("Checking singlet like")
#    print((mat5[0,3]-mat5[1,3])*np.sqrt(2))
#    print((mat4[0,3]-mat4[1,3])*np.sqrt(2))
    alphaL = (mat4[0,3]-mat4[1,3])*np.sqrt(2)
    alphaR = (mat5[0,3]-mat5[1,3])*np.sqrt(2) 
    return alphaL, alphaR  #mat4 = Left /mat5 = R

def evaluate(e_sum, e_delta, bL, bR):
    
    H,H_i,H_no = Hamiltonian(e_sum, e_delta, bL, bR)
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
    [nrjs, states2] = H.eigenstates()
    r_i_states = []
    #print("--------------------")
    for j in range(len(real_states)):
        maxi = 0
        iMaxi = 0
        for i in range(len(states2)):
            if np.abs(real_states[j].overlap(states2[i]))>maxi: #Calcul de <phi|phi_perturbe>
                maxi = np.abs(real_states[j].overlap(states2[i]))
                iMaxi = i
        #print(str(iMaxi)+" overlap : "+str(maxi)+" @"+str(np.round(nrjs[iMaxi],4))+" "+str(qt.expect(nL, states2[iMaxi]))+" "+str(qt.expect(nR,states2[iMaxi])))
        r_i_states.append(states2[iMaxi])
        
    alphaL, alphaR = proj(H, r_i_states, etats)

    return alphaL, alphaR

def firstorder(e_delta,side):
    
    if side == 'left':
        AL = 1/((bR*bR-bL*bL)-(U-e_delta)**2-2*bL*(U-e_delta))
        BL = 1/((bL*bL-bR*bR)-(U-e_delta)**2-2*bR*(U-e_delta))
        f_alphaL = tee**2*np.sin(theta)/np.sqrt(2)*(AL+BL)
        return f_alphaL
    
    if side == 'right':
        AR = 1/((bR*bR-bL*bL)-(U+e_delta)**2-2*bL*(U+e_delta))
        BR = 1/((bL*bL-bR*bR)-(U+e_delta)**2-2*bR*(U+e_delta))
        f_alphaR = tee**2*np.sin(theta)/np.sqrt(2)*(AR+BR)
        return f_alphaR

U = 250
Um = 0
#DeltaKKp = 500
tee = 5
teh = 0
theta = np.pi/2
e_sum = 1000
bL = 3.5
bR = 3
    
if 1==1:
    Nf = 201
    Ne = 101
    g = 0.2 
    Y = []
    y = []
    y2 = []
    alphaL, alphaR = [], []
    gtot = []
    for i_e, e_delta in enumerate(np.linspace(-1.1*U,1.1*U, Ne)):
     
        print(i_e)
#        A = 1/((b_r-b_l)-(U+e_delta))*1/(-(b_l+b_r)-(U+e_delta))
#        B = -1/((b_l-b_r)-(U+e_delta))*1/(-(b_l+b_r)-(U+e_delta))
#        A2 = 1/((b_r-b_l)-(U-e_delta))*1/(-(b_l+b_r)-(U-e_delta))
#        B2 = -1/((b_l-b_r)-(U-e_delta))*1/(-(b_l+b_r)-(U-e_delta))
#        y.append(tee**2*np.sin(theta)/np.sqrt(2)*(A-B))
#        y2.append(tee**2*np.sin(theta)/np.sqrt(2)*(A2-B2))
        
        aL, aR = evaluate(e_sum, e_delta, bL, bR, g)
        alphaL.append(aL)
        alphaR.append(aR)
        gtot.append(g*(aL-aR))
        
    f_alphaL = firstorder(np.linspace(-1.1*U,1.1*U,Nf),'left')
    f_alphaR = firstorder(np.linspace(-1.1*U,1.1*U,Nf),'right')
    f_gtot = g*(f_alphaL-f_alphaR)
    
    
#    plt.plot(es, y,label="alpha_L")
    plt.close('all')
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.linspace(-1.1*U,1.1*U, Nf)/U, f_alphaR, '--', label="1st order")
    ax[0].plot(np.linspace(-1.1*U,1.1*U, Ne)/U, alphaR, '*', label="numerical diag")
    ax[0].set_ylabel(r"$\alpha_R$")
    ax[0].set_ylim([-0.2,0.2])
    ax[0].legend()
    ax[0].grid()
    
    ax[1].plot(np.linspace(-1.1*U,1.1*U, Nf)/U, f_alphaL, '--', label="1st order")
    ax[1].plot(np.linspace(-1.1*U,1.1*U, Ne)/U, alphaL, '*', label="numerical diag")
    ax[1].set_ylabel(r"$\alpha_L$")
    ax[1].set_xlabel(r"$e_\Delta/U$")
    ax[1].set_ylim([-0.2,0.2])
    ax[1].legend()
    ax[1].grid()
    
    fig, ax = plt.subplots()
    ax.plot(np.linspace(-1.1*U,1.1*U, Nf)/U, f_gtot, '--', label="1st order")
    ax.plot(np.linspace(-1.1*U,1.1*U, Ne)/U, gtot, '*', label="numerical diag")
    ax.set_ylabel(r"$g(\alpha_L - \alpha_R)$ (GHz)")
    ax.set_ylim([-0.1,0.1])
    ax.legend()
    ax.grid()
    ax.set_xlabel(r"$e_\Delta/U$")
    #ax.title(r'$t_{ee}=5$, $g=0.2$, U=250')
    #plt.show()
    
if 1==0:
    g=0.2
    e_delta = 0.9*U
    psi0 = singlet
    times = np.linspace(0,200,2001)
    H,_,_ = Hamiltonian(e_sum, e_delta, bL, bR, g)
    observ = [nL, nR, aLeft.dag()*aLeft, aRight.dag()*aRight]
    result = qt.mesolve(H, psi0, times, [], observ)
#    
#    exp_nL, exp_nR, exp_aL, exp_aR = [], [], [], []
#    for t, tim in enumerate(times):
#        exp_nL.append(result.expect[0][t])
#        exp_nR.append(result.expect[1][t])
#        exp_aL.append(result.expect[2][t])
#        exp_aR.append(result.expect[3][t])
    
#    plt.close('all')
    fig, ax = plt.subplots(2,1)
    ax[0].plot(times, result.expect[0], label='left')
    ax[0].plot(times, result.expect[1], label='right')
    ax[0].set_ylabel("<n>")
    ax[0].legend()
    
    ax[1].plot(times, result.expect[2], label="left")
    ax[1].plot(times, result.expect[3], label='right')
    ax[1].set_ylabel("<a^\daggera>")
<<<<<<< HEAD
    ax[1].legend()
    
if 1==1:
    g = 0.2 
    es = np.linspace(-1.1*U,1.1*U, 51)
    Y = []
    y = []
    y2 = []
    alphaL, alphaR = [], []
    gtot = []
    for i_e, e_delta in enumerate(es):
        print(i_e)
        A = 1/((b_r-b_l)-(U+e_delta))*1/(-(b_l+b_r)-(U+e_delta))
        B = -1/((b_l-b_r)-(U+e_delta))*1/(-(b_l+b_r)-(U+e_delta))
        A2 = 1/((b_r-b_l)-(U-e_delta))*1/(-(b_l+b_r)-(U-e_delta))
        B2 = -1/((b_l-b_r)-(U-e_delta))*1/(-(b_l+b_r)-(U-e_delta))
        y.append(tee**2*np.sin(theta)/np.sqrt(2)*(A-B))
        y2.append(tee**2*np.sin(theta)/np.sqrt(2)*(A2-B2))
        aL, aR = evaluate(e_sum, e_delta, 5,0.1)
        alphaL.append(aL)
        alphaR.append(aR)
        gtot.append(g*(aL-aR))
    es /=U
#    plt.plot(es, y,label="alpha_L")
#    plt.close('all')
    fig, ax = plt.subplots(2,1)
    ax[0].plot(es, y, '--', label="1st order")
    ax[0].plot(es, alphaR, '*', label="exact diag")
    ax[0].set_ylabel("alpha_R")
    ax[0].legend()
    
    ax[1].plot(es, y2, '--', label="1st order")
    ax[1].plot(es, alphaL, '*', label="exact diag")
    ax[1].set_ylabel("alpha_L")
    ax[1].legend()
    
    fig, ax = plt.subplots()
    ax.plot(es, gtot, '*', label="exact diag")
    ax.set_ylabel("g*(alpha_L - alpha_R) (GHz)")
    ax.legend()
    ax.set_xlabel("e_delta/U")
    #plt.show()
    
=======
    ax[1].legend()
>>>>>>> rewriting first order
