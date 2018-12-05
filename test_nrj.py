import numpy as np
import qutip as qt
from numpy import linalg as LA
from scipy import stats
import matplotlib.pyplot as plt
import time


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
#print(comm(LDoKp.dag()*RDoKp, nL)==-LDoKp.dag()*RDoKp)
#print(comm(LDoKp.dag()*RUpKp.dag(), nL)==-LDoKp.dag()*RUpKp.dag())

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
    
def Hamiltonian(e_sum, e_delta, bL, bR, g=0.2, scanL=0, scanR=0):
        print("------")
        print('e_sum='+str(e_sum))
        print('e_delta='+str(e_delta))
        print('bL='+str(bL))
        print('bR='+str(bR))
        omega0L = 2*bL + scanL
        omega0R = 2*bR + scanR
        e_LUp = (e_sum+e_delta)/2 - bL
        e_LDo = (e_sum+e_delta)/2 + bL
        e_RUp = (e_sum-e_delta)/2 - bR
        e_RDo = (e_sum-e_delta)/2 + bR
        print('omega0L='+str(omega0L))
        print('omega0R='+str(omega0R))
        Hchem = e_LUp*(LUpKp.dag()*LUpKp) +\
                e_LDo*(LDoKp.dag()*LDoKp) + \
                e_RUp*(RUpKp.dag()*RUpKp) + \
                e_RDo*(RDoKp.dag()*RDoKp)

        Hint = (U/2)*(nL*(nL-1)+nR*(nR-1)) + Um*nL*nR

        Hteh = teh*np.cos(theta/2)*(LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag()) +\
               teh*np.sin(theta/2)*(LUpKp.dag()*RUpKp.dag()+LDoKp.dag()*RDoKp.dag())
        Hteh += Hteh.dag()
        
        Htee = tee*np.cos(theta/2)*(LUpKp.dag()*RUpKp + LDoKp.dag()*RDoKp) +\
               tee*np.sin(theta/2)*(-LUpKp.dag()*RDoKp + LDoKp.dag()*RUpKp)
        Htee += Htee.dag()
        
        Hcavite = g*nL*(aLeft+aLeft.dag()) + g*nR*(aRight+aRight.dag())
        Hphoton = omega0L*(aLeft.dag()*aLeft)+omega0R*(aRight.dag()*aRight)
#        print(nL*(LUpKp.dag()*RDoKp.dag()-LDoKp.dag()*RUpKp.dag()).dag()*(LUpKp.dag()*RUpKp.dag()))
        H = Hchem + Hint + Htee + Hteh + Hcavite + Hphoton
        H_p = Hchem + Hint + Hphoton
        H_i = Htee + Hteh + Hcavite
#        print(Htee)
        H_nocavity = Hchem+Hint+Htee+Hteh
#        [n,s] = H_nocavity.eigenstates()
#        print(n)
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
#    J = np.zeros((6,6))
#    base = [singlet, triplet, s_r, s_l,triplet_p, triplet_m]
#    for i in range(6):
#        for j in range(6):
#            J[i,j] = np.real(base[i].overlap(H0*base[j]))
#    passage = np.array([[1/np.sqrt(3),1/np.sqrt(3), 1/np.sqrt(3), 0,0,0],[1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3),0,0,0], [1/np.sqrt(3),-1/np.sqrt(3), -1/np.sqrt(3),0,0,0],[0,0,0,1,0,0], [0,0,0,0,1,0],[0,0,0,0,0,1]])
#    transfo = np.linalg.inv(passage)*J*passage   
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
#    passage2 = np.array([[1,1,0,0],[-1, 1,0,0],[0,0,1,0],[0,0,0,1]])
#    mat6 =np.linalg.inv(passage2)*mat4*passage2

    alphaL = (mat4[0,3]-mat4[1,3])*np.sqrt(2)
    alphaR = (mat5[0,3]-mat5[1,3])*np.sqrt(2)
    return alphaL, alphaR  #mat4 = Left /mat5 = R

def evaluate(e_sum, e_delta, bL, bR, g, scanL, scanR):
    
    H,H_i,H_no = Hamiltonian(e_sum, e_delta, bL, bR, g, scanL, scanR)
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
    
    #analyse(states2, nrjs)
    #print(states2[48])
    alphaL, alphaR = proj(H, r_i_states, etats)
    return alphaL, alphaR, nrjs

"""
n_min = 45
n_max = 62
epsilons = np.linspace(-U*1.05,-0.95*U,100)
trace = np.zeros((n_max-n_min+1,len(epsilons)))

for j,eps in enumerate(epsilons):
    H,_,_ = Hamiltonian(e_sum, eps,5, 0.1, 0.4)
    [nrjs, states] = H.eigenstates()
    trace[:,j] = nrjs[n_min:n_max+1].copy()
for i in range(trace.shape[0]):
    plt.scatter(epsilons/U, trace[i,:])
plt.show()
"""
def identify_transition(delta_e, states, energies, frequency):
    print("Looking for transition @ 1/2pi*"+str(np.round(delta_e,5))+" GHz")
    print("Cavity frequency : "+str(frequency))
    couple_maxi = (0,0)
    maxi = 1000
    for i in range(len(states)):
        for j in range(len(states)):
            if np.abs((energies[i]-(energies[j]+frequency))-delta_e)<maxi:
                maxi = np.abs((energies[i]-(energies[j]+frequency))-delta_e)
                couple_maxi = (i,j)

    print("Between states #"+str(couple_maxi[0])+" and #"+str(couple_maxi[1]))
    print("Frequency found : "+str(energies[couple_maxi[0]]-energies[couple_maxi[1]]-frequency)+" *1/2pi GHz")
    print()
    print(info_st(states[couple_maxi[0]], energies[couple_maxi[0]],couple_maxi[0]))
    print(info_st(states[couple_maxi[1]], energies[couple_maxi[1]],couple_maxi[1]))

if 1==1: 
    plt.close('all')
    
    U = 250
    Um = 0
    tee = 3.45
    teh = 3
    theta = np.pi/2
    e_sum = 1000
    e_delta = 0.85*U
    bL = 3.5
    bR = 3.25
    g = 0.2
    scanR = np.linspace(-0.1, 0.1, 21)
    
    psi0 = singlet
    n = 10000
    times = np.linspace(0,1000,n)
    k = np.arange(n)
    Fs = n/(times[-1]-times[0])
    print("Sample frequency : "+str(Fs))
    print("Fmin : 4")
    frq = np.linspace(0, Fs/2, int(n/2))*2*np.pi   # np.fft.fftfreq(n) ???
    
    maxL_time, maxR_time = [], []
    for i_s, sR in enumerate(scanR):
        start_time = time.time()
        H,_,_ = Hamiltonian(e_sum, e_delta, bL, bR, g, scanR=sR)
#       [nn, ss] = H.eigenstates()
        observ = [nL, nR, aLeft.dag()*aLeft, aRight.dag()*aRight]
        result = qt.mesolve(H, psi0, times, [], observ)
        maxL_time.append(np.max(result.expect[2]))
        argmaxL_time = np.argmax(result.expect[2])
        gL_time = np.pi/argmaxL_time
        maxR_time.append(np.max(result.expect[3]))
        argmaxR_time = np.argmax(result.expect[3])
        gR_time = np.pi/argmaxR_time

        YL_FFT = np.abs(np.fft.fft(result.expect[2]-np.max(result.expect[2])/2))/n
        YL_FFT = YL_FFT[range(int(n/2))]
        YR_FFT = np.abs(np.fft.fft(result.expect[3]-np.max(result.expect[3])/2))/n
        YR_FFT = YR_FFT[range(int(n/2))]
        gL_freq = frq[np.argmax(YL_FFT)]
        gR_freq = frq[np.argmax(YR_FFT)]
        
        print('location max time (L,R) = '+str(argmaxL_time)+', '+str(argmaxR_time))
        print('gL (time/freq, GHz) = '+str(gL_time)+', '+str(gL_freq))
        print('gR (time/freq, GHz) = '+str(gR_time)+', '+str(gR_freq))
        
    
        fig, ax = plt.subplots(2,1)
        ax[0].set_title("Evolution for e_delta = "+str(np.round(e_delta/U,3))+"U")
        ax[0].plot(times, result.expect[2], label='left')
        ax[0].plot(times, result.expect[3], label='right')
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("<âa>")
        ax[0].legend()
        
        ax[1].plot(frq, YL_FFT, label="FFT_L")
        ax[1].plot(frq, YR_FFT, label='FFT_R')
        ax[1].set_ylabel("amplitude")
        ax[1].set_xlabel("frequency")
        ax[1].legend()
        plt.show()
        
        print('time elapsed = '+str(time.time()-start_time))
       
    fig, ax = plt.subplots(2,1)
    ax[0].set_title('e_delta, bL, bR, g = '+str(e_delta)+', '+str(bL)+', '+str(bR)+', '+str(g))
    ax[0].plot(scanR, maxL_time, label='left')
    ax[0].plot(scanR, maxR_time, label='right')
    ax[0].set_xlabel(r"\omega_R - 2b_R")
    ax[0].set_ylabel(r"max(<a_i^\dagger a_i>)")
    ax[0].legend()
    
    plt.show()
        
    
if 1==0:
    U = 250
    Um = 0
    tee = 3.45
    teh = 3
    theta = np.pi/2
    e_sum = 1000
    e_deltas = np.linspace(-U,-0.97*U, 100)
    bL = 5.5
    bR = 4.5
    g = 0.2

    Y = []
    y = []
    y2 = []
    alphaL, alphaR = [], []
    gtot = []
    st_1, st_2 = [],[]
    for i_e, e_delta in enumerate(e_deltas):
        print(i_e)
        A = 1/((b_r-b_l)-(U+e_delta))*1/(-(b_l+b_r)-(U+e_delta))
        B = -1/((b_l-b_r)-(U+e_delta))*1/(-(b_l+b_r)-(U+e_delta))
        A2 = 1/((b_r-b_l)-(U-e_delta))*1/(-(b_l+b_r)-(U-e_delta))
        B2 = -1/((b_l-b_r)-(U-e_delta))*1/(-(b_l+b_r)-(U-e_delta))
        y.append(tee**2*np.sin(theta)/np.sqrt(2)*(A-B))
        y2.append(tee**2*np.sin(theta)/np.sqrt(2)*(A2-B2))
        aL, aR, nrjs = evaluate(e_sum, e_delta, bL, bR, g)
        alphaL.append(aL)
        alphaR.append(aR)
        st_1.append(nrjs[48])
        st_2.append(nrjs[50])
        gtot.append(g*(aL-aR))
    e_deltas /=U
#    plt.plot(es, y,label="alpha_L")
#    plt.close('all')
    fig, ax = plt.subplots(2,1)
    ax[0].plot(e_deltas, y, '--', label="1st order")
    ax[0].plot(e_deltas, alphaL, '*', label="exact diag")
    ax[0].set_ylabel("alpha_R")
    ax[0].set_xlabel("e_delta")
    ax[0].legend()
    
    ax[1].plot(e_deltas, y2, '--', label="1st order")
    ax[1].plot(e_deltas, alphaR, '*', label="exact diag")
    ax[1].set_ylabel("alpha_L")
    ax[1].set_xlabel("e_delta")
    ax[1].legend()
    #ax.title("alphaL and alphaR coefficients")
    fig, ax = plt.subplots()
    #ax.scatter(es,st_1)
    #ax.scatter(es,st_2)
    ax.plot(es, gtot, '*', label="exact      diag")
    ax.set_ylabel("g*(alpha_L - alpha_R) (GHz)")
    ax.legend()
    ax.set_xlabel("e_delta/U")
    plt.show()
    