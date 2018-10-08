#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:49:19 2018

@author: Zaki
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import qutip as qt
'''
Generate operators for an arbitrary number of orbitals
Each orbital is two-fold degenerate
'''
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

nL = LUpK.dag()*LUpK + LDoK.dag()*LDoK + LUpKp.dag()*LUpKp + LDoKp.dag()*LDoKp
nR = RUpK.dag()*RUpK + RDoK.dag()*RDoK + RUpKp.dag()*RUpKp + RDoKp.dag()*RDoKp
nUp = LUpK.dag()*LUpK + LUpKp.dag()*LUpKp +RUpK.dag()*RUpK+ RUpKp.dag()*RUpKp 
nDo = LDoK.dag()*LDoK + LDoKp.dag()*LDoKp +RDoK.dag()*RDoK+ RDoKp.dag()*RDoKp

nPhL = aLeft.dag()*aLeft
nPhR = aRight.dag()*aRight
nPhTot = nPhL + nPhR
ntot = nL + nR
SX, SY, SZ = [],[],[]
ops_b = [LUpK, LDoK, LUpKp, LDoKp, RUpK, RDoK, RUpKp, RDoKp]

for iSide in range(2):
    for iK in range(2):
        SZ.append(ops_b[iK*2+iSide*4].dag()*ops_b[iK*2+iSide*4]-ops_b[iK*2+iSide*4+1].dag()*ops_b[iK*2+iSide*4+1])
        SY.append(1j*(ops_b[iK*2+iSide*4+1].dag()*ops_b[iK*2+iSide*4]-ops_b[iK*2+iSide*4].dag()*ops_b[iK*2+iSide*4+1]))
        SX.append(ops_b[iK*2+iSide*4].dag()*ops_b[iK*2+iSide*4+1]+ops_b[iK*2+iSide*4+1].dag()*ops_b[iK*2+iSide*4])

S_X = SX[0]+SX[1]+SX[2]+SX[3]
S_Y = SY[0]+SY[1]+SY[2]+SY[3]
S_Z = SZ[0]+SZ[1]+SZ[2]+SZ[3]
S_tot = S_X**2+S_Y**2+S_Z**2
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


for k1, op1 in ops.items():
    for k2, op2 in ops.items():
        if op2 == op1:
            assert op1.dag()*op2+op2*op1.dag() == II
        else:
            assert op1.dag()*op2+op2*op1.dag() == 0*II

#assert qt.commutator(aRight, aRight.dag())==I
class CPSF:
    
    def __init__(self,e_sum,e_diff,e_mag,theta,mag_asym, omega):
        e_LUp = (e_sum+e_diff)/2 - e_mag*(1+mag_asym)
        e_LDo = (e_sum+e_diff)/2 + e_mag*(1+mag_asym)
        e_RUp = (e_sum-e_diff)/2 - e_mag*(1-mag_asym)
        e_RDo = (e_sum-e_diff)/2 + e_mag*(1-mag_asym)
        #print(e_LUp+e_RDo+Um)
        #print(e_RDo+e_RDo+U)
        #print(e_LDo+e_RDo+omega0+Um)
        '''
        e_left is epsilon_left
        e_right is same
        e_mag is the zeeman splitting
        mag_asym creates asymmetry between zeeman left and right
        theta is the ferro polarization angle
        '''
        gl=0.5
        gr=0.5
        omega0L = omega   
        Hcavite = gl*nL*(aLeft+aLeft.dag()) 
        Hphoton = omega0L*(aLeft.dag()*aLeft)

        Hchem = e_LUp*(LUpK.dag()*LUpK + LUpKp.dag()*LUpKp) +\
                e_LDo*(LDoK.dag()*LDoK + LDoKp.dag()*LDoKp) + \
                e_RUp*(RUpK.dag()*RUpK + RUpKp.dag()*RUpKp) + \
                e_RDo*(RDoK.dag()*RDoK + RDoKp.dag()*RDoKp)

        Hteh = teh*np.cos(theta/2)*(LUpK.dag()*RDoKp.dag() - LDoKp.dag()*RUpK.dag()) +\
               teh*np.cos(theta/2)*(LUpKp.dag()*RDoK.dag() - LDoK.dag()*RUpKp.dag()) +\
               teh*np.sin(theta/2)*(LUpK.dag()*RUpKp.dag() + LDoKp.dag()*RDoK.dag()) +\
               teh*np.sin(theta/2)*(LUpKp.dag()*RUpK.dag() + LDoK.dag()*RDoKp.dag())
        Hteh += Hteh.dag()
        
        Htee = tee*np.cos(theta/2)*(LUpK.dag()*RUpK + LDoK.dag()*RDoK+\
                                     LUpKp.dag()*RUpKp + LDoKp.dag()*RDoKp) +\
               tee*np.sin(theta/2)*(-LUpK.dag()*RDoK + LDoK.dag()*RUpK+\
                                    -LUpKp.dag()*RDoKp + LDoKp.dag()*RUpKp)
        Htee += Htee.dag()
        
        Hint = (U/2)*(nL*(nL-1)+nR*(nR-1)) + Um*nL*nR
        
        HKKp = DeltaKKp * (LUpK.dag()*LUpK+LDoK.dag()*LDoK+\
                           RUpK.dag()*RUpK+RDoK.dag()*RDoK-\
                           LUpKp.dag()*LUpKp-LDoKp.dag()*LDoKp-\
                           RUpKp.dag()*RUpKp-RDoKp.dag()*RDoKp)
        
        #self.H = Hchem + Hteh + Htee + Hint + HKKp + Hphoton + Hcavite
        self.H0 = Hchem + Hint + Hphoton
        self.H1 = Hcavite + Hteh + Htee
        self.H = self.H0 + self.H1
        '''
        the hamiltonian is written in the new basis
        LUp --> L+
        LDo --> L-
        '''

    
    def diagonalize(self):
        #if not hasattr(self,'energies'): #avoid diagonalizing mutlitple times
        Hamiltonian = self.H
        [nrj, st] = Hamiltonian.eigenstates()
        self.energies = nrj
        self.states = st
           
    def diagonalizeH0(self):
        Hamiltonian0 = self.H0
        [nrj, st] = Hamiltonian0.eigenstates()
        return [nrj, st]
    def overlap(self,state1,state2):
        s = np.abs(self.states[state1].overlap(self.states[state2]))**2
        return s

def debugH(H):

    basis_ops = [[II],
             [LUpK.dag()*RDoK.dag(), LUpKp.dag()*RDoK.dag(), LUpK.dag()*RDoKp.dag(), LUpKp.dag()*RDoKp.dag()],
             [LDoK.dag()*RUpK.dag(), LDoKp.dag()*RUpK.dag(), LDoK.dag()*RUpKp.dag(), LDoKp.dag()*RUpKp.dag()],
             [LUpK.dag()*RUpK.dag(), LUpKp.dag()*RUpK.dag(), LUpK.dag()*RUpKp.dag(), LUpKp.dag()*RUpKp.dag()],
             [LDoK.dag()*RDoK.dag(), LDoKp.dag()*RDoK.dag(), LDoK.dag()*RDoKp.dag(), LDoKp.dag()*RDoKp.dag()],
             [LUpK.dag()*LDoK.dag(), LUpKp.dag()*LDoK.dag(), LUpK.dag()*LDoKp.dag(), LUpKp.dag()*LDoKp.dag()],
             [RUpK.dag()*RDoK.dag(), RUpKp.dag()*RDoK.dag(), RUpK.dag()*RDoKp.dag(), RUpKp.dag()*RDoKp.dag()],
             [LUpK.dag()*LUpK.dag(), LUpKp.dag()*LUpK.dag(), LUpK.dag()*LUpKp.dag(), LUpKp.dag()*LUpKp.dag()],
             [RUpK.dag()*RUpK.dag(), RUpKp.dag()*RUpK.dag(), RUpK.dag()*RUpKp.dag(), RUpKp.dag()*RUpKp.dag()],
             [LDoK.dag()*LDoK.dag(), LDoKp.dag()*LDoK.dag(), LDoK.dag()*LDoKp.dag(), LDoKp.dag()*LDoKp.dag()],
             [RDoK.dag()*RDoK.dag(), RDoKp.dag()*RDoK.dag(), RDoK.dag()*RDoKp.dag(), RDoKp.dag()*RDoKp.dag()],
             [aLeft.dag()*aLeft], [aRight.dag()*aRight]]
    basis = []
    for basis_ops_group in basis_ops:
        bgroup = [b*vac for b in basis_ops_group]
        basis.append(bgroup)
    Hmat = np.zeros((13,13))
    nLmat = np.zeros((13,13))
    for jj, gj in enumerate(basis):
        for kk, gk in enumerate(basis):
            for vj in gj:
                for vk in gk:
                    Hmat[jj, kk] += np.abs(vj.overlap(H*vk))
                    nLmat[jj, kk] += np.abs(vj.overlap(nL*vk))
            Hmat[jj, kk] = np.round(Hmat[jj, kk]/4, decimals=1)
    print(Hmat)
    print(nLmat)

plt.close('all')

'''
teh and tee should vary with epsilon_sum
we set them to constant values
'''
def state_label(state):
    label = '('
    nLUpK = qt.expect(LUpK.dag()*LUpK, state)
    nLDoK = qt.expect(LDoK.dag()*LDoK, state)
    #print(('%s, %s') % (round(nLUpK,2), round(nLDoK,2)))
    if nLUpK+nLDoK<0.8:
        label+='o'
    elif nLUpK*nLDoK>0.8:
        label+=r'$\uparrow\downarrow$'
    elif nLUpK > 0.9:
        label += r"$\uparrow$"
    elif nLDoK > 0.9:
        label += r"$\downarrow$"
        
    label += ','
        
    nLUpKp = qt.expect(LUpKp.dag()*LUpKp, state)
    nLDoKp = qt.expect(LDoKp.dag()*LDoKp, state)
    #print(('%s, %s') % (round(nLUpKp,2), round(nLDoKp,2)))
    if nLUpKp+nLDoKp<0.8:
        label+='o'
    elif nLUpKp*nLDoKp>0.8:
        label+=r'$\uparrow\downarrow$'
    elif nLUpKp > 0.9:
        label+=r"$\uparrow$"
    elif nLDoKp > 0.9:
        label+=r"$\downarrow$"
        
    label += ';'
    
    nRUpK = qt.expect(RUpK.dag()*RUpK, state)
    nRDoK = qt.expect(RDoK.dag()*RDoK, state)
    print(('%s, %s') % (round(nRUpK,2), round(nRDoK,2)))
    if nRUpK+nRDoK<0.8:
        label+='o'
    elif nRUpK*nRDoK>0.8:
        label+=r'$\uparrow\downarrow$'
    elif nRUpK > 0.9:
        label+=r"$\uparrow$"
    elif nRDoK > 0.9:
        label+=r"$\downarrow$"
        
    label += ','
    
    nRUpKp = qt.expect(RUpKp.dag()*RUpKp, state)
    nRDoKp = qt.expect(RDoKp.dag()*RDoKp, state)
    print(('%s, %s') % (round(nRUpKp,2), round(nRDoKp,2)))
    if nRUpKp+nRDoKp<0.8:
        label+='o'
    elif nRUpKp*nRDoKp>0.8:
        label+=r'$\uparrow\downarrow$'
    elif nRUpKp > 0.9:
        label+=r"$\uparrow$"
    elif nRDoKp > 0.9:
        label+="$\downarrow$"
        
    label += ')'
    return label

def find(states,st_ref, photon1, photon2): #Essaie de trouver un état propre du hamiltonien correspondant à un état prédéfini
    maxi = 0
    iMaxi = 0
    for i,st in enumerate(states):
        if qt.expect(ntot, st)<2.2 and qt.expect(ntot, st)>0.8:
            if np.abs(st_ref.overlap(st))**2>maxi and (qt.expect(nPhL,st)>0.8 or not photon1) and (qt.expect(nPhR, st)>0.8 or not photon2):
                maxi = np.abs(st_ref.overlap(st))**2
                iMaxi = i
    if not iMaxi:
        print("Failed to find the desired state ! ")
    return iMaxi
def state_label_compact(ii, state):
    pos = 0
    
    label = '('+str(ii)+" "
    nLUp = qt.expect(LUpK.dag()*LUpK+LUpKp.dag()*LUpKp, state)
    nLDo = qt.expect(LDoK.dag()*LDoK+LDoKp.dag()*LDoKp, state)

    #print('nL: %s' % round(nLUp+nLDo,2))
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
    
    nRUp = qt.expect(RUpK.dag()*RUpK+RUpKp.dag()*RUpKp, state)
    nRDo = qt.expect(RDoK.dag()*RDoK+RDoKp.dag()*RDoKp, state)
    check2 = False
    #print('nR: %s' % round(nRUp+nRDo,2))
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
    nPL = qt.expect(nPhL, state)
    nPR = qt.expect(nPhR, state)
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
    """
    print(str(ii), end = " ,")
    print('nLUp: %s' % round(nLUp,2), end=" ")
    print('nLDo: %s' % round(nLDo,2), end=" ")
    print('nRUp: %s' % round(nRUp,2), end=" ")
    print('nRDo: %s' % round(nRDo,2), end=" ")
    print('nPhL: %s' % round(nPL,2), end=" ")
    print('nPhR: %s' % round(nPR,2), end=" ")
    print()
    """
    return label, pos
def analyse(nb,omega): 
    a = CPSF(e_sum,e_diff,e_mag,theta,mag_asym,omega)
    [nrj,st] = a.H.eigenstates()
    state = st[nb]
    nLUp = qt.expect(LUpK.dag()*LUpK+LUpKp.dag()*LUpKp, state)
    nLDo = qt.expect(LDoK.dag()*LDoK+LDoKp.dag()*LDoKp, state)
    if qt.expect(ntot, state)>1.2:
        print(str(i)+" ;",end="")
        if nLUp+nLDo<0.8:
            print("o,",end=" ")
        else:
            if nLUp<1.8 and nLDo<0.8:
                print("Up,", end=" ")
            elif nLUp > 1.8 and nLDo<0.8:
                print("UpUp,", end=" ")
            elif nLDo<1.8 and nLUp<0.8:
                print("Down,", end=" ")
            elif nLDo > 1.8 and nLUp<0.8:
                print("DownDown,",end=" ")
            elif nLDo>0.8 and nLDo<1.6 and nLUp>0.8 and nLUp<1.6:
                print("UpDown,",end="")
        
        nRUp = qt.expect(RUpK.dag()*RUpK+RUpKp.dag()*RUpKp, state)
        nRDo = qt.expect(RDoK.dag()*RDoK+RDoKp.dag()*RDoKp, state)
        if nRUp+nRDo<0.8:
            print("o,",end=" ")
        else:
            if nRUp<1.8 and nRDo<0.8:
                print("Up,", end=" ")
            elif nRUp > 1.8 and nRDo<0.8:
                print("UpUp,", end=" ")
            elif nRDo<1.8 and nRUp<0.8:
                print("Down,", end=" ")
            elif nRDo > 1.8 and nRUp<0.8:
                print("DownDown,",end=" ")
            elif nRDo>0.8 and nRDo<1.6 and nRUp>0.8 and nRUp<1.6:
                print("UpDown,",end="")

        nPL = qt.expect(nPhL, state)
        nPR = qt.expect(nPhR, state)
        if nPL+nPR<0.8:
            print("0")
        else:
            if nPL<1.8 and nPR<0.8:
                print("1,0")
            elif nPL > 0.8 and nPR>0.8:
                print("1,1")
            elif nPL<0.8 and nPR<1.8:
                print("0,1")

def trace(e_sum, e_diff, e_mag, theta, mag_asym,omega0): #Affichage des états propres + énergies associées
    fig, ax = plt.subplots()
    eigs = []

    cpsf = CPSF(e_sum,e_diff,e_mag,theta,mag_asym,omega0)
    H0 = cpsf.H0
    H1 = cpsf.H1
    [nrj, st] = H0.eigenstates()
    nrjs = []
    u=0
    pos_z = []
    scatter = [(-1,-1) for i in range(len(st))]

    for ii, s in enumerate(st):
        nn = qt.expect(ntot, st[ii])
        nnph = qt.expect(nPhTot, st[ii])
        if nn < 2.2 and nn>1.1 and nnph<1.5:
            u += 1
            st_label, pos = state_label_compact(ii,st[ii])
            if 1==1:
                pos_z.append((pos,nrj[ii]))
                scatter[ii] = (pos, nrj[ii])
                ax.scatter(pos, nrj[ii],marker='_', color='b', linewidths='10')
                ax.text(pos, nrj[ii]+0.0*(8-np.mod(u,16)),
                        st_label,
                        fontsize='14')

            state = st[ii]

    for ii, s in enumerate(st): #Affichage des couplages entre états du hamiltonien non perturbé
        for kk in range(ii, len(st)):
            t = st[kk]
            tmp = np.abs(s.overlap(H1*t))
            if scatter[ii]!=(-1,-1) and scatter[kk]!=(-1,-1) and ii!=kk and tmp>10**-8  and nrj[ii]>=200 and nrj[kk]>=200:
                xx,yy = scatter[ii]
                xx2,yy2 = scatter[kk]
                plt.plot([xx,xx2],[yy,yy2],linewidth=tmp*5)

    print("end")
    plt.show()
    #return scatter
def info_st(s,nrj):
    print("Informations sur l'état ")
    print("Energie : "+str(round(nrj,3)))
    print("Gauche : "+str(round(qt.expect(nL, s),2))+" Droite : "+str(round(qt.expect(nR, s),2)))
    print("Up : "+str(round(qt.expect(nUp, s),2))+ " Down : "+str(round(qt.expect(nDo, s),2)))
    print("Photons g. : "+str(round(qt.expect(aLeft.dag()*aLeft, s),2)))
    print("S_tot : "+str(round(qt.expect(S_tot,s),2))+" S_z : "+str(round(qt.expect(S_Z, s),2)))
    print("----------------------")
#Parameters

U = 250
Um = 20
teh =0.2
tee = 0.2
DeltaKKp = 0
e_sum = 200
e_mag, theta, mag_asym = -3.5, np.pi/2, 0.05
omega0 =  -2*e_mag*(1+mag_asym) #-2bl
t_lim = np.pi/(4*np.sqrt(2)*teh)

#Définition d'observables et d'opérateurs

 
psi0 = vac
Ppsi0 = psi0*psi0.dag()
down = (LDoK.dag()*LDoKp.dag())*vac
down /= down.norm()
Pdown = down*down.dag()



nl_t,nr_t,down_t, up_t, tot, photons,spins,sz,tt= [],[],[],[],[],[],[],[],[]
#Etape 1 
e_sum,e_diff = -Um,0 

"""
cpsf3 = CPSF(e_sum, e_diff, e_mag, theta, mag_asym,omega0)
H = cpsf3.H
times2 = np.linspace(0,t_lim, 1000)
"""
"""
result2 = qt.mesolve(H, psi0, times2, [], [])
for i,t in enumerate(times2):
    nl_t.append(qt.expect(nL, result2.states[i]))
    nr_t.append(qt.expect(nR,result2.states[i]))
    up_t.append(qt.expect(nUp, result2.states[i]))
    down_t.append(qt.expect(nDo,result2.states[i]))
    tot.append(qt.expect(ntot,result2.states[i]))
    spins.append(qt.expect(S_tot,result2.states[i]))

    photons.append(qt.expect(aLeft.dag()*aLeft, result2.states[i]))

    tt.append(t)
"""
#Etape 2

e_sum,e_diff  = 200,(U-Um+2*e_mag)
"""
cpsf = CPSF(e_sum, e_diff, e_mag, theta, mag_asym,10.5)
cpsf.diagonalize()
for e in range(len(cpsf.states)):
    if np.abs(qt.expect(S_tot,cpsf.states[e])) <10**-1:
        print("Debug : state n°"+str(e)+"@"+str(round(cpsf.energies[e], 4))+" GHz; nL/nR : "+str(round(qt.expect(nL, cpsf.states[e]),4))+" "+str(round(qt.expect(nR, cpsf.states[e]),4))+" photons : "+
            str(round(qt.expect(nPhL,cpsf.states[e]),4))+" , s_tot : "+str(round(qt.expect(S_tot, cpsf.states[e]),4))+ ", sz "+str(round(qt.expect(S_Z,cpsf.states[e]),4)))

"""
"""
cur_st = cpsf3.states[60]
print(info_st(cur_st, cpsf3.energies[60]))
"""
singlet = (LUpK.dag()*RDoKp.dag()-LDoKp.dag()*RUpK.dag())*vac+\
          (LUpKp.dag()*RDoK.dag()-LDoK.dag()*RUpKp.dag())*vac
triplet = (LUpK.dag()*RDoKp.dag()+LDoKp.dag()*RUpK.dag())*vac+\
          (LUpKp.dag()*RDoK.dag()+LDoK.dag()*RUpKp.dag())*vac
singlet = singlet/singlet.norm()
triplet = triplet/triplet.norm()

#print(info_st(triplet, 220.5))
e_1 = (LDoK.dag()*LDoKp.dag())*vac
e_2 = (RUpK.dag()*RUpKp.dag())*vac
e_3 = (LDoK.dag()*RDoKp.dag())*vac
e_3 = (LDoK.dag()*RUpKp.dag())*vac

#print(qt.expect(S_tot, singlet))
#print(qt.expect(S_tot, triplet))
#print(qt.expect(S_tot, e_1))
#print(qt.expect(S_tot, e_3))

times3 = np.linspace(0,1, 200)
if 1==1:
    cpsf4 = CPSF(e_sum, e_diff, e_mag, theta, mag_asym,7.35)
    result3 = qt.mesolve(cpsf4.H, triplet, times3, [np.sqrt(0.0001)*aLeft], []) #part avec le dernier état du précédent mesolve
    for i,t in enumerate(times3):
        print(qt.expect(cpsf4.H, result3.states[i]))
        nl_t.append(qt.expect(nL, result3.states[i]))
        nr_t.append(qt.expect(nR,result3.states[i]))
        up_t.append(qt.expect(nUp, result3.states[i]))
        down_t.append(qt.expect(nDo,result3.states[i]))
        tot.append(qt.expect(ntot,result3.states[i]))
        spins.append(qt.expect(S_tot,result3.states[i]))
        sz.append(qt.expect(S_Z, result3.states[i]))
        photons.append(qt.expect(aLeft.dag()*aLeft, result3.states[i]))
        tt.append(t+t_lim)

    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Expectation values')
    ax.plot(tt, nl_t,label="nL")
    ax.plot(tt, nr_t,label="nR")
    ax.plot(tt, up_t,label="nUp")
    ax.plot(tt, down_t,label="nDo")
    ax.plot(tt, photons,label="nb photons")
    ax.plot(tt, spins, label="Spin total")
    ax.plot(tt, sz, label="Spin z")
    ax.plot(tt, tot,label="nTot")

    ax.legend()
    plt.show() 

#trace(e_sum,e_diff,e_mag,theta,mag_asym, omega0)
#theta = 0
print("----- PARAMETRES -----")
print("e_sigma : "+str(e_sum))
print("e_diff : "+ str(e_diff))
print("omega0 : "+str(omega0))
print("---------------------")

if 1==1: #Mode debug, plot de tous les états propres du Hamiltonien dans la plage d'énergie pertinente
    start = 20
    end  = 45
    OMEGA_START  = 10.49
    OMEGA_END = 10.52
    omegas = np.linspace(OMEGA_START,OMEGA_END,50) #balayage en fréquence
    states = [[] for k in range(start,end)]         
    for omega in omegas:
        print("Current omega : "+str(omega))
        cpsf = CPSF(e_sum,e_diff,e_mag,theta,mag_asym, omega)
        cpsf.diagonalize()
        print("Energie singlet "+str(round(qt.expect(cpsf.H,singlet),4)))
        print("Energie triplet "+str(round(qt.expect(cpsf.H,triplet),4)))
        for e in range(start,end):
            states[e-start].append(cpsf.energies[e])  
            if np.abs(cpsf.energies[e] - 220.5)<5:
                print("Debug : state n°"+str(e)+"@"+str(round(cpsf.energies[e], 4))+" GHz; nL/nR : "+str(round(qt.expect(nL, cpsf.states[e]),4))+" "+str(round(qt.expect(nR, cpsf.states[e]),4))+" photons : "+
                    str(round(qt.expect(nPhL,cpsf.states[e]),4))+" , s_tot : "+str(round(qt.expect(S_tot, cpsf.states[e]),4))+ ", sz "+str(round(qt.expect(S_Z,cpsf.states[e]),4)))
    fig, ax = plt.subplots()
    ax.set_xlabel('omega')
    for e in range(start,end):
        ax.plot(omegas, states[e-start],label=str(e))
    ax.legend()
    plt.show()  

if 1==0: #Si on connaît déjà les deux états voulus
    st_1 = (LUpK.dag()*RDoKp.dag()-LDoKp.dag()*RUpK.dag()+LUpKp.dag()*RDoK.dag()-LDoK.dag()*RUpKp.dag())*vac
    st_1 = st_1/st_1.norm()
    st_2 = ((LDoK.dag()*RDoKp.dag()*aLeft.dag())+(LDoKp.dag()*RDoK.dag()*aLeft.dag()))*vac
    st_2 = st_2/st_2.norm()
    cpsf_1 = CPSF(e_sum,e_diff,e_mag,theta,mag_asym,omega0-0.2)
    test1 = find(cpsf_1.H.eigenstates()[1], st_1,0,0)
    test2 = find(cpsf_1.H.eigenstates()[1], st_2,1,0)
    cpsf_3 = CPSF(e_sum,e_diff,e_mag,theta,mag_asym,omega0+0.2)
    test3 = find(cpsf_3.H.eigenstates()[1], st_1,0,0)
    test4 = find(cpsf_3.H.eigenstates()[1], st_2,1,0)

    print(str(test1)+" "+str(test2)+" "+str(test3)+" "+str(test4))

    if test1!=test3:
        print("ATTENTION : Etats propres différents trouvés par la recherche")


    frequency_low = 10.4
    frequency_high = 10.6

    """
    for k in range(10): #Recherche dichotomique pour centrer le graphique
        M = (frequency_low+frequency_high)/2
        print("Bornes : "+str(frequency_low)+" "+str(frequency_high))
        c1 = CPSF(e_sum,e_diff,e_mag,theta,mag_asym,M)
        c1.diagonalize()
        e_1 = c1.energies[test1]
        e_2 = c1.energies[test2]
        if (e_1>e_2):
            frequency_low = (frequency_high+frequency_low)/2
        else:
            frequency_high = (frequency_high+frequency_low)/2
    """

    omegas = np.linspace(frequency_low,frequency_high,30)
    energies_1,energies_2 = [],[]
    for omega in omegas: 
        cpsf = CPSF(e_sum,e_diff,e_mag,theta,mag_asym,omega)
        cpsf.diagonalize()
        energies_1.append(cpsf.energies[test1])
        energies_2.append(cpsf.energies[test2])
    
    fig, ax = plt.subplots()
    ax.set_xlabel('omega')
    ax.scatter(omegas, energies_1,label = "Energies 1")
    ax.scatter(omegas, energies_2, label="Energies 2")
    ax.legend()
    plt.show()