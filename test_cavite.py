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


sm = qt.sigmam()
sz = qt.sigmaz()
I = qt.identity(2)
orbital_number = 1
ops = {}
aLeft = I
III = qt.qeye(3)

for i_ops in range(2*orbital_number):
    ops[i_ops] = sm  #initialize
for i_ops in range(2*orbital_number):
    kid = 2*orbital_number-1-i_ops
    ksz = i_ops
    while kid > 0:
        ops[i_ops] = qt.tensor(ops[i_ops], I) # pad with I to the right
        kid -= 1
    while ksz > 0:
        ops[i_ops] = qt.tensor(-sz, ops[i_ops]) # pad with -sz to the left
        ksz -= 1
    ops[i_ops] = qt.tensor(ops[i_ops],III)  #pad 2 I for photons
for k in range(2*orbital_number-1):
    aLeft = qt.tensor(aLeft,I)

# 1st orbital
LUp = ops[0]
LDo = ops[1]

aLeft = qt.tensor(aLeft, qt.destroy(3))

II = I
zero = qt.tensor(qt.fock(2,1)) 
zero_photons = qt.tensor(qt.fock(3,0))
vac = zero
for i_ops in range(2*orbital_number):
    if i_ops<2*orbital_number-1:
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
class TestCavite:
    
    def __init__(self,e_mag,theta, omega):
        e_LUp = e_mag
        e_LDo = -e_mag

        gl=0.5

        nL = LUp.dag()*LUp+LDo.dag()*LDo
        omega0L = omega  

        Hcavite = gl*nL*(aLeft+aLeft.dag()) 
        Hphoton = omega0L*(aLeft.dag()*aLeft)

        Hchem = e_LUp*(LUp.dag()*LUp) +\
                e_LDo*(LDo.dag()*LDo)

        Htee = tee*np.sin(theta/2)*(LUp.dag()*LDo)
        Htee += Htee.dag()
     
        self.H0 = Hchem  + Hphoton
        self.H1 = Hcavite  + Htee
        self.H = self.H0 + self.H1

    def diagonalize(self):
        #if not hasattr(self,'energies'): #avoid diagonalizing mutlitple times
        Hamiltonian = self.H
        [nrj, st] = Hamiltonian.eigenstates()
        self.energies = nrj
        self.states = st
           
plt.close('all')

'''
teh and tee should vary with epsilon_sum
we set them to constant values
'''
#Parameters

tee = 4
e_mag, theta  = -0.5, np.pi/2
omega0 =  2*e_mag #-2bl

nl_t,nr_t,down_t, up_t, tot, photons,tt= [],[],[],[],[],[],[]
cpsf3 = TestCavite(e_mag, theta,omega0)
H = cpsf3.H
times2 = np.linspace(0,5, 1000)
result2 = qt.mesolve(H, LDo.dag()*vac, times2, [np.sqrt(0.1)*aLeft], [])
for i,t in enumerate(times2):
    nl_t.append(qt.expect(LUp.dag()*LUp, result2.states[i]))
    nr_t.append(qt.expect(LDo.dag()*LDo,result2.states[i]))
    tot.append(qt.expect(LUp.dag()*LUp+LDo.dag()*LDo,result2.states[i]))
    photons.append(qt.expect(aLeft.dag()*aLeft, result2.states[i]))
    tt.append(t)

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('Expectation values')
ax.plot(tt, nl_t,label="Up")
ax.plot(tt, nr_t,label="Down")
ax.plot(tt, photons,label="nb photons")
ax.legend()
plt.show() 

print("----- PARAMETRES -----")
print("omega0 : "+str(omega0))
print("---------------------")

start = 0
end  = 11
OMEGA_START  = 7.47
OMEGA_END = 7.5
omegas = np.linspace(OMEGA_START,OMEGA_END,1000) #balayage en fréquence
states = [[] for k in range(start,end)]         
for omega in omegas:
    print("Current omega : "+str(omega))
    cpsf = TestCavite(e_mag,theta, omega)
    cpsf.diagonalize()
    for e in range(start,end):
        states[e-start].append(cpsf.energies[e])  
fig, ax = plt.subplots()
ax.set_xlabel('omega')
for e in range(start,end):
    ax.plot(omegas, states[e-start],label=str(e))
ax.legend()
plt.show()  