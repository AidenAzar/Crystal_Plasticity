#!/usr/bin/env python

''' Author
    ==========
      Mohammad Azarnoush (mazarnou@asu.edu)

    Description
    ===========
      Explicit point integration Crystal plastity using
      Busso (1990) flow rule modified by Luscher et al.
      There are three hardening models:
      1- Asaro and Needleman
      2- Bassani and Wu
      3- Luscher et al.
      User can also define crystal oreintation. 
      It is assumed that total deformation gradient, F,
      is known for each step.

    Date
    ==========
      28.12.2014
'''
from numpy import *
from  scipy.linalg import inv,det,logm,expm,block_diag,svd
import  matplotlib.pyplot as plt

class rotation():
  def __init__(self,A,n0,m0):
	self.A, self.n0 ,self.m0 = A, n0, m0
  def null(self):
 	A = vstack((self.A,[0,0,0]))
	u,s,vh = svd(A)
	return vh.T               
  def rot_c(self):
	R = self.null()
	c11,c12,c44 = 169e3, 122e3, 75.30e3
	C = zeros((3,3,3,3))
	C[0,0,0,0] = c11; C[1,1,1,1] = c11; C[2,2,2,2] = c11
        #
	C[1,2,1,2] = c44; C[1,2,2,1] = c44; C[2,1,1,2] = c44
	C[2,1,2,1] = c44
	#
	C[2,0,2,0] = c44; C[2,0,0,2] = c44; C[0,2,2,0] = c44
	C[0,2,0,2] = c44
	#
	C[0,1,0,1] = c44; C[0,1,1,0] = c44; C[1,0,0,1] = c44
	C[1,0,1,0] = c44
   	#
	C[0,0,1,1] = c12; C[1,1,0,0] = c12; C[0,0,2,2] = c12
	C[2,2,0,0] = c12; C[1,1,2,2] = c12; C[2,2,1,1] = c12
	a = zeros((3,3,3,3))
	for p in arange(3):
           for q in arange(3):
             for r in arange(3):
		for s in arange(3):
		  for i in arange(3):
	             for j in arange(3):
			for k in arange(3):
			  for l in arange(3):
		   	      a[p,q,r,s] += R[p,i]*R[q,j]*R[r,k]*R[s,l]*C[i,j,k,l] 
        C_new = zeros((6,6))
        C_new[0:,0] = [0.5*a[0,0,0,0],a[1,1,0,0],a[2,2,0,0],a[1,2,0,0],a[2,0,0,0],a[0,1,0,0]]
        C_new[1:,1] = [0.5*a[1,1,1,1],a[2,2,1,1],a[1,2,1,1],a[2,0,1,1],a[0,1,1,1]]
        C_new[2:,2] = [0.5*a[2,2,2,2],a[1,2,2,2],a[2,0,2,2],a[0,1,2,2]]
        C_new[3:,3] = [0.5*a[1,2,1,2],a[2,0,1,2],a[0,1,1,2]]
        C_new[4:,4] = [0.5*a[2,0,2,0],a[0,1,2,0]]
        C_new[5:,5] = [0.5*a[0,1,0,1]]
        C_new += C_new.T
	return C_new

  def rot_s(self):
	R = self.null()
	n = dot(self.n0, R.T)
	m = dot(self.m0, R.T)
	return n,m

class store():
    # to store all variables
  def __init__(self,step):    
    self.Sigma = zeros((6,step+1))
    self.E     = zeros((6,step+1))
    self.Sp    = zeros((12,step+1))
    self.tau   = zeros((12,step+1)) 
    self.gamma = zeros((12,step+1))
    self.gamma_dot = zeros((12,step+1))
  def abs_(self): 
    self.tau   = abs(self.tau)
    self.gamma = abs(self.gamma)
    return self

def main(args):
    n0 = array([[1,1,-1],[1,1,-1],[1,1,-1],[1,-1,1],[1,-1,1],[1,-1,1],[1,-1,-1],
		[1,-1,-1],[1,-1,-1],[1,1,1],[1,1,1],[1,1,1]],dtype=float)
    m0 = array([[1,0,1],[0,1,1],[1,-1,0],[1,1,0],[0,1,1],[1,0,-1],[1,0,1],[1,1,0],
		[0,1,-1],[1,0,-1],[0,-1,1],[1,-1,0]],dtype=float)     
    
    n0 = n0 / linalg.norm(n0[0,:])
    m0 = m0 / linalg.norm(m0[0,:])
    print 'Crystal oreintation:'
    print '1- <100>\n2- <110>\n3- <310>\n4- <580>\n5- other'
    prompt = int(raw_input('> '))
    if   prompt ==1: direction=[1,0,0]
    elif prompt ==2: direction=[1,1,0]
    elif prompt ==3: direction=[3,1,0]
    elif prompt ==4: direction=[5,8,0]
    else: 
       direction = raw_input('Enter crytal orientation: \n')
       direction = map(float, direction.split())
    ROT = rotation(direction,n0,m0)
    C = ROT.rot_c()
    n0,m0 = ROT.rot_s()
    
    # discritizing time
    dt ,t  = 1e-12, 10e-9
    step = trunc(t/dt)

    # define total F
    max_lambda = 1.1
    _lambda = linspace(1, max_lambda,step + 1)

    # Initialize all variables
    tau   = zeros((12,1))
    Sp    = zeros((12,1))
    Fp    = eye(3)
    gamma = zeros((12,1))
    s = store(step)
    
    # Choosing model
    print 'Which hardening model:'
    print '1- Asaro and Needleman hardening model '
    print '2- Bassani and Wu hardening model'
    print '3- Luscher et al. hardening model'    
    load_case = int(raw_input('> '))

    print 'Main code starts ...' 
    for ts in arange(1,step+1):
       gamma_dot = FlowRule(tau,Sp)        
       if   load_case == 1:  Hab = asaro(gamma)
       elif load_case == 2: Hab = bassani(gamma)
       else: Hab = luscher(gamma_dot,Sp)

       Sp_dot = dot(Hab, abs(gamma_dot))
       # update gamma and Sp
       Sp    +=  Sp_dot   * dt
       gamma += gamma_dot * dt
       # calculating Lp      
       Lp = zeros((3,3))
       for k in arange(12):
           Lp += gamma_dot[k] * outer(m0[k,:], n0[k,:])
       # updating Fp
       Fp = dot(expm(Lp*dt), Fp)
       F  = eye(3)
       F[0,0] = _lambda[ts]
       # finding Fe
       Fe = linalg.solve(Fp.T,F).T
       Ee = 0.5 * (dot(Fe.T,Fe)-eye(3)) 
       # Calculating Pk2 stress
       S_voigt = dot(C,to_voigt_strain(Ee))
       S = to_tensor(S_voigt)
       # push forward pk2 to cauchy stress
       Sigma_c = dot(dot(Fe,S),Fe.T) / det(Fe)
       n = dot(n0,inv(Fe))
       m = dot(m0,Fe.T)
       for j in arange(12):
           tau[j] = sum(Sigma_c * outer(m[j,:], n[j,:]))
       B = dot(F, F.T)
       e_log = -0.5 * logm(inv(B))
       # store all variables in object s
       s.E[:,ts]     = to_voigt_strain(e_log).T
       s.Sigma[:,ts] = to_voigt_stress(Sigma_c).T
       s.Sp[:,ts]    = Sp.T
       s.tau[:,ts]   = tau.T
       s.gamma[:,ts] = gamma.T
       s.gamma_dot[:,ts] = gamma_dot.T  
    print 'Plotting ...'
    t_total = linspace(0,t,step+1)
    plotting(s.abs_(),t_total)

# Busso flow rule (1990)
def FlowRule(tau,Sp):
   gamma_dot0 = 1e7
   E0  = 1e-18
   Sl  = 20
   m1  = 0.33
   m2  = 1.66
   kT  = 4.11e-21
   return gamma_dot0 * sign(tau) * exp(-E0/(kT)*\
        ((macaulay(1-(macaulay((abs(tau)-Sp)/Sl))**m1))**m2))
  
# Asaro and Needleman hardening model
def asaro(gamma):
   q, tau0  = 1.4, 16.0
   tau_s = 4.4 * tau0
   h0 = 8.25 * tau0
   hs = 0.48 * tau0
   Qab = block_diag(ones((3,3)),ones((3,3)),ones((3,3)),ones((3,3)))
   Qab[Qab==0] = q
   gamma_bar = sum(abs(gamma))
   h   = hs + (h0-hs)* cosh(gamma_bar*(h0-hs)/(tau_s-tau0))**-2.0
   return  h * Qab

# Bassani and Wu hardening model
def bassani(gamma):
  tau0,gamma0  = 1.0,1e-3
  tau_I  = 1.3  * tau0
  h0     = 90.0 * tau0
  hs_I   = 4.45 * tau0
  hs_III = 0.15 * tau0
  gamma0_III = 1.75 * tau0
  q,n,h = 0.0, 8.0, 8.0
  c,g,s = 8.0, 15.0, 20.0
  v     = [0.0]*11
  v[0]  = [0.0,c,c,s,g,h,n,g,g,h,s,g]
  v[1]  = [0.0,c,g,n,g,g,s,h,s,h,g]
  v[2]  = [0.0,h,g,s,g,h,s,g,g,n]
  v[3]  = [0.0,c,c,g,n,g,g,s,h]
  v[4]  = [0.0,c,s,g,h,g,h,s]
  v[5]  = [0.0,h,g,s,n,g,g]
  v[6]  = [0.0,c,c,h,g,s]
  v[7]  = [0.0,c,s,g,h]
  v[8]  = [0.0,g,n,g]
  v[9]  = [0.0,c,c]
  v[10] = [0.0,c]
  fab   = zeros((12,12))
  for i in arange(11):
     fab[i:,i] = v[i]  
  fab += fab.T
  hs  = hs_I + (hs_III-hs_I) * tanh(sum(abs(gamma))/gamma0_III)
  LHS = hs   + (h0-hs) * (cosh((h0-hs)/(tau_I-tau0) * abs(gamma)))**-2
  RHS = dot(fab,tanh(abs(gamma)/gamma0)) + 1
  x   = LHS * RHS
  hab = diag(x[:,0])   
  for i in arange(11):
    hab[i+1:,i] = q * hab[i,i]
  for k in arange(11,0,-1):
    hab[:k, k] = q * hab[k,k]
  return hab

def luscher(gamma_dot,Sp):
   h0,r,A, S0 = 200.0, 1.4, 1.5e-19,1
   kT,Ss_0,gamma_dot0 = 4.11e-21, 205.0, 1e7
   Ss  = Ss_0 * (abs(gamma_dot/gamma_dot0))**(kT/A)
   II  = h0 * r * ones((12,12))+ h0 * (1.0 - r) * eye(12)
   III = (Ss-Sp)/(Ss-S0)
   return dot(II, diag(III[:,0]))

def plotting(s,t):
   pos = [0.1,0.1,0.7,0.8]
   w,ft   = 4, 15
   box_pos = (1,0.5)
   # plotting normalized stress versus strain
   fig1 = plt.figure(1)
   sigma0 = 117
   s.Sigma = s.Sigma / sigma0
   ax1 = fig1.add_subplot(111)
   ax1.set_position(pos)
   pl1 = ax1.plot(s.E[0,:],s.Sigma[0,:],'-r',s.E[0,:],s.Sigma[1,:],'-b',\
                  s.E[0,:],s.Sigma[2,:],'-g',s.E[0,:],s.Sigma[3,:],'-k',\
                  s.E[0,:],s.Sigma[4,:],'-y',s.E[0,:],s.Sigma[5,:],'-m',lw=w)
   ax1.legend(pl1,[r'$\sigma_{xx}/\sigma_{0}$',r'$\sigma_{yy}/\sigma_{0}$',\
                   r'$\sigma_{zz}/\sigma_{0}$',r'$\sigma_{yz}/\sigma_{0}$',\
                   r'$\sigma_{xz}/\sigma_{0}$',r'$\sigma_{xy}/\sigma_{0}$'],
                   loc='center left',bbox_to_anchor= box_pos)
   plt.xlabel(r'$\bf{ln(\epsilon_{xx})}$',fontsize=ft)
   plt.ylabel(r'$\bf{\sigma/\sigma_0}$',fontsize=ft)
   plt.grid()
   # plotting Sp versus time
   fig2 = plt.figure(2)
   ax2  = fig2.add_subplot(111)
   ax2.set_position(pos)
   pl2  = ax2.plot(t,s.Sp[0,:],'-r',t,s.Sp[1,:],'-b',t,s.Sp[2,:],'-g',\
                   t,s.Sp[3,:],'-m',t,s.Sp[4,:],'-k',t,s.Sp[5,:],'-y',\
                   t,s.Sp[6,:],'--r',t,s.Sp[7,:],'--b',t,s.Sp[8,:],'-g',\
                   t,s.Sp[9,:],'--m',t,s.Sp[10,:],'--k',t,s.Sp[11,:],'--y',lw=w)
   ax2.legend(pl2,['B4','B5','B2','C1','C5','C3','D4','D1','D6','A3','A6','A2'],\
             loc='center left',bbox_to_anchor=box_pos)
   plt.xlabel('t (sec)',fontsize=ft)
   plt.ylabel(r'$\bf{S_{\rho}}$',fontsize=ft)
   plt.grid()
   # plotting gamma_dot versus time
   fig3 = plt.figure(3)
   ax3  = fig3.add_subplot(111)
   ax3.set_position(pos)
   pl3 = ax3.plot(t,s.gamma_dot[0,:],'-r',t,s.gamma_dot[1,:],'-b',t,s.gamma_dot[2,:],'-g',\
           t,s.gamma_dot[3,:],'-m',t,s.gamma_dot[4,:],'-k',t,s.gamma_dot[5,:],'-y',\
           t,s.gamma_dot[6,:],'--r',t,s.gamma_dot[7,:],'--b',t,s.gamma_dot[8,:],'-g',\
           t,s.gamma_dot[9,:],'--m',t,s.gamma_dot[10,:],'--k',t,s.gamma_dot[11,:],'--y',lw=w)
   ax3.legend(pl3,['B4','B5','B2','C1','C5','C3','D4','D1','D6','A3','A6','A2'],\
             loc='center left',bbox_to_anchor=box_pos)
   plt.xlabel('t (sec)',fontsize=ft)
   plt.ylabel(r'$\bf{\.{\gamma}}$',fontsize=ft)
   plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
   plt.grid()
   # plotting tau versus gamma
   fig4 = plt.figure(4)
   ax4  = fig4.add_subplot(111)
   ax4.set_position(pos)
   pl4  = ax4.plot(s.gamma[0,:],s.tau[0,:],'-r',s.gamma[1,:],s.tau[1,:],'-b',\
                   s.gamma[2,:],s.tau[2,:],'-g',s.gamma[3,:],s.tau[3,:],'-m',\
                   s.gamma[4,:],s.tau[4,:],'-k',s.gamma[5,:],s.tau[5,:],'-y',\
                   s.gamma[6,:],s.tau[6,:],'--r',s.gamma[7,:],s.tau[7,:],'--b',\
                   s.gamma[8,:],s.tau[8,:],'-g',s.gamma[9,:],s.tau[9,:],'--m',\
                   s.gamma[10,:],s.tau[10,:],'--k',s.gamma[11,:],s.tau[11,:],'--y',lw=w)
   ax4.legend(pl4,['B4','B5','B2','C1','C5','C3','D4','D1','D6','A3','A6','A2'],\
             loc='center left',bbox_to_anchor=box_pos)
   plt.xlabel(r'$\bf{\gamma}$',fontsize=12)
   plt.ylabel(r'$\bf{\tau^\alpha}$',fontsize=12)
   plt.grid()
   plt.show()
    
def macaulay(x):
   return 0.5 * (x + abs(x))

def to_voigt_stress(v):
   return array([[v[0,0]],[v[1,1]],[v[2,2]],[v[1,2]],[v[0,2]],[v[0,1]]])

def to_voigt_strain(v):
   return array([[v[0,0]],[v[1,1]],[v[2,2]],[2*v[1,2]],[2*v[0,2]],[2*v[0,1]]])

def to_tensor(v):
   return array([(v[0,0],v[5,0],v[4,0]),
                 (v[5,0],v[1,0],v[3,0]),
                 (v[4,0],v[3,0],v[2,0])])

if __name__=='__main__':
        import sys
	main(sys.argv)
