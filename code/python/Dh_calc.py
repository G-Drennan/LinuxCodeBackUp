#self.theta = [0, R[0], R[1]] #, R[1]   
 #       self.d = [P[0], 0, 0] #P[1]  
  #      self.a = [0, self.L2, self.L3 + P[1]] #self.L2, self.L3 + P[1]] 
   #     self.alpha = [self.nnt, 0, self.nnt]   


import math
import numpy as np 

def Trans_Z(d):
    return np.array([[1,0,0,0],[0,1,0,0], [0,0,1,d], [0,0,0,1]])

def Trans_X(r):
    return np.array([[1,0,0,r],[0,1,0,0], [0,0,1,0], [0,0,0,1]]) 

def Rot_z(theta):
    Cos_theta = math.cos(theta)
    Sin_theta = math.sin(theta)
    return np.array([[Cos_theta,-Sin_theta,0,0],[Sin_theta,Cos_theta,0,0],[0,0,1,0],[0,0,0,1]])

def Rot_x(alpha):
    Cos_alpha = math.cos(alpha)
    Sin_alpha = math.sin(alpha)
    return np.array([[1,0,0,0],[0,Cos_alpha,-Sin_alpha,0],[0,Sin_alpha,Cos_alpha,0],[0,0,0,1]])

def n_1_A_n(theta, d, r, alpha):
    Tz = Trans_Z(d)
    Rz = Rot_z(theta) 
    Tx = Trans_X(r) 
    Rx = Rot_x(alpha) 
    return np.matmul(np.matmul(np.matmul(Rz, Tz), Tx), Rx) #np.dot() np.matmul() np.multiply() 

def Base_2_end_effector(matrix_of_all_dh_param):
    
    term = 0
    A = 0
    AT = np.identity(4)
    for i in matrix_of_all_dh_param:
        term = 0
        for x in i:
            term = term + 1 
            if term == 1:
                theta = x
            elif term == 2:
                d = x
            elif term == 3:
                alpha = x
            elif term == 4:
                r = x
        
        A = n_1_A_n(theta,d,r,alpha)
        AT = np.matmul(AT,A)    
    return AT   
                    #theta, d, alpha, r 
dh_params = np.array([[0,163,90,0],[0,0,0,-425],[0,0,0,-392],[0,134,90,0],[0,100,-90,0]])
print(Base_2_end_effector(dh_params))  