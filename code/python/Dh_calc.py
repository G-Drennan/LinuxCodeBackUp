import math

def Trans_Z(d)
    return [[1,0,0,0],[0,1,00], [0,0,1,d], [0,0,0,1]]

def Trans_X(r)
    return [[1,0,0,r],[0,1,00], [0,0,1,0], [0,0,0,1]]

def Rot_z(theta)
    Cos_theta = math.cos(theta)
    Sin_theta = math.sin(theta)
    return [[Cos_theta,-Sin_theta,0,0],[Sin_theta,Cos_theta,0,0],[0,0,1,0],[0,0,0,1]]

def Rot_x(alpha)
    Cos_alpha = math.cos(alpha)
    Sin_alpha = math.sin(alpha)
    return [[1,0,0,0],[0,Cos_alpha,-Sin_alpha,0],[0,Sin_alpha,Cos_alpha,0],[0,0,0,1]]

def n_1_A_n(theta, d, r, alpha)
    Tz = Trans_Z(d)
    Rz = Rot_z(theta)
    Tx = Trans_X(r)
    Rx = Rot_x(alpha) 
    return Rz*Tz*Tx*Rx