'''Testing OBL'''

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from interpolation.splines import eval_linear
from interpolation import interp


# Pi<p<Pi+1
# Zj<z<Zj+1

# p,z---------p,z
# ¦ \        /¦
# ¦   \    /  ¦
# ¦    \ /    ¦
# ¦    p',z'  ¦
# ¦  /     \  ¦
# p,z---------p,z


# def RachfordRice(z1, z2):
#     z = np.array([z1, z2, 1-z1-z2])
#     K = np.array([6, 0.8, 0.07])
#     r = lambda v: np.sum(z*(K-1)/(v*(K-1)+1))
#     a = 1 / (1 - np.max(K))
#     b = 1 / (1 - np.min(K))
#     for i in range(1000):
#         v = 0.5 * (a + b)
#         if r(v) > 0:
#             a = v
#         else:
#             b = v
#         if np.abs(r(v)) < 1e-12:
#             break
#     x = z/(v*(K-1)+1)
#     y = K * x
#     return x, y

def RachfordRice_binary(z1):
    z = np.array([z1, 1-z1])
    K = np.array([3, 0.1])
    r = lambda v: np.sum(z*(K-1)/(v*(K-1)+1))
    a = 1 / (1 - np.max(K))
    b = 1 / (1 - np.min(K))
    for i in range(1000):
        v = 0.5 * (a + b)
        if r(v) > 0:
            a = v
        else:
            b = v
        if np.abs(r(v)) < 1e-12:
            break
    x = z/(v*(K-1)+1)
    y = K * x
    return x, y

f = lambda s: s**2 / (s**2 + 10.*(1-s)**2)
df_ds = lambda s: (f(s + 1e-8) - f(s)) / 1e-8 if s > 0 and s < 1 else 0


pref = 200
p0 = 50
pe = 500
list_len = 5
P = np.linspace(p0,pe,list_len)
Z = np.linspace(0,1,list_len)
p = 215
z = 0.45
rho = 1000
kr = 5
mu = 0.01
#print(scipy.ndimage.map_coordinates([P,Z],240))
for i in range(len(P)):
    if p == P[i]:
        p_index = i
        print('equal p')
        break
    if p < P[i]:
        indexP1 = i
        indexP = i-1
        Pi1 = P[i]
        Pi = P[i-1]
        p_i = (p - Pi) / (Pi1 - Pi) # percentage where the solution is located
        p_index = p_i + indexP
        print('p_i',p_i)
        print('p_index',p_index)
        break
for j in range(len(Z)):
    if z == Z[j]:
        z_index = j
        print('equal z')
        break
    if z < Z[j]:
        indexZ1 = j
        indexZ = j-1
        Zj1 = Z[j]
        Zj = Z[j-1]
        z_j = (z - Zj) / (Zj1 - Zj)
        z_index = z_j + indexZ
        print('z_j',z_j)
        print('z_index',z_index)
        break
#f_alpha = lambda alpha:
def alpha(p,z):
    return (1+1e-5*(p-pref))*(z*rho)
def beta(p,z):
    return z*(kr/mu)*rho
#F_alpha2 = np.zeros((len(P),len(Z)))
alpha_mat = np.zeros((len(P),len(Z)))
beta_mat = np.zeros((len(P),len(Z)))
#F_alpha = (1-Zj)*((1-Pi)*alpha(Pi,Zj)+Pi*alpha(Pi1,Zj))+Zj*((1-Pi)*alpha(Pi,Zj1)+Pi*alpha(Pi1,Zj1))
for i in range(len(P)):
    for j in range(len(Z)):
        alpha_mat[i, j] = alpha(P[i], Z[j])
        beta_mat[i, j] = beta(P[i], Z[j])

#for i in range(len(P)-1):
#    for j in range(len(Z)-1):
#        F_alpha2[i,j] = (1-Z[j])*((1-P[i])*alpha(P[i],Z[j])+P[i]*alpha(P[i+1],Z[j]))+Z[j]*((1-P[i])*alpha(P[i],Z[j+1])+P[i]*alpha(P[i+1],Z[j+1]))
alpha_interpol = scipy.ndimage.map_coordinates(alpha_mat,[[p_index,1],[z_index,1]],order=1)
beta_interpol = scipy.ndimage.map_coordinates(beta_mat,[[p_index,0],[z_index,0]],order=1)
#print(alpha_mat)
#print(beta_mat)
print('alpha with scipy map_coordinates method',alpha_interpol[0])
print(beta_interpol[0])
point = np.array([p_index,z_index])
#val = eval_linear(grid, values, point)  # float


def simulate_comp_impl(nb, Theta_ref, NT):
    z_inj = 0.99
    z = np.ones(nb) * 0.01
    z[0] = z_inj
    x, y = RachfordRice_binary(z[0])
    rhs = np.zeros(nb)
    jac = np.zeros([nb, nb])
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    for t in range(NT):
        zn = np.array(z, copy=True)
        for n in range(100):
            sl = (z[0] - y[0]) / (x[0] - y[0])
            if sl < 0 or sl > 1:
                Fl = z[0]
            else:
                Fl = x[0] * f(sl) + y[0] * (1 - f(sl))
            rhs[0] = 0
            jac[0, 0] = 1
            for i in range(1, nb):
                s = (z[i] - y[0]) / (x[0] - y[0])
                if s < 0 or s > 1:
                    F = z[i]
                else:
                    F = x[0] * f(s) + y[0] * (1 - f(s))
                # z[i] -= Theta * (F - Fl)
                rhs[i] = z[i] - zn[i] + Theta * (F - Fl)
                jac[i, i - 1] = - Theta * df_ds(sl)
                jac[i, i] = 1 + Theta * df_ds(s)
                Fl = F
                sl = s
            res = np.linalg.norm(rhs)
            if res < 1e-4:
                nit += n + 1
                break
            dz = np.linalg.solve(jac, -rhs)
            z += dz
        Theta = Theta_ref
    return z

nb = 100
Theta = 0.2
NT = 100

x = np.linspace(0, 1, nb)
#z1_plot, z2_plot = simulate_comp_expl(nb, Theta, NT)
#plt.plot(x, z1_plot, label='z1')
#plt.plot(x, z2_plot, label='z2')
#plt.ylabel('Saturation')
#plt.xlabel('x dimensionless')
#plt.legend()
# z2_plot = simulate_comp_impl(nb, Theta, NT)
# plt.plot(x, z2_plot, label='implicit')
# plt.ylabel('Saturation')
# plt.xlabel('x dimensionless')
# plt.legend()
# plt.show()

# a = np.arange(12.).reshape((4, 3))
# print(a)
# print(ndimage.map_coordinates(a, [[0.5, 2], [1, 1]], order=1))

from interpolation.splines import UCGrid, CGrid, nodes

# we interpolate function
#f = lambda x,y: np.sin(np.sqrt(x**2+y**2+0.00001))/np.sqrt(x**2+y**2+0.00001)
lower_grid = 0
upper_grid = 10
number_of_points_grid = 10
# uniform cartesian grid
#grid = UCGrid((-np.inf, np.inf, 10), (-np.inf, np.inf, 10))

# get grid points
#gp = nodes(grid)   # 100x2 matrix

# compute values on grid points
#values = f(gp[:,0], gp[:,1]).reshape((10,10))

from interpolation.splines import eval_linear
# interpolate at one point
point = np.array([p_index,z_index]) # 1d array
grid = UCGrid((0,list_len-1,list_len),(0,list_len-1,list_len))
values = np.array(alpha_mat)
val = eval_linear(grid, values, point)
#print(values)
print('alpha with eval method',val)