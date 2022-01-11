'''Testing OBL'''
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
from src.darts_interpolator import DartsInterpolator

def RachfordRice(*z):
    z_RR = list(z[0])
    #z = np.array([z,0.49,0.01])
    z2 = np.append(z,1-sum(z_RR))
    #K = np.array([3, 0.1])
    K = np.array([6, 0.8, 0.07])
    r = lambda v: np.sum(z2*(K-1)/(v*(K-1)+1))
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
    x = z2/(v*(K-1)+1)
    y = K * x
    return x, y

def beta_test(z, values):
    z_beta = z
    #beta_operator = np.zeros(len(z))
    mu_L = 1
    mu_g = 1
    rho_g = 1
    rho_L = 1
    n1 = 2
    n2 = 2
    x, y = RachfordRice(z)
    Ln = (z_beta[0] - y[0]) / (x[0] - y[0])  # Ln = (z[1] - y[1]) / (x[1] - y[1]) gives the same sat
    for i in range(len(z_beta)):
        #Ln = (z_beta[i] - y[i]) / (x[i] - y[i])
        if Ln <= 0:
            beta_operator = (z_beta[i] * (((1 - 0) ** n2) / mu_g)) / (((1 - 0) ** n2) / mu_g)
            #alpha_operator = (z_beta[i] * rho_g)
            #Ln = 0
            #x = np.zeros(len(z))
            #y = z
        elif Ln >= 1:
            beta_operator = (z_beta[i] * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
            #alpha_operator = (z_beta[i] * rho_L)
            #Ln = 1
            #x = z
            #y = np.zeros(len(z))
        else:
            beta_operator = (x[i]*((Ln**n1)/mu_L) + y[i]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
            #rho_t = (x[i] * rho_L + y[i] * rho_g) / (x[i] + y[i])
            #alpha_operator = z_beta[i] * rho_t
        values[i] = beta_operator
        #values[len(z_beta)+i] = alpha_operator
    #beta_operator = (x[1]*((Ln**n1)/mu_L) + y[1]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
    #return beta_operator
    return 0

def simulate_comp_impl(nb, Theta_ref, NT, z):
    comp = len(z)-1
    rhs = np.zeros(nb*comp)
    jac = np.zeros([nb*comp, nb*comp])  # the more compositions, the larger the jacobian
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    beta_interpol = DartsInterpolator(beta_test, axes_points=[1000, 1000], axes_min=[0, 0], axes_max=[1, 1], amount_of_int=2)
    for t in range(NT):
        zn = np.array(z, copy=True)
        for n in range(100):
            z_column = [row[0] for row in z][0:-1]
            beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives(z_column)
            beta_L = np.array(beta_L, copy=True)
            beta_L_deriv = np.array(beta_L_deriv, copy=True)
            beta_L_deriv = np.reshape(beta_L_deriv, (2,2))
            for j in range(comp):
                rhs[j] = 0
                for j2 in range(len(z)-1):
                    jac[j,j2] = 1
            for i in range(comp, nb*comp):
                if i % comp == 0:
                    u = int(i / comp)
                    z_column = [row[u] for row in z][0:-1]
                    beta, beta_deriv = beta_interpol.interpolate_point_with_derivatives(z_column)
                    beta = np.array(beta, copy=True)
                    beta_deriv = np.array(beta_deriv, copy=True)
                    beta_deriv = np.reshape(beta_deriv, (2, 2))
                    for j in range(comp):
                        rhs[i+j] = z[j][u] - zn[j][u] + Theta * (beta[j] - beta_L[j])
                        for j2 in range(comp):
                            jac[i+j, i-comp+j2] = - Theta * beta_L_deriv[j][j2]
                            jac[i+j, i+j2] = 1 + Theta * beta_deriv[j][j2]
                    beta_L = beta
                    beta_L_deriv = beta_deriv
            print(jac)
            res = np.linalg.norm(rhs)
            if res < 1e-4:
                nit += n + 1
                break
            if n == 99:
                print('newton itor problems')
            dz = np.linalg.solve(jac, -rhs)
            print('no singular matrix')
            for i in range(nb*comp):
                if i % comp == 0:
                    u = int(i / comp)
                    for j in range(comp):
                        z[j,u] += dz[i+j]
                #z[0] += dz#
        Theta = Theta_ref
    return z

nb = 3
Theta = 0.1
NT = 10
components = 3 # Change K in rachfordrice, z_inj and z
z_last_comp = np.zeros(nb)
z_inj = np.zeros(components)
z_inj[0] = 0.8 # injection first composition
z_inj[1] = 0.15
z = np.array(np.ones(nb)*0.1) #composition of the cells
z = np.append(z,np.array(np.ones(nb)*0.1))
z[0] = z_inj[0] # set first cell to injection
z[nb] = z_inj[1]
for i in range(nb):
    z_last_comp[i] = z[i] + z[nb+i]
z2 = np.append(z,np.array(np.ones(nb)-z_last_comp,dtype=float))
z3 = np.reshape(z2,[components,nb])
x = np.linspace(0, 1, nb)

z_plot = simulate_comp_impl(nb, Theta, NT, z3)
z1_plot = z_plot[0]
z2_plot = z_plot[1]
z3_plot = np.zeros(len(z1_plot))
for i in range(len(z1_plot)):
    z3_plot[i] = 1 - z1_plot[i]-z2_plot[i]
plt.plot(x, z1_plot, label='z1')
plt.plot(x, z2_plot, label='z2')
plt.plot(x, z3_plot, label='z3')
plt.ylabel('Saturation')
plt.xlabel('x dimensionless')
#plt.ylim(0,1)
plt.legend()
plt.show()
