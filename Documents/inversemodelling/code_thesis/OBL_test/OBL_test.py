'''Testing OBL'''
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
from darts_interpolator import DartsInterpolator

def RachfordRice(z):
    z = np.array([z, 1-z])
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

def beta_test(z,values):
    mu_L = 1
    mu_g = 1
    rho_g = 1
    rho_L = 1
    x, y = RachfordRice(z[0])
    Ln = (z[0] - y[0]) / (x[0] - y[0])
    n1 = 2
    n2 = 2
    if Ln <= 0:
        #Ln = 0
        #x = [0]
        #y = [z]
        beta_operator = (z[0]*(((1-0)**n2)/mu_g)) / (((1-0)**n2)/mu_g)
        alpha_operator = (z[0] * rho_g)
    elif Ln >= 1:
        beta_operator = (z[0] * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
        alpha_operator = (z[0] * rho_L)
        #Ln = 1
        #x = [z]
        #y = [0]
        #beta_operator = z / mu_g
    else:
        beta_operator = (x[0]*((Ln**n1)/mu_L) + y[0]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
        rho_t = (x[0] * rho_L + y[0] * rho_g) / (x[0] + y[0])
        alpha_operator = z[0] * rho_t
        #alpha_operator = (x[0] * rho_L + y[0] * rho_g)
        #alpha_operator = z[0] * rho_t
    #if beta_operator > 1:
    #    print(beta_operator)
    values[0] = beta_operator
    values[1] = alpha_operator
    return 0

def simulate_comp_impl(nb, Theta_ref, NT, z):
    rhs = np.zeros(nb)
    jac = np.zeros([nb, nb]) # the more compositions, the larger the jacobian
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    beta_interpol = DartsInterpolator(beta_test, axes_points=[1000], axes_min=[0], axes_max=[1], amount_of_int=2)
    for t in range(NT):
        zn = np.array(z, copy=True)
        for n in range(100):
            beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives([z[0]])
            beta_L = np.array(beta_L, copy=True)
            beta_L_deriv = np.array(beta_L_deriv, copy=True)
            #alpha_n = beta_L[1]
            beta_L = beta_L[0]
            rhs[0] = 0
            jac[0,0] = 1
            for i in range(1,nb):
                beta, beta_deriv = beta_interpol.interpolate_point_with_derivatives([z[i]])
                beta = np.array(beta, copy=True)
                beta_deriv = np.array(beta_deriv, copy=True)
                alpha_n = beta_interpol.interpolate_point([zn[i]])
                alpha_n = np.array(alpha_n,copy=True)
                alpha = beta[1]
                beta = beta[0]
                rhs[i] = z[i] - zn[i] + Theta * (beta - beta_L)
                #rhs[i] = alpha - alpha_n[1] + Theta * (beta - beta_L) # alpha-alpha_(t-1)
                jac[i, i-1] = - Theta * beta_L_deriv[0]
                jac[i, i] = 1 + Theta * beta_deriv[0]
                #jac[i, i] = beta_deriv[1] + Theta * beta_deriv[0]
                #alpha_n = alpha
                beta_L = beta
                beta_L_deriv = beta_deriv
            print(jac)
            res = np.linalg.norm(rhs)
            if res < 1e-4:
                nit += n + 1
                break
            if n == 99:
                print('newton iterator problems')
            dz = np.linalg.solve(jac, -rhs)
            z += dz
            #z[1] -= dz
        Theta = Theta_ref
    return z

#def simulate_comp_expl_binary(nb, Theta_ref_expl, NT_expl, z_expl):
#    #x, y = RachfordRice_binary(z[0])
#    Theta_expl = Theta_ref_expl / 1000
#    for t in range(NT_expl):
#        beta_L_expl = beta_test(z_expl[0])
#        for i in range(1, nb):
#            beta_expl = beta_test(z_expl[i])
#            z_expl[i] -= Theta_expl * (beta_expl - beta_L_expl)
#            beta_L_expl = beta_expl
#        Theta_expl = Theta_ref_expl
#    return z_expl

nb = 3
Theta = 0.2
NT = 3
components = 2
z_inj = [0.60] # array of composition-1
z_org = np.array(np.ones(nb)*0.1) #composition of the cells
z_org[0] = z_inj[0]
# z2 = np.append(z,np.array(np.ones(nb)-z,dtype=float))
# z3 = np.reshape(z2,[components,nb])
# K = np.array([3,0.1])
x = np.linspace(0, 1, nb)
z_org2 = np.array(z_org, copy=True)
#z1_plot, z2_plot = simulate_comp_expl(nb, Theta, NT)
#plt.plot(x, z1_plot, label='z1')
#plt.plot(x, z2_plot, label='z2')
#plt.ylabel('Saturation')
#plt.xlabel('x dimensionless')
#plt.legend()
#z2_plot_expl = simulate_comp_expl_binary(nb,Theta,NT,z_org)
z2_plot = simulate_comp_impl(nb, Theta, NT, z_org2)

#print(z2_plot)
plt.plot(x, z2_plot, label='impl')
#plt.plot(x, z2_plot_expl, '--',label='expl')
plt.ylabel('Saturation')
plt.xlabel('x dimensionless')
#plt.ylim(0,1)
plt.legend()
plt.show()
print(z2_plot)