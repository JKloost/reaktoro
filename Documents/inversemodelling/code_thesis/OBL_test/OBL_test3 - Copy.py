'''Testing OBL'''
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
# from interpolation.splines import UCGrid, CGrid, nodes
# from interpolation.splines import eval_linear
from darts_interpolator import DartsInterpolator

def RachfordRice(*z):
    z_RR = list(z[0])
    z2 = np.append(z,1-sum(z_RR))
    K = np.array([3, 0.1])
    #K = np.array([6, 0.8, 0.07])
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
        if Ln < 0 or Ln > 1:
            beta_operator = z_beta[i]
        else:
            beta_operator = (x[i]*((Ln**n1)/mu_L) + y[i]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)

        values[i] = beta_operator
    return 0

def simulate_comp_impl(nb, Theta_ref, NT, z):
    C = int(len(z) / nb)
    z_column = np.zeros(C)
    rhs = np.zeros(nb * C)
    jac = np.zeros([nb * C, nb * C])  # the more compositions, the larger the jacobian
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    beta_interpol = DartsInterpolator(beta_test, axes_points=[1000] * C, axes_min=[0] * C, axes_max=[1] * C,
                                      amount_of_int=C)
    for t in range(NT):
        zn = np.array(z, copy=True)
        for n in range(100):
            for i in range(C):
                z_column[i] = z[i*(nb)]
            beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives(z_column)
            beta_L = np.array(beta_L, copy=True)
            beta_L_deriv = np.array(beta_L_deriv, copy=True)
            print('beta_L',beta_L_deriv)
            beta_L_deriv = np.reshape(beta_L_deriv, (C, C))
            for j in range(C):
                rhs[j] = 0
                for j2 in range(C):
                    jac[j, j2] = 1
            for i in range(C, nb*C):
                if i % C == 0:
                    u = int(i / C)
                    #z_column = [row[u] for row in z][0:-1]
                    for q in range(C):
                        z_column[q] = z[u+q*nb] #[z[u], z[C+u+1]]
                    beta, beta_deriv = beta_interpol.interpolate_point_with_derivatives(z_column)
                    beta = np.array(beta, copy=True)
                    beta_deriv = np.array(beta_deriv, copy=True)
                    print('beta',beta_deriv)                            # returns deriv [1,0,0,1]
                    beta_deriv = np.reshape(beta_deriv, (C, C))
                    for j in range(C):
                        rhs[i+j] = z[(j+1)*u] - zn[(j+1)*u] + Theta * (beta[j] - beta_L[j])
                        for j2 in range(C):
                            jac[i+j, i-C+j2] = - Theta * beta_L_deriv[j][j2]
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
            #print('no singular matrix')
            for i in range(nb*C):
                if i % C == 0:
                    u = int(i / C)
                    for j in range(C):
                        z[(j+1)*u] += dz[i+j]
                #z[0] += dz#
        Theta = Theta_ref
    return z

nb = 3
Theta = 0.2
NT = 3
components = 2      # Also change K in rachfordrice (line 14-15)
C = components - 1
z_inj = [0.6, 0.3]
z_ini = [0.1, 0.2]
z = np.zeros(nb * C)
for i in range(C):
    z[nb*i:nb*(i+1)] = z_ini[i]
    z[i * nb] = z_inj[i]
x = np.linspace(0, 1, nb)

z_plot = simulate_comp_impl(nb, Theta, NT, z)
#z1_plot = z_plot[0]
#z2_plot = z_plot[1]
#z3_plot = np.zeros(len(z1_plot))
#for i in range(len(z1_plot)):
#    z3_plot[i] = 1 - z1_plot[i]-z2_plot[i]
plt.plot(x, z_plot, label='z1')
#plt.plot(x, z2_plot, label='z2')
#plt.plot(x, z3_plot, label='z3')
plt.ylabel('Saturation')
plt.xlabel('x dimensionless')
#plt.ylim(0,1)
plt.legend()
plt.show()
print('final z', z_plot)