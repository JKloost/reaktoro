'''Testing OBL'''
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
# from interpolation.splines import UCGrid, CGrid, nodes
# from interpolation.splines import eval_linear
from darts_interpolator import DartsInterpolator
import timeit

def RachfordRice(*z):
    z_RR = list(z[0])
    z2 = np.append(z,1-sum(z_RR))
    #K = np.array([3, 0.1])
    K = np.array([6, 0.8, 0.07])
    #K = np.array([6,3,0.8,0.07])
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
    mu_L = 10
    mu_g = 1
    rho_g = 1
    rho_L = 1
    n1 = 2
    n2 = 2
    x, y = RachfordRice(z)
    Ln = (z_beta[0] - y[0]) / (x[0] - y[0])  # Ln = (z[1] - y[1]) / (x[1] - y[1]) gives the same sat
    for i in range(len(z_beta)):
        #if Ln < 0 or Ln > 1:
        #    beta_operator = z_beta[i]
        if Ln <= 0:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (z[i] * (((1 - 0) ** n2) / mu_g)) / (((1 - 0) ** n2) / mu_g)
            rho_t = (1-0) * rho_g # s * rho
            alpha_operator = z_beta[i] * rho_t
        elif Ln >= 1:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (z[i] * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
            rho_t = 1 * rho_L # s * rho
            alpha_operator = z_beta[i] * rho_t
        else:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (x[i]*((Ln**n1)/mu_L) + y[i]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
            #rho_t = (x[i] * rho_L + y[i] * rho_g) / (x[i] + y[i])
            #alpha_operator = z[i] * rho_t
            rho_t = Ln*rho_L + (1-Ln)*rho_g
            alpha_operator = z_beta[i] * rho_t
        values[i] = beta_operator
        values[i+len(z_beta)] = alpha_operator
        values[i+2*len(z_beta)] = z_comp
    return 0

def simulate_comp_impl(nb, Theta_ref, NT, z):
    start = timeit.default_timer()
    C = len(z[0])
    C_interpol = C*3
    jac = np.zeros([nb * C, nb * C])  # the more compositions, the larger the jacobian
    z_comp_cells = np.zeros(NT)
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    beta_interpol = DartsInterpolator(beta_test, axes_points=[1000] * C, axes_min=[0] * C, axes_max=[1] * C,
                                      amount_of_int=C_interpol)
    for t in range(NT):
        zn = np.array(z, copy=True, dtype=object)
        for n in range(100):
            rhs = np.zeros(C)
            beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives(z[0])
            beta_L = np.array(beta_L, copy=True)
            beta_L_deriv = np.array(beta_L_deriv, copy=True)
            half = int(len(beta_L_deriv)/3)
            beta_L_deriv2 = beta_L_deriv[:half]
            alpha_n_deriv = beta_L_deriv[half:2*half]
            beta_L_deriv2 = Theta * beta_L_deriv2
            for q in range(C):
                #beta_L_deriv2[q*C+q] += 1
                beta_L_deriv2[q*C+q] += alpha_n_deriv[q*C+q]  # Alpha operator
            beta_L_deriv2 = np.reshape(beta_L_deriv2, (C, C))
            jac[0:C, 0:C] = beta_L_deriv2
            for j in range(1,nb):
                beta, beta_deriv = beta_interpol.interpolate_point_with_derivatives(z[j])
                beta, beta_deriv = np.array(beta, copy=True), np.array(beta_deriv, copy=True)
                beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives(z[j-1])
                beta_L, beta_L_deriv = np.array(beta_L, copy=True), np.array(beta_L_deriv, copy=True)
                alpha_n = beta_interpol.interpolate_point(list(zn[j]))
                alpha_n = np.array(alpha_n, copy=True)

                C_half = int(len(beta) / 3)

                alpha_n = alpha_n[C_half:2*C_half]
                alpha = beta[C_half:2*C_half]
                beta = beta[:C_half]

                beta_L_deriv2, beta_deriv2 = beta_L_deriv[:half], beta_deriv[:half]
                alpha_deriv = beta_deriv[half:2*half]

                if j == int(nb/3):
                    z_comp = beta[-1]
                    z_comp_cells[t] = z_comp

                beta_L_deriv2, beta_deriv2 = Theta * beta_L_deriv2, Theta * beta_deriv2
                for q in range(C):
                    beta_deriv2[q * C + q] += alpha_deriv[q*C+q] # Alpha operator
                    #beta_deriv2[q * C + q] += 1
                beta_L_deriv2, beta_deriv2 = np.reshape(beta_L_deriv2, (C, C)), np.reshape(beta_deriv2, (C,C))

                for i in range(C):
                    #rhs = np.append(rhs, z[j][i] - zn[j][i] + Theta * (beta[i] - beta_L[i]))  # needs better way
                    rhs = np.append(rhs, alpha[i] - alpha_n[i] + Theta * (beta[i] - beta_L[i]))
                jac[C*j:C*j+C, C*(j-1):C*(j-1)+C] = beta_L_deriv2
                jac[C*j:C*j+C, C*j:C*j+C] = beta_deriv2
            #np.set_printoptions(precision=7,suppress=True)
            #print(jac)
            #print('rhs',rhs)
            #exit()
            res = np.linalg.norm(rhs)
            if res < 1e-4:
                nit += n + 1
                break
            if n == 99:
                print('newton itor problems')
            dz = np.linalg.solve(jac, -rhs)
            #print('dz',dz)
            #print('dz',dz)
            #print('z',z)
            z_dz = np.array(z).flatten('C')
            z_dz = [sum(t) for t in zip(dz, z_dz)]

            z = [z_dz[x:x+C] for x in range(0, len(z_dz), C)]  #C = 2

            #z = [[z_dz[x * C], z_dz[x * C + 1], z_dz[x*C+2]] for x in range(nb)] # C =3
            #z = [[z_dz[x * C]] for x in range(nb)] # C = 1

            #print('z',z)
            #print('final',z)
            #exit()
        Theta = Theta_ref
    stop = timeit.default_timer()
    print('CPU = %5.3f sec, NT = %d' % ((stop - start), t))
    return z,z_comp_cells

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

def simulate_comp_impl_example(nb, Theta_ref, NT):
    start = timeit.default_timer()
    z_inj = 0.9
    z = np.ones(nb) * 0.1

    z[0] = z_inj
    x, y = RachfordRice_binary(z[0])
    rhs = np.zeros(nb)
    jac = np.zeros([nb, nb])
    nit = 0  # Counter for non-linear iterations

    Theta = Theta_ref / 1000
    for t in range(NT):
        # print(rhs)
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

    stop = timeit.default_timer()

    print('CPU = %5.3f sec, NT = %d' % ((stop - start), t))
    return z

def RachfordRice_expl(z1, z2):

    z = np.array([z1, z2, 1-z1-z2])
    K = np.array([6, 0.8, 0.07])
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

def simulate_comp_expl(nb, Theta_ref, NT):
    start = timeit.default_timer()
    z_inj1 = 0.9
    z_inj2 = 0.1
    z1 = np.ones(nb) * 0.1
    z2 = np.ones(nb) * 0.5

    z1[0] = z_inj1
    z2[0] = z_inj2
    x, y = RachfordRice_expl(z1[0], z2[0])

    Theta = Theta_ref / 1000
    z_comp_expl = np.zeros(NT)
    for t in range(NT):
        x, y = RachfordRice_expl(z1[0], z2[0])
        s = (z1[0] - y[0]) / (x[0] - y[0])
        # s = (z2[0] - y[1]) / (x[1] - y[1]) This is same saturation

        if s < 0 or s > 1:
            Fl1 = z1[0]
            Fl2 = z2[0]
        else:
            Fl1 = x[0] * f(s) + y[0] * (1 - f(s))
            Fl2 = x[1] * f(s) + y[1] * (1 - f(s))

        for i in range(1, nb):
            x, y = RachfordRice_expl(z1[i], z2[i])
            s = (z1[i] - y[0]) / (x[0] - y[0])
            if i == int(nb/3):
                z_comp_expl[t] = s*x[0]+(1-s)*y[0]
            if s < 0 or s > 1:
                F1 = z1[i]
                F2 = z2[i]
            else:
                F1 = x[0] * f(s) + y[0] * (1 - f(s))
                F2 = x[1] * f(s) + y[1] * (1 - f(s))
            z1[i] -= Theta * (F1 - Fl1)
            z2[i] -= Theta * (F2 - Fl2)
            Fl1 = F1
            Fl2 = F2
        Theta = Theta_ref

    stop = timeit.default_timer()

    print('CPU = %5.3f sec, NT = %d' % ((stop - start), t))

    return z1, z2,z_comp_expl

nb = 100
Theta = 0.2
NT = 100
components = 3          # Also change K in rachfordrice (line 14-15) and line 102 and z_inj/z_ini
C = components - 1
#z_inj = [0.9]
#z_ini = [0.1]
z_inj = [0.9, 0.1]
z_ini = [0.1, 0.5]
#z_inj = [0.4,0.3,0.3]
#z_ini = [0.3,0.1,0.2]
x = np.linspace(0, 1, nb)
t = np.linspace(0, 1, NT)
z = [z_inj]
for i in range(nb-1):
    z.append(z_ini)
z_og = np.array(z, copy=True)

z_plot, z_comp = simulate_comp_impl(nb, Theta, NT, z_og)
#z_plot_example = simulate_comp_impl_example(nb, Theta, NT)
z_plot_example_expl, z_plot_example_expl2,z_comp_expl = simulate_comp_expl(nb, Theta, NT)
#z1_plot = z_plot[0]
#z2_plot = z_plot[1]
z3_plot = np.zeros(nb)
for i in range(nb):
    z3_plot[i] = 1 - sum(z_plot[i])

#plt.plot(t,z_comp_expl,label='z_1_expl example at nb/3')
#plt.plot(t,z_comp,label='z_1 at nb/3')

plt.plot(x, z_plot,label='z, own code')
#plt.plot(x, z_plot_example,'--', label='implicit, example')
plt.plot(x,z_plot_example_expl, '--', label='expl example')

plt.ylabel('Saturation')
plt.xlabel('x dimensionless')
plt.legend()
plt.show()
#print('final z', z_plot)