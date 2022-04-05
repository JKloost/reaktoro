'''Testing OBL'''
import matplotlib.pyplot as plt
import numpy as np
from darts_interpolator import DartsInterpolator
import timeit


def comp_correction(z, min_z):
    n_correct = 0
    for i in range(nb):
        sum_z = 0
        z_correct = False
        for c in range(C):
            new_z = z[i][c]
            if new_z < min_z:
                new_z = min_z
                z_correct = True
            elif new_z > 1 - min_z:
                new_z = 1 - min_z
                z_correct = True
            sum_z += new_z  # sum previous z of the loop
        new_z = 1-sum_z  # Get z_final
        if new_z < min_z:
            new_z = min_z
            z_correct = True
        sum_z += new_z  # Total sum of all z's
        if z_correct:  # if correction is needed
            for c in range(C):
                new_z = z[i][c]
                new_z = max(min_z, new_z)
                new_z = min(1 - min_z, new_z)  # Check whether z is in between min_z and 1-min_z
                new_z = new_z / sum_z  # Rescale
                z[i][c] = new_z
            n_correct += 1
    if n_correct:
        print('composition correct applied in block: ' + str(n_correct))
    return z


def comp_out_of_bounds(vec_composition, min_z):
    # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
    temp_sum = 0
    count_corr = 0
    check_vec = np.zeros((len(vec_composition),))

    for ith_comp in range(len(vec_composition)):
        if vec_composition[ith_comp] < min_z:
            vec_composition[ith_comp] = min_z
            count_corr += 1
            check_vec[ith_comp] = 1
        elif vec_composition[ith_comp] > 1 - min_z:
            vec_composition[ith_comp] = 1 - min_z
            temp_sum += vec_composition[ith_comp]
        else:
            temp_sum += vec_composition[ith_comp]

    for ith_comp in range(len(vec_composition)):
        if check_vec[ith_comp] != 1:
            vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * min_z)
    return vec_composition


def RachfordRice(*z):
    z2 = list(z[0])
    #z2 = np.append(z, 1-sum(z_RR))
    #K = np.array([3, 0.1])
    K = np.array([6, 3, 0.07])
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
    z_total = np.append(z_beta, 1-sum(z_beta))
    if (z_total < 0).any():
        print('correction with: ' + str(z_total) + ' of sum: ' + str(sum(z_total)))
        z_total_copy = np.array(z_total)
        z_total = comp_out_of_bounds(z_total_copy, 1e-12)  # check last comp
        z_beta = z_total[:-1]
    x, y = RachfordRice(z_total)
    mu_L = 1
    mu_g = 0.01
    rho_g = 1
    rho_L = 1
    n1 = 2
    n2 = 2
    Ln = (z_beta[0] - y[0]) / (x[0] - y[0])  # Ln = (z[1] - y[1]) / (x[1] - y[1]) gives the same sat
    #Ln = 1 # single phase
    for i in range(len(z_beta)):
        #if Ln < 0 or Ln > 1:
        #    beta_operator = z_beta[i]
        if Ln <= 0:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (rho_g * z_beta[i] * (((1 - 0) ** n2) / mu_g)) / (((1 - 0) ** n2) / mu_g)
            rho_t = (1-0) * rho_g  # s * rho
            alpha_operator = z_beta[i] * rho_t
        elif Ln >= 1:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (rho_L * z_beta[i] * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
            rho_t = 1 * rho_L  # s * rho
            alpha_operator = z_beta[i] * rho_t
        else:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (rho_L * x[i]*((Ln**n1)/mu_L) + rho_g * y[i]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
            rho_t = Ln*rho_L + (1-Ln)*rho_g
            alpha_operator = z_beta[i] * rho_t
        values[i] = beta_operator
        values[i+len(z_beta)] = alpha_operator
        values[i+2*len(z_beta)] = z_comp
    #print(z, values)  # Use for debugging
    return 0

def simulate_comp_impl(nb, Theta_ref, NT, z):
    start = timeit.default_timer()
    C = len(z[0])
    C_interpol = C*3
    jac = np.zeros([nb * C, nb * C])  # the more compositions, the larger the jacobian
    z_comp_cells = np.zeros(NT)
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    min_z = 1e-12
    beta_interpol = DartsInterpolator(beta_test, axes_points=[1000] * C, axes_min=[min_z/10] * C, axes_max=[1-min_z/10] * C,
                                      amount_of_int=C_interpol)
    for t in range(NT):
        #print('timestep: ',t)
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
            for q in range(C*C):
               #beta_L_deriv2[q*C+q] += 1
               #beta_L_deriv2[q*C+q] += alpha_n_deriv[q*C+q]  # Alpha operator
               beta_L_deriv2[q] += alpha_n_deriv[q]  # Alpha operator
            beta_L_deriv2 = np.reshape(beta_L_deriv2, (C, C))
            jac[0:C, 0:C] = beta_L_deriv2
            for j in range(1, nb):
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
                for q in range(C*C):
                    #beta_deriv2[q*C+q] += alpha_deriv[q*C+q] # Alpha operator
                    #beta_deriv2[q*C-q] = alpha_deriv[q*C-q]
                    beta_deriv2[q] += alpha_deriv[q]  # Alpha operator
                    #beta_deriv2[q * C + q] += 1
                #print(beta_deriv2)
                beta_L_deriv2, beta_deriv2 = np.reshape(beta_L_deriv2, (C, C)), np.reshape(beta_deriv2, (C, C))
                for i in range(C):
                    #rhs = np.append(rhs, z[j][i] - zn[j][i] + Theta * (beta[i] - beta_L[i]))  # needs better way
                    rhs = np.append(rhs, alpha[i] - alpha_n[i] + Theta * (beta[i] - beta_L[i]))
                #print('L',beta_L_deriv2)
                jac[C*j:C*j+C, C*(j-1):C*(j-1)+C] = -beta_L_deriv2 # -theta db/dz
                jac[C*j:C*j+C, C*j:C*j+C] = beta_deriv2 # da/dz + theta db/dz
            #np.set_printoptions(precision=7,suppress=True)
            #print(jac)
            #print('rhs',rhs)
            #exit()
            res = np.linalg.norm(rhs)
            print('timestep: '+str(t)+' iteration: '+str(n))  # print newton itor
            if res < 1e-4:
                nit += n + 1
                break
            if n == 99:
                print('#####newton itor problems#####')
            dz = np.linalg.solve(jac, -rhs)
            z_dz = np.array(z).flatten('C')
            z_dz = [sum(t) for t in zip(dz, z_dz)]
            z = [z_dz[x:x+C] for x in range(0, len(z_dz), C)]
            # z3 = np.zeros(nb)
            z = comp_correction(z, min_z)

            # for i in range(nb):
            #     z3[i] = 1-sum(z[i])
            #     vec_comp = np.append(z[i],z3[i])
            #     z[i] = comp_out_of_bounds(min_z, vec_comp)
            #     z[i] = z[i][:C]
        # z1 = np.zeros(nb)
        # z2 = np.zeros(nb)
        # for i in range(nb):
        #     z1[i] = z[i][0]
        #     z2[i] = z[i][1]
        # z3_plot = np.zeros(nb)
        # for i in range(nb):
        #     z3_plot[i] = 1 - sum(z[i])
        # x = np.linspace(0, 1, nb)
        # plt.figure()
        # plt.plot(x, z1, label='z_c')
        # plt.plot(x, z2, label='z_c')
        # plt.plot(x, z3_plot, label='z_c')
        # plt.ylabel('Saturation')
        # plt.xlabel('x dimensionless')
        # plt.ylim(-0.05, 1)
        # plt.legend()
        # plt.savefig("comp" + str(t) + ".png")
        # plt.close()
        Theta = Theta_ref
    stop = timeit.default_timer()
    print('CPU = %5.3f sec, NT = %d' % ((stop - start), t))
    return z, z_comp_cells

nb = 100
Theta = 0.2
NT = 150
components = 3          # Also change K in rachfordrice (line 14-15) and line 102 and z_inj/z_ini
C = components - 1
#z_inj = [0.8]
#z_ini = [0.2]
# z_inj = [0.6, 0.1]
# z_ini = [0.1, 0.7]
z_inj = [0.98, 0.01]
z_ini = [0.01, 0.7]
# z_inj = [0.5,0.2,0.3]
# z_ini = [0.3,0.1,0.6]
x = np.linspace(0, 1, nb)
t = np.linspace(0, 1, NT)
z = [z_inj]
for i in range(nb-1):
    z.append(z_ini)
z_og = np.array(z, copy=True)

z_plot, z_comp = simulate_comp_impl(nb, Theta, NT, z_og)
#z_plot_example = simulate_comp_impl_example(nb, Theta, NT)
#z_plot_example_expl, z_plot_example_expl2,z_comp_expl = simulate_comp_expl(nb, Theta, NT)
#z1_plot = z_plot[0]
#z2_plot = z_plot[1]
z3_plot = np.zeros(nb)
for i in range(nb):
    z3_plot[i] = 1 - sum(z_plot[i])

#plt.plot(t,z_comp_expl,label='z_1_expl example at nb/3')
#plt.plot(t,z_comp,label='z_1 at nb/3')

plt.plot(x, z_plot,label='z_c')
plt.plot(x,z3_plot,label='z_c')
#plt.plot(x, z_plot_example,'--', label='implicit, example')
#plt.plot(x,z_plot_example_expl, '--', label='expl example')

plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.legend()
plt.show()
#print('final z', z_plot)