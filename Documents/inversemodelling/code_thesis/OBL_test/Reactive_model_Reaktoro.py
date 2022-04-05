'''One phase solver'''
import numpy as np
from darts_interpolator import DartsInterpolator
import timeit
import matplotlib.pyplot as plt
from reaktoro import *

'''Components to elements'''
'''We switch the components, such as CaCO3 to elements (e.g. CO3-- and Ca++), the smallest we can make the chemical
components. This way, we transport the elements'''

# TODO: get density working


E = np.array([[1,0,0,0,0],      # elimination matrix, to transform comp to elem
              [0,1,0,0,0],
              [0,0,1,0,1],
              [0,0,0,1,1]])


def comp_correction(z, min_z):
    n_correct = 0
    for i in range(nb):
        sum_z = 0
        z_correct = False
        C = len(z[0])
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

def comp_to_element(z_c, E):
    z_e = np.zeros(E.shape[0])
    for i in range(E.shape[0]):
        z_e[i] = np.divide(np.sum(np.multiply(E[i], z_c)),np.sum(np.multiply(E, z_c))) # ze e_i z - Ez
    return z_e


class Variables():
    def __init__(self):
        self.p = 1e7    # in pascal
        self.T = 320    # in kelvin
        number_cells = nb

class Model(Variables):
    def __init__(self):
        super().__init__()
        self.editor = ChemicalEditor(Database('supcrt98.xml'))              # Database that Reaktoro uses
        self.editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")  # Aqueous phase with elem
        self.system = ChemicalSystem(self.editor)       # Set the system
        self.solver = EquilibriumSolver(self.system)    # solves the system
        self.reactions = ReactionSystem(self.editor)
        self.reaction = ReactionEquation('Ca++ + CO3-- = CaCO3(aq)')
        # print(self.reaction.numSpecies())
        self.states = []
        # Two version can be coded with the same solution. One uses the EquilibriumProblem function,
        # Other will use the state function. Both need to output a state in order to be solved by EquilibriumSolver

    def addingproblem(self, temp, pres, z_e_RR):
        mole_element = np.zeros(len(z_e_RR))
        Mtotal = sum(z_e_RR)
        for i in range(len(z_e_RR)):
            mole_element[i] = z_e_RR[i] #*Mtotal
        #print('mole element',mole_element)
        self.problem = EquilibriumProblem(self.system)
        self.problem.setTemperature(temp, 'kelvin')
        self.problem.setPressure(pres, 'pascal')
        #self.problem.add('H2O',1,'kg')
        self.problem.add('H2O', mole_element[0], 'mol')
        self.problem.add('CO2', mole_element[1], 'mol')
        self.problem.add('Ca++', mole_element[2], 'mol')
        self.problem.add('CO3--', mole_element[3], 'mol')
        self.state = equilibrate(self.problem)  # Equilibrate the problem in order to write to state
        #print(self.state)
        self.states.append(self.state)

    def addingstates(self, temp, pres, z_e_RR):
        mole_element = np.zeros(len(z_e_RR))
        Mtotal = sum(z_e_RR)
        for i in range(len(z_e_RR)):
            mole_element[i] = z_e_RR[i]# * Mtotal
        self.state = ChemicalState(self.system)
        self.state.setTemperature(temp, 'kelvin')
        self.state.setPressure(pres, 'pascal')
        self.state.setSpeciesAmount('H2O(l)', mole_element[0], 'mol')
        self.state.setSpeciesAmount('CO2(aq)', mole_element[1], 'mol')
        self.state.setSpeciesAmount('Ca++', mole_element[2], 'mol')
        self.state.setSpeciesAmount('CO3--', mole_element[3], 'mol')
        self.states.append(self.state)

    def output(self):
        H2O_amount = [state.speciesAmount("H2O(l)") for state in self.states]
        CO2_amount = [state.speciesAmount("CO2(aq)") for state in self.states]
        CaCO3_amount = [state.speciesAmount("CaCO3(aq)") for state in self.states]
        Ca_amount = [state.speciesAmount("Ca++") for state in self.states]
        CO3_amount = [state.speciesAmount("CO3--") for state in self.states]
        mass = [ChemicalQuantity(state).value("phaseMass(Aqueous)") for state in self.states]
        volume = [ChemicalQuantity(state).value("phaseVolume(Aqueous)") for state in self.states]
        mole_total = [ChemicalQuantity(state).value("phaseAmount(Aqueous)") for state in self.states]
        density = np.zeros(len(mass))
        z_h2O = np.zeros(len(mass))
        z_co2 = np.zeros(len(mass))
        z_ca = np.zeros(len(mass))
        z_co3 = np.zeros(len(mass))
        z_caco3 = np.zeros(len(mass))
        for i in range(len(mass)):
            z_h2O[i] = H2O_amount[i]/mole_total[i] # zc = Mc/Mtot, (M = mol/L)
            z_co2[i] = CO2_amount[i]/mole_total[i]
            z_ca[i] = Ca_amount[i]/mole_total[i]
            z_co3[i] = CO3_amount[i]/mole_total[i]
            z_caco3[i] = CaCO3_amount[i]/mole_total[i]
        for i in range(len(mass)):
            density[i] = mass[i] / abs(volume[i])
        #z_c = [z_h2O[0],z_co2[0],z_ca[0],z_co3[0],z_caco3[0]]
        mole_c = [H2O_amount[0], CO2_amount[0], Ca_amount[0], CO3_amount[0], CaCO3_amount[0]]
        return mole_c, density

    def output2(self):
        H2O_amount = [state.speciesAmount("H2O(l)") for state in self.states]
        CO2_amount = [state.speciesAmount("CO2(aq)") for state in self.states]
        CaCO3_amount = [state.speciesAmount("CaCO3(aq)") for state in self.states]
        Ca_amount = [state.speciesAmount("Ca++") for state in self.states]
        CO3_amount = [state.speciesAmount("CO3--") for state in self.states]
        mass = [ChemicalQuantity(state).value("phaseMass(Aqueous)") for state in self.states]
        volume = [ChemicalQuantity(state).value("phaseVolume(Aqueous)") for state in self.states]
        mole_total = [ChemicalQuantity(state).value("phaseAmount(Aqueous)") for state in self.states]
        density = np.zeros(len(mass))
        z_h2O = np.zeros(len(mass))
        z_co2 = np.zeros(len(mass))
        z_ca = np.zeros(len(mass))
        z_co3 = np.zeros(len(mass))
        z_caco3 = np.zeros(len(mass))
        for i in range(len(mass)):
            z_h2O[i] = H2O_amount[i]/mole_total[i] # zc = Mc/Mtot, (M = mol/L)
            z_co2[i] = CO2_amount[i]/mole_total[i]
            z_ca[i] = Ca_amount[i]/mole_total[i]
            z_co3[i] = CO3_amount[i]/mole_total[i]
            z_caco3[i] = CaCO3_amount[i]/mole_total[i]
        for i in range(len(mass)):
            density[i] = mass[i] / volume[i]
        z_c = [z_h2O, z_co2, z_ca, z_co3, z_caco3]
        return z_c, density

def RachfordRice(*z):
    change_K_flag = False  # Set false for no change, true for change in K-value
    z_list = list(z[0])
    z2 = z_list[:-1]
    ionicRR = z_list[-1]            # Increase in ionic strength means lower activity coefficient gamma
    K = np.array([6, 3, 0.8, 0.07])  # K = y/x  (vapor/liquid)
    if change_K_flag:
        for x in range(len(z2)):
            K[x] = K[x] - 0.01 * ionicRR.val
            if K[x] <= 0:
                K[x] = 1e-5
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


def Gas_Reaktoro(n_vap, T, p):
    editor = ChemicalEditor(Database('supcrt98.xml'))  # Database that Reaktoro uses
    editor.addGaseousPhase("H2O(g) CO2(g)")  # Gaseous phase with elem
    system = ChemicalSystem(editor)  # Set the system
    state = ChemicalState(system)
    state.setTemperature(T, 'kelvin')
    state.setPressure(p, 'pascal')
    state.setSpeciesAmount('H2O(g)', n_vap[0], 'mol')
    state.setSpeciesAmount('CO2(g)', n_vap[1], 'mol')
    mass_gas = ChemicalQuantity(state).value("phaseMass(Gaseous)")
    volume_gas = ChemicalQuantity(state).value("phaseVolume(Gaseous)")
    rho_vap = mass_gas / volume_gas
    return rho_vap

def beta_test(z_beta, values):
    z_total = np.append(z_beta, 1 - sum(z_beta))
    if (z_total < 0).any():
        print('correction with: ' + str(z_total)+' of sum: ' + str(sum(z_total)))
        z_total_copy = np.array(z_total)
        z_total = comp_out_of_bounds(z_total_copy, 1e-12)  # check last comp
        z_beta = z_total[:-1]
    mu_L = 1
    mu_g = 0.01
    rho_liq = 1200
    rho_vap = 800
    n1 = 2
    n2 = 2
    #m_ionic = Model()
    #m_ionic.addingproblem(m_ionic.T, m_ionic.p, z_beta)
    #properties = m_ionic.states[0].properties()
    #ionicfunc = ChemicalProperty.ionicStrength(m_ionic.system)
    #ionicstrength = ionicfunc(properties)
    ionicstrength = 1
    z_RR = np.append(z_beta, ionicstrength)
    #print('z',z_total)
    x, y = RachfordRice(z_RR)
    #print('x',x)
    # number of moles of that component in that phase divided by the total number of moles of all comp in that phase
    n_liq, n_vap = np.zeros(len(x)), np.zeros(len(x))      # mole in phase
    liqfrac = ((-z_beta[0] + y[0]) / (-x[0] + y[0]))  # * sum(z_beta)
    vapfrac = ((-z_beta[0] + x[0]) / (x[0] - y[0]))  # * sum(z_beta)
    nty = ((z_beta[0] - x[0]) / (y[0] - x[0]))
    ntx = ((z_beta[0] - y[0]) / (x[0] - y[0]))
    for i in range(len(x)):
        n_liq[i] = x[i] * ntx
        n_vap[i] = y[i] * nty
        if i > 10:
            n_liq[i] += n_vap[i]
    # print('ntx', ntx)
    # print('nty', nty)
    # print(sum(n_liq))
    # print('nliq', n_liq)
    # print('nvap', n_vap)
    m = Model()
    m.addingproblem(m.T, m.p, n_liq)
    amount_elements = m.states[0].elementAmounts()  # element amounts in alphabet order: C, Ca, H, O
    m.solver.solve(m.states[0], m.T, m.p, amount_elements)
    mole_c_operator, rho_l = m.output()  # this outputs comp h20, co2, ca, co3, caco3
    del m
    #rho_vap = Gas_Reaktoro(n_vap, m.T, m.p)
    rho_t = liqfrac * rho_liq + vapfrac * rho_vap
    for i in range(len(x)):
        mole_c_operator[i] += n_vap[i]

    z_c_operator = [float(i) / sum(mole_c_operator) for i in mole_c_operator]  # normalise the liq phase and vap phase
    z_e_operator = comp_to_element(z_c_operator, E)  # change to elem
    for i in range(len(z_e_operator)):
        if liqfrac <= 0:
            beta_operator = (rho_vap * z_e_operator[i] * (((1 - 0) ** n2) / mu_g)) / (((1 - 0) ** n2) / mu_g)
            alpha_operator = z_e_operator[i] * rho_t
        elif liqfrac >= 1:
            beta_operator = (rho_liq * z_e_operator[i] * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
            alpha_operator = z_e_operator[i] * rho_t
        else:
            beta_operator = (rho_liq * x[i]*((liqfrac**n1)/mu_L) + rho_vap * y[i]*(((1-liqfrac)**n2)/mu_g)) / \
                            ((liqfrac**n1)/mu_L + ((1-liqfrac)**n2)/mu_g)
            alpha_operator = z_e_operator[i] * rho_t
        values[i] = beta_operator
        values[i+len(z_e_operator)] = alpha_operator
    return 0


def simulate_comp_impl(nb, Theta_ref, NT, z):
    min_z = 1e-12
    density_flag = True
    start = timeit.default_timer()
    # Need to get the z of all comp
    interpolators = 2
    C = len(z[0])
    C_interpol = C*interpolators
    jac = np.zeros([nb * C, nb * C])  # the more compositions, the larger the jacobian
    z_comp_cells = np.zeros(NT)
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    beta_interpol = DartsInterpolator(beta_test, axes_points=[100] * C, axes_min=[0] * C, axes_max=[1] * C,
                                      amount_of_int=C_interpol)
    for t in range(NT):
        zn = np.array(z, copy=True, dtype=object)
        for n in range(100):
            rhs = np.zeros(C)
            beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives(z[0])
            beta_L = np.array(beta_L, copy=True)
            beta_L_deriv = np.array(beta_L_deriv, copy=True)
            half = int(len(beta_L_deriv)/interpolators)
            beta_L_deriv2 = beta_L_deriv[:half]
            alpha_n_deriv = beta_L_deriv[half:2*half]
            beta_L_deriv2 = Theta * beta_L_deriv2
            if density_flag:
                for q in range(C*C):
                    beta_L_deriv2[q] += alpha_n_deriv[q]
            else:
                for q in range(C):
                    beta_L_deriv2[q*C+q] += 1
            beta_L_deriv2 = np.reshape(beta_L_deriv2, (C, C))
            jac[0:C, 0:C] = beta_L_deriv2
            for j in range(1, nb):
                beta, beta_deriv = beta_interpol.interpolate_point_with_derivatives(z[j])
                beta, beta_deriv = np.array(beta, copy=True), np.array(beta_deriv, copy=True)
                beta_L, beta_L_deriv = beta_interpol.interpolate_point_with_derivatives(z[j-1])
                beta_L, beta_L_deriv = np.array(beta_L, copy=True), np.array(beta_L_deriv, copy=True)
                alpha_n = beta_interpol.interpolate_point(list(zn[j]))
                alpha_n = np.array(alpha_n, copy=True)

                C_half = int(len(beta) / interpolators)

                alpha_n = alpha_n[C_half:2*C_half]
                alpha = beta[C_half:2*C_half]
                beta = beta[:C_half]

                beta_L_deriv2, beta_deriv2 = beta_L_deriv[:half], beta_deriv[:half]
                alpha_deriv = beta_deriv[half:2*half]

                beta_L_deriv2, beta_deriv2 = Theta * beta_L_deriv2, Theta * beta_deriv2
                if density_flag:
                    for q in range(C*C):
                        beta_deriv2[q] += alpha_deriv[q]
                else:
                    for q in range(C):
                        beta_deriv2[q*C+q] += 1

                for i in range(C):
                    if density_flag:
                        rhs = np.append(rhs, alpha[i] - alpha_n[i] + Theta * (beta[i] - beta_L[i]))
                    else:
                        rhs = np.append(rhs, z[j][i] - zn[j][i] + Theta * (beta[i] - beta_L[i]))  # needs better way

                beta_L_deriv2, beta_deriv2 = np.reshape(beta_L_deriv2, (C, C)), np.reshape(beta_deriv2, (C, C))
                jac[C*j:C*j+C, C*(j-1):C*(j-1)+C] = -beta_L_deriv2
                jac[C*j:C*j+C, C*j:C*j+C] = beta_deriv2
            res = np.linalg.norm(rhs)
            print('time: ' + str(t) + ' iteration: ' + str(n))  # print newton iteration
            if res < 1e-4:
                nit += n + 1
                break
            if n == 99:
                print('##################newton itor problems##################')
            dz = np.linalg.solve(jac, -rhs)
            z_dz = np.array(z).flatten('C')
            z_dz = [sum(t) for t in zip(dz, z_dz)]
            z = [z_dz[x:x+C] for x in range(0, len(z_dz), C)]
            z = comp_correction(z, min_z)
        Theta = Theta_ref
    stop = timeit.default_timer()
    print('CPU = %5.3f sec, NT = %d' % ((stop - start), t))
    return z, z_comp_cells


nb = 50                # number of blocks
Theta = 0.1
NT = 800                # number of time steps
#z_inj = [0.2, 0.45, 0.15, 0.1, 0.1]   # h20, co2, ca, co3, caco3
#z_ini = [0.3, 0.25, 0.2, 0.2, 0.05]
z_inj = [0.2,0.4,0.2,0.1,0.05]
z_ini = [0.5,0.1,0.1,0.2,0.05]
#z_ini = [0.2, 0.45, 0.2, 0.05, 0.1]
x = np.linspace(0, 1, nb)
t = np.linspace(0, 1, NT)
z = [z_inj]                 # put the first cell in a list
for i in range(nb-1):
    z.append(z_ini)         # append list of cells with initial value
z_og = np.array(z, copy=True)
z_e = []                    # empty list for z_elements
for i in range(nb):
    z_e = np.append(z_e, comp_to_element(z_og[i], E))       # switch the initial comp list to element list
elem = int(len(z_e)/nb)                                     # amount of elements
z_e = [z_e[x:x+elem] for x in range(0, len(z_e), elem)]     # restructure the list each cell contains the cells elements
print(z_e)


z_plot, z_comp = simulate_comp_impl(nb, Theta, NT, z_e)     # Start the reactive transport solver code
print(z_plot)

m_final = Model()
for i in range(nb):
    m_final.addingstates(m_final.T, m_final.p, z_plot[i])
for i in range(nb):
    amount_elements = m_final.states[i].elementAmounts()
    m_final.solver.solve(m_final.states[i], m_final.T, m_final.p, amount_elements)
z_plot, density = m_final.output2()
#z1_plot = z_plot[0]
#z2_plot = z_plot[1]
#z3_plot = np.zeros(nb)
#for i in range(nb):
#    z3_plot[i] = 1 - sum(z_plot[i])

#plt.plot(t,z_comp_expl,label='z_1_expl example at nb/3')
#plt.plot(t,z_comp,label='z_1 at nb/3')

plt.plot(x, z_plot[0], label='H2O')
plt.plot(x, z_plot[1], label='CO2')
plt.plot(x, z_plot[2], label='Ca++')
plt.plot(x, z_plot[3], label='CO3--')
plt.plot(x, z_plot[4], label='CaCO3')

plt.ylabel('z_c')
plt.xlabel('x dimensionless')
plt.legend()
plt.show()
#print('final z', z_plot)