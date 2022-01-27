'''One phase solver'''
import numpy as np
from darts_interpolator import DartsInterpolator
import timeit
import matplotlib.pyplot as plt
from reaktoro import *

'''Components to elements'''

# TODO:
#  Elimination matrix
#  comp -> element
#  Combine with OBL code

E = np.array([[1,0,0,0,0],
              [0,1,0,0,0],
              [0,0,1,0,1],
              [0,0,0,1,1]])

def comp_to_element(z_c,E):
    z_e = np.zeros(E.shape[0])
    for i in range(E.shape[0]):
        z_e[i] = np.divide(np.sum(np.multiply(E[i], z_c)),np.sum(np.multiply(E, z_c))) # ze e_i z - Ez
    return z_e

def density_comp_to_element(z_c,rho_t,E):
    rho_e = np.zeros(E.shape[0])
    for i in range(E.shape[0]):
        rho_e[i] = np.multiply(rho_t, np.sum(np.multiply(E[i], z_c)))  # rho_t e_i z
    return rho_e

###INPUT###
class Variables():
    def __init__(self):
        self.p = 1e7 # in bar
        self.T = 320 # in celsius
        number_cells = nb

class Model(Variables):
    def __init__(self):
        super().__init__()
        self.editor = ChemicalEditor(Database('supcrt98.xml'))
        self.editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")
        self.system = ChemicalSystem(self.editor)
        self.solver = EquilibriumSolver(self.system)  # solves the system
        self.reactions = ReactionSystem(self.editor)
        self.reaction = ReactionEquation('Ca++ + HCO3- = CaCO3(aq)')
        # print(self.reaction.numSpecies())
        self.states = []
        # Two version can be coded with the same solution. One uses the EquilibriumProblem function,
        # Other will use the state function. Both need to output a state in order to be solved by EquilibriumSolver

    def addingproblem(self, temp, pres, z_e):
        mole_element = np.zeros(len(z_e))
        Mtotal = sum(z_e)
        for i in range(len(z_e)):
            mole_element[i] = Mtotal * z_e[i]
        #print('mole element',mole_element)
        self.problem = EquilibriumProblem(self.system)
        self.problem.setTemperature(temp, 'kelvin')
        self.problem.setPressure(pres, 'pascal')
        # self.problem.add('H', 0.4, 'mol')
        # self.problem.add('O', 0.3, 'mol')
        # self.problem.add('C', 0.2, 'mol')
        # self.problem.add('Ca', 0.1, 'mol')
        # self.problem.add('O', mole_element[0], 'mol')
        # self.problem.add('H', mole_element[1], 'mol')
        # self.problem.add('C', mole_element[2], 'mol')
        # self.problem.add('Ca', mole_element[3], 'mol')
        self.problem.add('H2O', mole_element[0], 'mol')
        self.problem.add('CO2', mole_element[1], 'mol')
        self.problem.add('Ca++', mole_element[2], 'mol')
        self.problem.add('CO3--', mole_element[3], 'mol')
        self.state = equilibrate(self.problem)  # Equilibrate the problem in order to write to state
        #print(self.state)
        self.states.append(self.state)

    def addingstates(self, temp, pres, z_e):
        mole_element = np.zeros(len(z_e))
        Mtotal = sum(z_e)
        for i in range(len(z_e)):
            mole_element[i] = Mtotal * z_e[i]
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
            density[i] = mass[i] / volume[i]
        z_c = [z_h2O[0],z_co2[0],z_ca[0],z_co3[0],z_caco3[0]]
        return z_c, density

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
        z_c = [z_h2O,z_co2,z_ca,z_co3,z_caco3]
        return z_c, density


def RachfordRice(*z):
    z2 = list(z[0])
    #K = np.array([3, 0.1])
    #K = np.array([6, 0.8, 0.07])
    K = np.array([6, 3, 0.8, 0.07])
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
    mu_L = 1
    mu_g = 1
    rho_g = 1
    rho_L = 1
    n1 = 2
    n2 = 2
    m = Model()
    m.addingproblem(m.T, m.p, z_beta)
    #print(m.states[0])
    #print('initial', z)
    amount_elements = m.states[0].elementAmounts() # C, Ca, H, O
    m.solver.solve(m.states[0], m.T, m.p, amount_elements)
    z_c_operator, rho_t = m.output()
    z_e_operator = comp_to_element(z_c_operator, E)
    rho_e = density_comp_to_element(z_c_operator, rho_t, E)
    x, y = RachfordRice(z_e_operator)
    Ln = (z_e_operator[0] - y[0]) / (x[0] - y[0])  # Ln = (z[1] - y[1]) / (x[1] - y[1]) gives the same sat
    Ln = 1 # single phase
    #density_reaktoro = 1
    for i in range(len(z_e_operator)):
        #if Ln < 0 or Ln > 1:
        #    beta_operator = z_beta[i]
        if Ln <= 0:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (z_e_operator[i] * (((1 - 0) ** n2) / mu_g)) / (((1 - 0) ** n2) / mu_g)
            #beta_operator = z_beta[i]
            rho_t = (1-0) * rho_g # s * rho
            alpha_operator = z_e_operator[i] * rho_e[i]
        elif Ln >= 1:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            #beta_operator = (z_e_operator[i] * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
            #rho_t = 1 * rho_L # s * rho
            beta_operator = z_e_operator[i]
            alpha_operator = z_e_operator[i] * rho_e[i]
        else:
            z_comp = Ln*x[0]+(1-Ln)*y[0]
            beta_operator = (x[i]*((Ln**n1)/mu_L) + y[i]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
            #beta_operator = z_beta[i]
            rho_t = Ln*rho_L + (1-Ln)*rho_g
            alpha_operator = z_e_operator[i] * rho_e[i]
        values[i] = beta_operator
        values[i+len(z_e_operator)] = alpha_operator
        #values[i+2*len(z_beta)] = z_comp
    return 0

def simulate_comp_impl(nb, Theta_ref, NT, z):
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
        print(t)
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
            for q in range(C):
                beta_L_deriv2[q*C+q] += 1
                #beta_L_deriv2[q*C+q] += alpha_n_deriv[q*C+q]  # Alpha operator
            beta_L_deriv2 = np.reshape(beta_L_deriv2, (C, C))
            jac[0:C, 0:C] = beta_L_deriv2
            for j in range(1,nb):
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

                if j == int(nb/3):
                    z_comp = beta[-1]
                    z_comp_cells[t] = z_comp

                beta_L_deriv2, beta_deriv2 = Theta * beta_L_deriv2, Theta * beta_deriv2
                for q in range(C):
                    #beta_deriv2[q * C + q] += alpha_deriv[q*C+q] # Alpha operator
                    beta_deriv2[q * C + q] += 1
                beta_L_deriv2, beta_deriv2 = np.reshape(beta_L_deriv2, (C, C)), np.reshape(beta_deriv2, (C,C))

                for i in range(C):
                    rhs = np.append(rhs, z[j][i] - zn[j][i] + Theta * (beta[i] - beta_L[i]))  # needs better way
                    #rhs = np.append(rhs, alpha[i] - alpha_n[i] + Theta * (beta[i] - beta_L[i]))
                jac[C*j:C*j+C, C*(j-1):C*(j-1)+C] = beta_L_deriv2
                jac[C*j:C*j+C, C*j:C*j+C] = beta_deriv2
            #np.set_printoptions(precision=10,suppress=True)
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
            z_dz = np.array(z).flatten('C')
            z_dz = [sum(t) for t in zip(dz, z_dz)]
            z = [z_dz[x:x+C] for x in range(0, len(z_dz), C)]
        Theta = Theta_ref
    stop = timeit.default_timer()
    print('CPU = %5.3f sec, NT = %d' % ((stop - start), t))
    return z, z_comp_cells

nb = 10
Theta = 0.2
NT = 10
components = 3          # Also change K in rachfordrice (line 14-15) and line 102 and z_inj/z_ini
C = components - 1
#z_inj = [0.9]
#z_ini = [0.1]
#z_inj = [0.9, 0.1]
#z_ini = [0.1, 0.5]
z_inj = [0.3,0.25,0.2,0.15,0.1] #h20, co2, ca, co3, caco3
z_ini = [0.3,0.2,0.3,0.1,0.1]
x = np.linspace(0, 1, nb)
t = np.linspace(0, 1, NT)
z = [z_inj]
for i in range(nb-1):
    z.append(z_ini)
z_og = np.array(z, copy=True)
z_e = []
for i in range(nb):
    z_e = np.append(z_e,comp_to_element(z_og[i], E))
elem = int(len(z_e)/nb)
z_e = [z_e[x:x+elem] for x in range(0, len(z_e), elem)]
print(z_e)
z_plot, z_comp = simulate_comp_impl(nb, Theta, NT, z_e)
m_final = Model()
for i in range(nb):
    m_final.addingproblem(m_final.T, m_final.p,z_plot[i])
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

plt.ylabel('Saturation')
plt.xlabel('x dimensionless')
plt.legend()
plt.show()
#print('final z', z_plot)