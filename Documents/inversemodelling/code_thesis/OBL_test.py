'''Testing OBL'''
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from interpolation.splines import UCGrid, CGrid, nodes
from interpolation.splines import eval_linear
from src.darts_interpolator import DartsInterpolator
from darts.physics import *
from darts.engines import *
from darts.models.physics.physics_base import PhysicsBase

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

def beta_test(z):
    mu_L = 1
    mu_g = 1
    x, y = RachfordRice(z)
    Ln = (z - y[0]) / (x[0] - y[0])
    n1 = 2
    n2 = 2
    if Ln <= 0:
        #Ln = 0
        #x = [0]
        #y = [z]
        beta_operator = (z*(((1-0)**n2)/mu_g)) / (((1-0)**n2)/mu_g)
    elif Ln >= 1:
        beta_operator = (z * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
        #Ln = 1
        #x = [z]
        #y = [0]
        #beta_operator = z / mu_g
    else:
        beta_operator = (x[0]*((Ln**n1)/mu_L) + y[0]*(((1-Ln)**n2)/mu_g)) / ((Ln**n1)/mu_L + ((1-Ln)**n2)/mu_g)
    #if beta_operator > 1:
    #    print(beta_operator)
    return beta_operator

def alpha_test(z):
    rho_L = 1
    rho_g = 1
    #x, y = RachfordRice(z)
    #Ln = (z - y[0]) / (x[0] - y[0])
    if Ln <= 0:
        alpha_operator = (z * rho_g) / ((1 ** 2) / rho_g)
        #Ln = 0
        #x = [0]
        #y = [z]
    elif Ln >= 1:
        alpha_operator = (z * rho_L) / ((1 ** 2) / rho_L)
        #Ln = 1
        #x = [z]
        #y = [0]
    else:
        alpha_operator = (x[0] * rho_L + y[0] * rho_g) / ((Ln**2)/rho_L + ((1-Ln)**2)/rho_g) # z * rho / fc
    return alpha_operator


class property_container(property_evaluator_iface):
    def __init__(self, phase_name, component_name):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.n_phases = len(phase_name)
        self.nc = len(component_name)
        self.component_name = component_name
        self.phase_name = phase_name

        # # Allocate (empty) evaluators
        # self.density_ev = []
        # self.viscosity_ev = []
        # self.rel_perm_ev = []
        # self.enthalpy_ev = []
        # self.kin_rate_ev = []
        # self.flash_ev = 0
        # self.state_ev = []
        # self.init_flash_ev = []

class MyOwnDataStruct:
    def __init__(self, nc, zmin):
        """
        Data structure class which holds various input parameters for simulation
        :param nc: number of components used in simulation
        :param zmin: actual 0 used for composition (ussualy >0, around some small epsilon)
        :param temp: temperature
        """
        self.num_comp = nc
        self.min_z = zmin
        # self.temperature = temp
        # self.stoich_matrix = stoich_matrix
        # self.exp_w = exp_w
        # self.exp_g = exp_g
        # self.pressure_init = pressure_init
        # self.c_r = c_r
        # self.kin_fact = kin_fact

class my_own_interpol(operator_set_evaluator_iface):
    def __init__(self, input_data, properties):
        super().__init__()  # Initialize base-class
        # Store your input parameters in self here, and initialize other parameters here in self
        self.input_data = input_data
        # self.num_comp = input_data.num_comp
        # self.min_z = input_data.min_z
        # self.temperature = input_data.temperature
        # self.exp_w = input_data.exp_w
        # self.exp_g = input_data.exp_g
        # self.c_r = input_data.c_r
        # self.kin_fact = input_data.kin_fact
        self.property = properties
        self.cache = False

        self.counter = 0

    def evaluate(self, state, values):
        x,y = RachfordRice(state)
        Ln = (z - y[0]) / (x[0] - y[0])
        n1 = 2
        n2 = 2
        if Ln <= 0:
            beta_operator = (z * (((1 - 0) ** n2) / mu_g)) / (((1 - 0) ** n2) / mu_g)
        elif Ln >= 1:
            beta_operator = (z * ((1 ** n1) / mu_L)) / ((1 ** n1) / mu_L)
        else:
            beta_operator = (x[0] * ((Ln ** n1) / mu_L) + y[0] * (((1 - Ln) ** n2) / mu_g)) / (
                        (Ln ** n1) / mu_L + ((1 - Ln) ** n2) / mu_g)
        for i in range(len(z)):
            values[i] = beta_operator
        return 0

class Model(PhysicsBase):
    def __init__(self):
        super().__init__()
        self.comp = 1
        input_data_struct = MyOwnDataStruct(2, 0)
        self.property_container2 = property_container(['liq', 'gas'], ['h20'])  # phase and component
        self.values = value_vector([0])
        self.comp_etor = my_own_interpol(input_data_struct, self.property_container2)
        axes_points = index_vector([1000] + [1000] * (self.comp - 1))
        axes_min = value_vector([0] + [0] * (self.comp - 1))
        axes_max = value_vector([1] + [1] * (self.comp - 1))
        phases = 2
        comp_itor = self.create_interpolator(self.comp_etor, self.comp, phases, axes_points, axes_min, axes_max)
        z_interpol = value_vector([0.9])
        comp_itor.evaluate(z_interpol, self.values)

def simulate_comp_impl(nb, Theta_ref, NT, z):
    # comp = 2
    # input_data_struct = MyOwnDataStruct(2,0)
    # property_container2 = property_container(['liq','gas'],['h20','vapor'])  # phase and component
    # #values = [0]
    # comp_etor = my_own_interpol(input_data_struct, property_container2)
    # axes_points = index_vector([1000]+[1000]*(comp-1))
    # axes_min = value_vector([0] + [0] * (comp-1))
    # axes_max = value_vector([1] + [1] * (comp-1))
    # phases = 2
    # comp_itor = PhysicsBase.create_interpolator(comp_etor, comp_etor, comp, phases, axes_points, axes_min, axes_max)
    # z_interpol = value_vector([z[0]])
    # values = comp_itor.evaluate(z_interpol, values)
    m = Model()
    rhs = np.zeros(nb)
    jac = np.zeros([nb, nb]) # the more compositions, the larger the jacobian
    nit = 0  # Counter for non-linear iterations
    Theta = Theta_ref / 1000
    #beta_interpol = DartsInterpolator(beta_test, axes_points=[100], axes_min=[0], axes_max=[1])
    #alpha_interpol = DartsInterpolator(alpha_test, axes_points=[100], axes_min=[0], axes_max=[1])
    for t in range(NT):
        zn = np.array(z, copy=True)
        for n in range(100):
            beta_L, beta_L_deriv = comp_itor.interpolate_point_with_derivatives([z[0]])
            rhs[0] = 0
            jac[0,0] = 1
            for i in range(1,nb):
                beta, beta_deriv = beta_interpol.interpolate_point_with_derivatives([z[i]])
                #alpha_n = alpha_interpol.interpolate_point([zn[i]])
                #alpha, alpha_deriv = alpha_interpol.interpolate_point_with_derivatives([z[i]])
                rhs[i] = z[i] - zn[i] + Theta * (beta - beta_L)
                #rhs[i] = alpha - alpha_n + Theta * (beta - beta_L) # alpha-alpha_(t-1)
                jac[i, i-1] = - Theta * beta_L_deriv[0]
                jac[i,i] = 1 + Theta * beta_deriv[0]
                #jac[i, i] = alpha_deriv[0] + Theta * beta_deriv[0]
                beta_L = beta
                beta_L_deriv = beta_deriv
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

def simulate_comp_expl_binary(nb, Theta_ref_expl, NT_expl, z_expl):
    #x, y = RachfordRice_binary(z[0])
    Theta_expl = Theta_ref_expl / 1000
    for t in range(NT_expl):
        beta_L_expl = beta_test(z_expl[0])
        for i in range(1, nb):
            beta_expl = beta_test(z_expl[i])
            z_expl[i] -= Theta_expl * (beta_expl - beta_L_expl)
            beta_L_expl = beta_expl
        Theta_expl = Theta_ref_expl
    return z_expl



nb = 10
Theta = 0.1
NT = 30
components = 2
z_inj = [0.90] # array of composition-1
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
z2_plot_expl = simulate_comp_expl_binary(nb,Theta,NT,z_org)
z2_plot = simulate_comp_impl(nb, Theta, NT, z_org2)

#print(z2_plot)
plt.plot(x, z2_plot, label='impl')
plt.plot(x, z2_plot_expl, '--',label='expl')
plt.ylabel('Saturation')
plt.xlabel('x dimensionless')
#plt.ylim(0,1)
plt.legend()
plt.show()