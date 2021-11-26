''''Testing reaktoro'''
import matplotlib.pyplot as plt
import numpy as np
from reaktoro import *

'''Components to elements'''

# TODO:
#  input components,                                            Y
#  get them to elements with elimination matrix,                NOT, USED ELEMENTAMOUNTS()
#  Perform reaction with reaktoro,                              CHECK, WITH ELEMENTS
#  output density,                                              Y
#  and component distribution,                                  Y
#  Try in 1D vector code, with injection and producer well,     Y
#  OBL                                                          Y

#############################################
# Instead of using problem, a new ChemicalState is given. This makes it easier to compute things such as
# mole_total = ChemicalQuantity(state).value("phaseAmount(Aqueous)")
#############################################

###INPUT###
class Variables():
    def __init__(self):
        self.p = 100 # in bar
        self.T = 60 # in celsius
        self.mole_h20 = 100
        self.mole_ca = 5
        self.mole_CO3 = 5
        number_cells = 100
        self.cells = np.arange(0, number_cells, 1)
        self.mole_cells_co2 = np.linspace(5, 0, len(self.cells))

class Model(Variables):
    def __init__(self):
        super().__init__()
        self.editor = ChemicalEditor(Database('supcrt98.xml'))
        self.editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")
        self.system = ChemicalSystem(self.editor)
        self.solver = EquilibriumSolver(self.system)  # solves the system
        self.states = []
        # Two version can be coded with the same solution. One uses the EquilibriumProblem function,
        # Other will use the state function. Both need to output a state in order to be solved by EquilibriumSolver

    def addingproblem(self, temp, pres, mole_comp):
        self.problem = EquilibriumProblem(self.system)
        self.problem.setTemperature(temp, 'celsius')
        self.problem.setPressure(pres, 'bar')
        self.problem.add('H2O', self.mole_h20, 'mol')
        self.problem.add('Ca++', self.mole_ca, 'mol')
        self.problem.add('CO3--', self.mole_CO3, 'mol')
        self.problem.add('CO2', mole_comp, 'mol')
        self.state = equilibrate(self.problem) # Equilibrate the problem in order to write to state
        self.states.append(self.state)

    def addingstates(self, temp,pres,mole_comp):
        self.state = ChemicalState(self.system)
        self.state.setTemperature(temp, 'celsius')
        self.state.setPressure(pres, 'bar')
        self.state.setSpeciesAmount('H2O(l)', self.mole_h20, 'mol')
        self.state.setSpeciesAmount('Ca++', self.mole_ca, 'mol')
        self.state.setSpeciesAmount('CO3--', self.mole_CO3, 'mol')
        self.state.setSpeciesAmount('CO2(aq)', mole_comp, 'mol')
        self.states.append(self.state)

    def ouput(self):
        H20_amount = [state.speciesAmount("H2O(l)") for state in self.states]
        CO2_amount = [state.speciesAmount("CO2(aq)") for state in self.states]
        CaCO3_amount = [state.speciesAmount("CaCO3(aq)") for state in self.states]
        Ca_amount = [state.speciesAmount("Ca++") for state in self.states]
        CO3_amount = [state.speciesAmount("CO3--") for state in self.states]
        mass = [ChemicalQuantity(state).value("phaseMass(Aqueous)") for state in self.states]
        volume = [ChemicalQuantity(state).value("phaseVolume(Aqueous)") for state in self.states]
        mole_total = [ChemicalQuantity(state).value("phaseAmount(Aqueous)") for state in self.states]
        density = np.zeros(len(mass))
        z_h20 = np.zeros(len(mass))
        z_co2 = np.zeros(len(mass))
        z_ca = np.zeros(len(mass))
        z_co3 = np.zeros(len(mass))
        for i in range(len(mass)):
            z_h20[i] = H20_amount[i]/mole_total[i]
            z_co2[i] = CO2_amount[i]/mole_total[i]
            z_ca[i] = Ca_amount[i]/mole_total[i]
            z_co3[i] = CO3_amount[i]/mole_total[i]
        for i in range(len(mass)):
            density[i] = mass[i] / volume[i]
        return z_h20, z_co2, z_ca, z_co3, density

m = Model()
for i in range(len(m.cells)):
    # m.addingproblem(m.T, m.p, m.mole_cells_co2[i])
    m.addingstates(m.T, m.p, m.mole_cells_co2[i])
for i in range(len(m.cells)):
    amount_elements = m.states[i].elementAmounts()
    print(amount_elements)
    m.solver.solve(m.states[i], m.T, m.p, amount_elements)

z_h20, z_co2, z_ca, z_co3, density = m.ouput()

plt.figure(1)
plt.plot(m.cells, density, label='Density')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(m.cells, z_co2, label='CO2')
plt.plot(m.cells, z_h20, label='H2O')
plt.plot(m.cells, z_ca, label='Ca++')
plt.plot(m.cells, z_co3, label='CO3--')
plt.legend()
plt.show()