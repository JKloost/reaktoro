''''Testing reaktoro'''
'''Components to elements'''
from reaktoro import *

db = Database('supcrt98.xml')
editor = ChemicalEditor(db)
editor.addAqueousPhaseWithElementsOf("H2O CO2 CaCO3") # aqueous
#editor.addGaseousPhase(['CO2(g)'])
editor.addMineralPhase("Calcite") # solid

system = ChemicalSystem(editor)
p = 100
T = 60
problem = EquilibriumProblem(system)
problem.setTemperature(T, 'celsius')
problem.setPressure(p, 'bar')
mole_water = 10
mole_co2 = 5
mole_ca = 1
problem.add('H2O', mole_water, 'mol')
problem.add('CO2', mole_co2, 'mol')
problem.add('Ca++', mole_ca, 'mol')

#Elimination matrix
#           ['H2O', 'CO2', 'Ca++', 'CO3--', 'CaCO3']
E = np.array([[1,   0,      0,      0,      0],
            [  0,   1,      0,      0,      0],
            [  0,   0,      1,      0,      1],
            [  0,   0,      0,      1,      1]])
stoich_matr = np.array([0, 0, 1, 1, -1])
stoich_matr.transpose()

mole_total = mole_water + mole_ca + mole_co2
mole_fraction_water = mole_water / mole_total
mole_fraction_co2 = mole_co2 / mole_total
mole_fraction_ca = mole_ca / mole_total

# ['H2O', 'CO2', 'Ca++', 'CO3--', 'CaCO3']
component_mole_frac = [mole_fraction_water, mole_fraction_co2, mole_fraction_ca, 0, 0]
element = np.zeros(E.shape[0])
for i in range(E.shape[0]):
    element[i] = np.divide(np.sum(np.multiply(E[i], component_mole_frac)),
                                      np.sum(np.multiply(E, component_mole_frac)))
vec_state_as_np = np.asarray(element)

z = np.append(vec_state_as_np, 1 - np.sum(vec_state_as_np))
print(z)
#Creaction = Reaction(system)
#print(Creaction.equilibriumConstant())
K = np.array([6, 0.8, 0.07, 0.4, 0.4]) # TODO how to get equilibrium constants
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


# TODO flash operations
solver = EquilibriumSolver(system) # solves the system
#state = ChemicalState(system)
state = equilibrate(problem) # equilibrate the added elements
p_solve = problem.pressure()
T_solve = problem.temperature()
amount_elements = problem.elementAmounts()
solver.solve(state, T_solve, p_solve, amount_elements) # solve
Cquantity = ChemicalQuantity(state) # to get quantity values of the state/sytem


#total_mole_aq = system.elementAmountInPhase(iphase=0)
total_mole_aq = state.phaseAmount('Aqueous', 'mol')
total_mole_solid = state.phaseAmount('Calcite', 'mol')
vap = total_mole_solid / (total_mole_aq + total_mole_solid)
volume_aq = Cquantity.value("phaseVolume(Aqueous)")
volume_solid = Cquantity.value("phaseVolume(Calcite)")
#print(state)
state.output('result_simple.txt')
print('Amount of H2O:', state.speciesAmount('H2O(l)'))
print('Amount of CO2(aq):', state.speciesAmount('CO2(aq)'))
#print('Amount of CO2(g):', state.speciesAmount('CO2(g)'))
print('Amount of Ca++:', state.speciesAmount('Ca++'))
print('Amount of CO3--:', state.speciesAmount('CO3--'))
print('Amount of CaCO3:', state.speciesAmount('Calcite'))

#print('Amount of C in aqueous phase:', state.elementAmountInPhase('C', 'Aqueous'))
#print('Amount of C in gaseous phase:', state.elementAmountInPhase('C', 'Gaseous'))