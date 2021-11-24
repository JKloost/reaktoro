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

#############################################
# Instead of using problem, a new ChemicalState is given. This makes it easier to compute things such as
# mole_total = ChemicalQuantity(state).value("phaseAmount(Aqueous)")
#############################################
db = Database('supcrt98.xml')
editor = ChemicalEditor(db)
#editor.addAqueousPhaseWithElementsOf("H2O CO2 CaCO3") # aqueous
editor.addAqueousPhase("H2O(l) CO2(aq) Ca++ CO3-- CaCO3(aq)")
#editor.addGaseousPhase(['CO2(g)'])
#editor.addMineralPhase("Calcite") # solid

# def moltokg(component,kg):
#     return kg * Cspecies.molarMass(component)

system = ChemicalSystem(editor)
# problem_ic = EquilibriumProblem(system)
# problem_bc = EquilibriumProblem(system)
# state_ic = ChemicalState(system)
# state_bc = ChemicalState(system)
# state = ChemicalState(system)
# path = EquilibriumPath(system)
# Cplot = ChemicalPlot(system)
# Cspecies = Species()
# output = path.output()
# output.filename("path_result.txt")
# output.add("phaseMass(Aqueous)")
# output.add("phaseVolume(Aqueous)")
# output.add("speciesAmount(CO2(aq))")

p = 100
T = 60
mole_h20 = 100
mole_co2 = 5
mole_ca = 5
mole_CO3 = 5
cells = np.arange(0, 100, 1)
mole_cells_co2 = np.linspace(5, 0, len(cells))

state_ic.setTemperature(T, 'celsius')
state_ic.setPressure(p, 'bar')
state_ic.setSpeciesAmount('H2O(l)', mole_h20, 'mol')
state_ic.setSpeciesAmount('Ca++', mole_ca, 'mol')
state_ic.setSpeciesAmount('CO3--',mole_CO3, 'mol')
#problem_ic.add('CaCO3(aq)',mole_ca,'mol')

state_bc.setTemperature(T, 'celsius')
state_bc.setPressure(p, 'bar')
state_bc.setSpeciesAmount('H2O(l)', mole_h20, 'mol')
state_bc.setSpeciesAmount('Ca++', mole_ca, 'mol')
state_bc.setSpeciesAmount('CO3--',mole_CO3, 'mol')
state_bc.setSpeciesAmount('CO2(aq)', mole_co2, 'mol')
#problem_bc.add('CaCO3(aq)', mole_ca, 'mol')

states = []
for i in range(len(cells)):
    state = ChemicalState(system)
    state.setTemperature(T, 'celsius')
    state.setPressure(p, 'bar')
    state.setSpeciesAmount('H2O(l)', mole_h20, 'mol')
    state.setSpeciesAmount('Ca++', mole_ca, 'mol')
    state.setSpeciesAmount('CO3--', mole_CO3, 'mol')
    state.setSpeciesAmount('CO2(aq)', mole_cells_co2[i], 'mol')
    states.append(state)

# print(states[0])
# print(states[50])
# print(states[-1])
#Elimination matrix
#           ['H2O', 'CO2', 'Ca++', 'CO3--', 'CaCO3']
# E = np.array([[1,   0,      0,      0,      0],
#             [  0,   1,      0,      0,      0],
#             [  0,   0,      1,      0,      1],
#             [  0,   0,      0,      1,      1]])
# stoich_matr = np.array([0, 0, 1, 1, -1])
# stoich_matr.transpose()
# #mole_total = ChemicalQuantity(state_ic).value("phaseAmount(Aqueous)")
# mole_total = mole_h20+mole_ca+mole_co2+mole_CO3
# #mole_total = mole_water + mole_ca + mole_co2 + mole_CO3
# mole_fraction_water = mole_h20 / mole_total
# mole_fraction_co2 = mole_co2 / mole_total
# mole_fraction_ca = mole_ca / mole_total
# mole_fraction_CO3 = mole_CO3 / mole_total
#
# # ['H2O', 'CO2', 'Ca++', 'CO3--', 'CaCO3']
# component_mole_frac = [mole_fraction_water, mole_fraction_co2, mole_fraction_ca, mole_fraction_CO3, 0]
# element = np.zeros(E.shape[0])
# for i in range(E.shape[0]):
#     element[i] = np.divide(np.sum(np.multiply(E[i], component_mole_frac)),
#                                       np.sum(np.multiply(E, component_mole_frac)))
# vec_state_as_np = np.asarray(element)
# ze = np.append(vec_state_as_np, 1 - np.sum(vec_state_as_np))
#
# print('amount components by elimination matrix:', ze)
#Creaction = Reaction(system)
#print(Creaction.equilibriumConstant())
solver = EquilibriumSolver(system) # solves the system

for i in range(len(cells)):
    amount_elements = states[i].elementAmounts()
    solver.solve(states[i], T, p, amount_elements)
# p_solve = state_ic.pressure()
# T_solve = state_ic.temperature()
# amount_elements_ic = state_ic.elementAmounts()
# print('amount elements by reaktoro [C, Ca, H, O, Z]:', amount_elements_ic)
# ic_solved = solver.solve(state_ic, T_solve, p_solve, amount_elements_ic) # solve
# #print(system.numElements())
#
# p_solve_bc = state_bc.pressure()
# T_solve_bc = state_bc.temperature()
# amount_elements_bc = state_bc.elementAmounts()
# print('amount elements by reaktoro producer', amount_elements_bc)
# bc_solved = solver.solve(state_bc, T_solve_bc, p_solve_bc, amount_elements_bc) # solve
#
# path = path.solve(state_ic, state_bc)
#Cquantity = ChemicalQuantity(state) # to get quantity values of the state/sytem


#total_mole_aq = system.elementAmountInPhase(iphase=0)
#total_mole_aq = state.phaseAmount('Aqueous', 'mol')
#total_mole_solid = state.phaseAmount('Calcite', 'mol')
#vap = total_mole_solid / (total_mole_aq + total_mole_solid)
#volume_aq = Cquantity.value("phaseVolume(Aqueous)")
#volume_solid = Cquantity.value("phaseVolume(Calcite)")
# mass_aq_ic = ChemicalQuantity(state_ic).value("phaseMass(Aqueous)")
# volume_aq_ic = ChemicalQuantity(state_ic).value("phaseVolume(Aqueous)")
# density_ic = mass_aq_ic/volume_aq_ic
# mass_aq_bc = ChemicalQuantity(state_bc).value("phaseMass(Aqueous)")
# volume_aq_bc = ChemicalQuantity(state_bc).value("phaseVolume(Aqueous)")
# density_bc = mass_aq_bc/volume_aq_bc
#
# #print(state_ic)
# state_ic.output('result_simple_ic.txt')
# state_bc.output('result_simple_bc.txt')
#
# print('Amount of H2O ic:', state_ic.speciesAmount('H2O(l)'))
# print('Amount of CO2(aq) ic:', state_ic.speciesAmount('CO2(aq)'))
# print('Amount of CO3-- ic:', state_ic.speciesAmount('CO3--'))
# print('Density of Aqueous phase ic:', str(density_ic))
# print('Density of Aqueous phase bc:', str(density_bc))

# print('Amount of C in aqueous phase:', state_ic.elementAmountInPhase('C', 'Aqueous'))
# print('Amount of C in gaseous phase:', state.elementAmountInPhase('C', 'Gaseous'))

# point1 = [0, density_bc]
# point2 = [100, density_ic]
# x_val = [point1[0], point2[0]]
# y_val = [point1[1], point2[1]]
# plt.figure(1)
# plt.plot(x_val, y_val, label='Density')
# plt.legend()
# plt.show()
# plt.figure(2)
# plt.plot([0, 100], [state_ic.speciesAmount('CO2(aq)'), state_bc.speciesAmount('CO2(aq)')], label='mol CO2')
# plt.plot([0, 100], [state_ic.speciesAmount('CaCO3(aq)'), state_bc.speciesAmount('CaCO3(aq)')], label='mol CaCO3(aq)')
# plt.plot([0, 100], [state_ic.speciesAmount('Ca++'), state_bc.speciesAmount('Ca++')], label='mol Ca++')
# plt.plot([0, 100], [state_ic.speciesAmount('CO3--'), state_bc.speciesAmount('CO3--')], label='mol CO3--')
# plt.legend()
# plt.show()
# filearray = np.loadtxt("path_result.txt", skiprows=1)
# data = filearray.T
# [mass, volume, amountCO2] = np.arange(3)
# cells = np.linspace(1,len(data[mass]),len(data[mass]))
# plt.figure(4)
# plt.plot(cells,data[mass]/data[volume])
# #plt.plot(cells,data[amountCO2])
# plt.show()

CO2_amount = [state.speciesAmount("CO2(aq)") for state in states]
CaCO3_amount = [state.speciesAmount("CaCO3(aq)") for state in states]
Ca_amount = [state.speciesAmount("Ca++") for state in states]
CO3_amount = [state.speciesAmount("CO3--") for state in states]
mass = [ChemicalQuantity(state).value("phaseMass(Aqueous)") for state in states]
volume = [ChemicalQuantity(state).value("phaseVolume(Aqueous)") for state in states]
density = np.zeros(len(mass))
for i in range(len(mass)):
    density[i] = mass[i]/volume[i]

plt.figure(5)
plt.plot(cells,density,label='Density')
plt.legend()
plt.show()
plt.figure(6)
plt.plot(cells, CO2_amount, label='CO2')
plt.plot(cells, CaCO3_amount, label='CaCO3(aq)')
plt.plot(cells, Ca_amount, label='Ca++')
plt.plot(cells, CO3_amount, label='CO3--')
plt.legend()
plt.show()