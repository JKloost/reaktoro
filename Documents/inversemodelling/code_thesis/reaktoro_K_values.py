from reaktoro import *

thermo = Thermo(Database("supcrt98.xml"))
T = 320 # K
p = 1e7 # Pa


reac1 = 'H+ + OH- = H2O(l)'
reac2 = 'HCO3- + H+ = CO2(aq) + H2O(l)'
reac3 = 'CO3-- + H+ = HCO3-'
reac4 = 'CaCO3(aq) + H+ = Ca++ + HCO3-'
print('reac1',thermo.lnEquilibriumConstant(T, p, reac1).val)
print('reac2',thermo.lnEquilibriumConstant(T, p, reac2).val)
print('reac3',thermo.lnEquilibriumConstant(T, p, reac3).val)
print('reac4',thermo.lnEquilibriumConstant(T, p, reac4).val)
