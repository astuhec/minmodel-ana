import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13

####################################################################################################################
## In this example, we calculate Seebeck coefficients from Boltzmann description.
####################################################################################################################

print("Which computer are you on? (ana for Apple, anast for Lenovo)")

while True:
    user_input = input("Enter 'ana' or 'anast': ").strip().lower()
    if user_input in ('ana', 'anast'):
        computer = user_input
        break
    print("Invalid input. Please type 'ana' or 'anast'.")

if computer == 'anast':
    DIR = '/Users/anast/OneDrive/Namizje/minmodel-ana/'
else:  # ana
    DIR = '/Users/ana/Desktop/minmodel-ana/'

print(f"Selected computer: {computer}")
print(f"Directory: {DIR}")

sys.path.insert(0, DIR + 'main-programs/')
import minmodel_module as module
import minmodel_helpers as helpers

input_file = DIR + 'examples/mass-ratios/input.json5'
####################################################################################################################

## Fix t0 = 1.0, which means then m_a = 0.5. Fix t1 = 3.0, which means m_b = 1/6, and m_b/m_b = 3.

## For different gaps, calculate temperature dependence of the Seebeck coefficient according to Boltzmann description.
Delta = 1.0
t0 = 1.0
t1 = 3.0

m_a = 1/(2*t0)
m_b = 1/(2*t1)

gaps = [-0.5, -0.25, 0.0, 0.25, 0.5]

for i, gap in enumerate(gaps):
    s = module.model(input_file, m_a=m_a, m_b=m_b, gap=gap)

    s.run_Tdependence()

    Seebeck = s.collect_transport()['Seebeck_boltz']
    Ts = s.Ts

    plt.plot(Ts, Seebeck, label=rf'${gap}$')
plt.legend(frameon=False).set_title(r'$E_g$', prop={'size' : 15})
plt.xlabel('$T/t_0$')
plt.ylabel(r'$S\,[k_b/e_0]$')
plt.tight_layout()
plt.show()