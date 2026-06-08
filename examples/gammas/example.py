import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13

####################################################################################################################
## In this example, we calculate Seebeck coefficients for different Gamma
####################################################################################################################

print("Which computer are you on? (ana for Apple, anast for Lenovo)")

while True:
    user_input = input("Enter 'ana' or 'anast': ").strip().lower()
    if user_input in ('ana', 'anast'):
        computer = user_input
        break
    print("Invalid input. Please type 'ana' or 'anast'.")

if computer == 'anast':
    DIR = '/Users/anast/minmodel-ana/'
else:  # ana
    DIR = '/Users/ana/Desktop/minmodel-ana/'

print(f"Selected computer: {computer}")
print(f"Directory: {DIR}")

sys.path.insert(0, DIR + 'main-programs/')
import minmodel_module as module
import minmodel_helpers as helpers

input_file = DIR + 'examples/gammas/input.json5'
####################################################################################################################

## Fix t0 = 1.0, which means then m_a = 0.5. Fix t1 = 3.0, which means m_b = 1/6, and m_b/m_b = 3.

## For different gaps, calculate temperature dependence of the Seebeck coefficient according to Boltzmann description.
Delta = 1.0
t0 = 1.0
t1 = 3.0

m_a = 1/(2*t0)
m_b = 1/(2*t1)

gap = 0.5

fig, ax = plt.subplots(ncols=3, figsize=(12,4))

epsilons = np.linspace(-3.0,3.0,1501)
s = module.model(input_file, m_a=m_a, m_b=m_b, gap=gap)

s.run_Tdependence()
s.run_lowT_dependence()

Seebeck_Boltz = s.collect_transport()['Seebeck_boltz']
Seebeck_Kubo = s.collect_transport()['Seebeck_kubo']

phys_data = s.collect_physical_data()
Ts = phys_data['Ts']
mus = phys_data['mus']

#ax[0].plot(Ts, mus, '.-', label=rf'${gap}$', )

epsilons = np.linspace(-3.0,3.0,1501)
for i in range(len(s.Gammas)):
    ax[1].plot(Ts, Seebeck_Kubo[:,i], '.-', label=rf'Kubo, $\Gamma={s.Gammas[i]}$')
    trans = s.transport_functions(epsilons, s.Gammas[i], dict_form=True)
    ax[0].plot(epsilons, trans['phi_kubo'])

ax[2].plot(s.K, s.energije[0] - s.mu_GS)
ax[2].plot(s.K, s.energije[1] - s.mu_GS)

    

ax[0].axhline(0.5*(np.min(s.mu_GS)))
ax[1].legend(frameon=False).set_title(r'$E_g$', prop={'size' : 15})
plt.xlabel('$T/t_0$')
plt.ylabel(r'$S\,[k_B/e_0]$')
plt.tight_layout()
plt.show()