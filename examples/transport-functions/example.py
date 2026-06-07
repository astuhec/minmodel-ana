import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13

####################################################################################################################
## This example illustrates Kubo and Boltzmann transport functions and the meaning of Gamma, spectral function width.
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

input_file = DIR + 'examples/transport-functions/input.json5'
####################################################################################################################

## (1) Initialize model
s = module.model(input_file)

## (2) Transport functions
fig, ax = plt.subplots(ncols=3, figsize=(14,4))
epsilons = np.linspace(-3.0,3.0,1501)
Gamma = 0.005
transport_functions = s.transport_functions(epsilons, Gamma, dict_form=True)
phi_kubo = transport_functions["phi_kubo"]
phi_boltz = transport_functions["phi_boltz"]

## (3.0) Quasiparticle dispersion
ax[0].set_title('Effective band structure')
ax[0].set_xlabel(r'$k$')
ax[0].set_ylabel(r'$\epsilon_{m \,k}$')
ax[0].plot(s.K, s.energije_bare[0], color='black', ls='dashed', label='bare bands')
ax[0].plot(s.K, s.energije_bare[1], color='black', ls='dashed')
ax[0].plot(s.K, s.energije[0], label='$m=-$')
ax[0].plot(s.K, s.energije[1], label='$m=+$')
ax[0].legend(frameon=False)
ax[0].set_ylim(-3.0,3.0)
ax[0].set_xlim(-3.0,3.0)

## (3.1) Distinction between Kubo and Boltzmann transport function
ax[1].set_title('Kubo VS Boltzmann transport function,' + '\n' + rf'$\Gamma={Gamma}$')
ax[1].set_xlabel(r'$\epsilon$')
ax[1].set_ylabel(r'$\sigma(\epsilon)$')
ax[1].plot(epsilons, phi_kubo, label='Kubo')
ax[1].plot(epsilons, phi_boltz, label='Boltzmann')
ax[1].legend(frameon=False)
ax[1].set_yscale('log')
ax[1].set_ylim(min(phi_kubo)*0.1, np.max([max(phi_kubo), max(phi_boltz)]) * 5.0)

## (3.2) Role of spectral function width Gamma in Kubo transport function
ax[2].set_title('Kubo transport function for different $\Gamma$')
Gammas = [0.005, 0.008, 0.01]
mins, maxs = [], []
for i, Gamma in enumerate(Gammas):
    phi = s.transport_functions(epsilons, Gamma)
    mins.append(min(phi))
    maxs.append(max(phi))
    ax[2].plot(epsilons, phi, label=f'${Gamma}$')
ax[2].legend(frameon=False).set_title(r'$\Gamma$', prop={'size' : 13})
ax[2].set_yscale('log')
ax[2].set_ylim(min(mins), max(maxs))

plt.tight_layout()
plt.show()