import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13

####################################################################################################################
## In this example, we calculate mass ratios in the effective band structure.
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

input_file = DIR + 'examples/mass-ratios/input.json5'
####################################################################################################################

## (1) Initialize model just to get K
s = module.model(input_file)
K = s.K

## (2) Fix t0 = 1.0, which means then m_a = 0.5.
# Calculate effective masses for different m_b and bare gaps, at fixed order parameter Delta=1.0
Delta = 1.0
m_a = 0.5
m_bs = np.linspace(0.1,0.5,200)
gaps = [-0.5, -0.25, 0.0, 0.25, 0.5]

m_plus = np.zeros((len(gaps), len(m_bs)))
m_minus = np.zeros_like(m_plus)

for i, gap in enumerate(gaps):
    for j, m_b in enumerate(m_bs):
        _, _, m_vale, m_cond, _, _ = helpers.Extrema(m_a, m_b, gap, Delta, K)
        m_minus[i,j] = m_vale
        m_plus[i,j] = m_cond

bare_ratio = m_a / m_bs
dressed_ratio = m_minus / m_plus

## (3) Plot dressed mass ratio against bare mass ratio
for i, gap in enumerate(gaps):
    plt.plot(bare_ratio, dressed_ratio[i], label=rf'${gap}$')
plt.legend(frameon=False).set_title(r'$E_g$', prop={'size' : 15})
plt.xlabel(r'$m_0/m_1$')
plt.ylabel(r'$m_-/m_+$')
plt.xlim(1.0,5.0)
plt.ylim(0.0,3.0)
plt.tight_layout()
plt.show()