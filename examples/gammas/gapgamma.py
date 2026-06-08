import os, sys
os.environ['PATH'] = '/Library/TeX/texbin:' + os.environ['PATH']
import numpy as np
from scipy.special import roots_legendre
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13

from itertools import product

####################################################################################################################
## In this example, we calculate Seebeck coefficients for different Gamma
####################################################################################################################

DIR = '/Users/ana/Desktop/minmodel-ana/'


sys.path.insert(0, DIR + 'main-programs/')
import minmodel_module as module
import minmodel_helpers as helpers

input_file = DIR + 'examples/gammas/input2.json5'
####################################################################################################################

## Fix t0 = 1.0, which means then m_a = 0.5. Fix t1 = 3.0, which means m_b = 1/6, and m_b/m_b = 3.

## For different gaps, calculate temperature dependence of the Seebeck coefficient according to Boltzmann description.
t0 = 1.0
t1 = 3.0

m_a = 1/(2*t0)
m_b = 1/(2*t1)

Deltas = [0.1, 0.2, 0.3, 0.5, 1.0]
Gaps = [-0.5, -0.25, 0.0, 0.25, 0.5]

params = list(product(Deltas, Gaps))

task_id = int(sys.argv[1])
(Delta, gap) = params[task_id]

s = module.model(input_file, m_a=m_a, m_b=m_b, gap=gap, Delta=Delta)

s.run_Tdependence()
s.run_lowT_dependence()

transport_data = s.collect_transport()

np.savez(f'gapgamma2_Delta{Delta}_gap{gap}.npz', **transport_data)