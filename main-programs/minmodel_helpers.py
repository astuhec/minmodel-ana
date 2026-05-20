import numpy as np
from sympy import var, lambdify, diff
from sympy.parsing.mathematica import parse_mathematica
from numba import njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning
from scipy.special import expit

# Suppress NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

def meja_valencni(m_a, m_b):
    x = m_a/m_b
    return (x-1)/np.sqrt(x)

def meja_prevodni(m_a, m_b):
    x = m_a/m_b
    return (1-x)/np.sqrt(x)

a, b, e, d, k = var('a b e d k')
# bare lower band, and its first and second derivative
expr_a = parse_mathematica('-e/2 - k^2/(2 a)')
E_a = lambdify([a, e, k], expr_a, 'numpy')
dE_a = lambdify([a, b, k], diff(expr_a, k), 'numpy')
d2E_a = lambdify([a, b, k], diff(expr_a, k, 2), 'numpy')

# bare upper band, and its first and second derivative
expr_b = parse_mathematica('e/2 + k^2/(2 b)')
E_b = lambdify([b, e, k], expr_b, 'numpy')
dE_b = lambdify([b, e, k], diff(expr_b, k), 'numpy')
d2E_b = lambdify([b, e, k], diff(expr_b, k, 2), 'numpy')

# quasiparticle lower band, and its first and second derivative
expr_alpha = parse_mathematica('1/4 (-(1/a) + 1/b) k^2 - Sqrt[(d/1)^2 + 1/4 (e + ((a+b) k^2)/(2 a b))^2]')
def E_alpha(m_a, m_b, gap, Delta, K, epsE=1e-10):
    if np.max(np.abs(Delta)) < epsE:
        return E_a(m_a, gap, K)
    return lambdify([a, b, e, d, k], expr_alpha, 'numpy')(m_a, m_b, gap, Delta, K)

def E_alpha1(m_a, m_b, gap, Delta, K, epsE=1e-10):
    if np.max(np.abs(Delta)) < epsE:
        return dE_a(m_a, gap, K)
    return lambdify([a, b, e, d, k], diff(expr_alpha, k), 'numpy')(m_a, m_b, gap, Delta, K)

def E_alpha2(m_a, m_b, gap, Delta, K,  epsE=1e-10):
    if np.max(np.abs(Delta)) < epsE:
        return d2E_a(m_a, gap, K)
    return lambdify([a, b, e, d, k], diff(expr_alpha, k, 2), 'numpy')(m_a, m_b, gap, Delta, K)

# quasiparticle upper band, and its first and second derivative
expr_beta = parse_mathematica('1/4 (-(1/a) + 1/b) k^2 + Sqrt[(d/1)^2 + 1/4 (e + ((a+b) k^2)/(2 a b))^2]')
def E_beta(m_a, m_b, gap, Delta, K, epsE=1e-10):
    if np.max(np.abs(Delta)) < epsE:
        return E_b(m_b, gap, K)
    return lambdify([a, b, e, d, k], expr_beta, 'numpy')(m_a, m_b, gap, Delta, K)

def E_beta1(m_a, m_b, gap, Delta, K, epsE=1e-10):
    if np.max(np.abs(Delta)) < epsE:
        return dE_b(m_b, gap, K)
    return lambdify([a, b, e, d, k], diff(expr_beta, k), 'numpy')(m_a, m_b, gap, Delta, K)
   
def E_beta2(m_a, m_b, gap, Delta, K, epsE=1e-10):
    if np.max(np.abs(Delta)) < epsE:
        return d2E_b(m_b, gap, K)
    return lambdify([a, b, e, d, k], diff(expr_beta, k, 2), 'numpy')(m_a, m_b, gap, Delta, K)

# k at which valence band has a maximum, holds ONLY if Delta = 1
def max_vale(m_a, m_b, gap):
    x = m_a / m_b
    if gap < meja_valencni(m_a, m_b):
        return np.sqrt(2*x/(1+x) * (np.abs(gap) - (1-x)/np.sqrt(x)))
    return 0.

# k at which conduction band has a minimum, holds ONLY if Delta = 1
def min_vale(m_a, m_b, gap):
    x = m_a / m_b
    if gap < meja_prevodni(m_a, m_b):
        return np.sqrt(2*x/(1+x) * ((1-x)/np.sqrt(x) - gap))
    return 0.

def Energies(m_a, m_b, gap, Delta, K, epsE=1e-10):
    mat = np.zeros((2, len(K)))
    mat[0,:] = E_alpha(m_a,  m_b, gap, Delta, K, epsE)
    mat[1,:] = E_beta(m_a, m_b, gap, Delta, K, epsE)
    return mat

def Energies_bare(m_a, m_b, gap, K):
    mat = np.zeros((2, len(K)))
    mat[0,:] = E_a(m_a, gap, K)
    mat[1,:] = E_b(m_b, gap, K)
    return mat

def Gap(energije):
    return np.min(energije[1]) - np.max(energije[0])

def Extrema(m_a, m_b, gap, Delta, K, epsE=1e-10):
    energije = Energies(m_a, m_b, gap, Delta, K, epsE)
    
    k_vale = K[np.argmax(energije[0])]
    k_cond = K[np.argmin(energije[1])]
    
    m_vale = 1/np.abs(E_alpha2(m_a, m_b, gap, Delta, k_vale, epsE))
    m_cond = 1/np.abs(E_beta2(m_a, m_b, gap, Delta, k_cond, epsE))

    edge_vale = np.max(energije[0])
    edge_cond = np.min(energije[1])
    return k_vale, k_cond, m_vale, m_cond, edge_vale, edge_cond

def xiE_k(m_a, m_b, gap, K, Delta):
    Ea, Eb = E_a(m_a, gap, K), E_b(m_b, gap, K)
    xi_k = (Eb - Ea)/2
    E_k = np.sqrt(xi_k**2 + Delta**2)
    return xi_k, E_k

# unitary transformation which diagonalizes h_k: h_k = U_k @ diag(E_alpha, E_beta) @ U_k.conj().T [can do without conj]
def U_k(m_a, m_b, gap, K, Delta):
    xi_k, E_k = xiE_k(m_a, m_b, gap, K, Delta)
    uk = (0.5 * (1 + xi_k/E_k))**0.5
    vk = (0.5 * (1 - xi_k/E_k))**0.5
    u = np.zeros((2, 2, len(K)))
    u[0,0,:] = uk
    u[0,1,:] = vk
    u[1,0,:] = -vk
    u[1,1,:] = uk
    return u

# Fermi-Dirac distribution function
def fd(E, T):
    if T == 0: return E < 0
    return expit(-E/T)

def condition_mu(m_a, m_b, gap, K, Delta, mu, T, epsE=1e-10):
    Nk = len(K)
    if T == 0: return 0.
    return(np.sum(fd(E_alpha(m_a, m_b, gap, Delta, K, epsE) - mu, T)) / Nk + \
           np.sum(fd(E_beta(m_a, m_b, gap, Delta, K, epsE) - mu, T)) / Nk - 1)

# analytic chemical potential at low T
def mu_analytic(m_a, m_b, e_a, e_b, z, T):
    return (e_a + e_b)/2 + T * (0.5 * np.log(z) + 0.25 * np.log(m_a/m_b))

# find mu if Delta is fixed
def bisect_condition_mu_fixedDelta(mu, T, K, m_a, m_b, gap, Delta, epsE=1e-10):
    return condition_mu(m_a, m_b, gap, K, Delta, mu, T, epsE)

# converge order parameter at some T, mu (self-consistently determined order parameter)
def F_full(m_a, m_b, gap, Delta, K, V0, mu, T, tol=1e-8, maxiter=500, epsE=1e-8):
    Nk = len(K)
    err, N_iters = 1.0, 0
    while err > tol and N_iters < maxiter:
        _, E_k = xiE_k(m_a, m_b, gap, K, Delta)
        Ealpha = E_alpha(m_a, m_b, gap, Delta, K, epsE)
        Ebeta = E_beta(m_a, m_b, gap, Delta, K, epsE)
        Delta_new = V0/(2*Nk) * np.sum(Delta/E_k * (fd(Ealpha - mu, T) - fd(Ebeta - mu, T)))
        err = np.abs(Delta - Delta_new)
        Delta = Delta_new
    return Delta, err

def bisect_condition_mu_consistentDelta(mu, T, K, m_a, m_b, gap, Delta, V0, tol=1e-8, maxiter=500):
    Delta, _ = F_full(m_a, m_b, gap, Delta, K, V0, mu, T, tol, maxiter, tol)
    return condition_mu(m_a, m_b, gap, K, Delta, mu, T)

def delta_approximation(x, width):
    return 1/(2*np.pi*width**2)**0.5 * np.exp(-x**2/(2*width**2))

''' density of states '''
def DoS(K, energije, epsilons, mu, velocity, faktor=0.2, shape='Gaussian'):
    Nk = len(K)
    v_max = np.max(np.abs(velocity))
    sigma = np.sqrt(v_max * (epsilons[1] - epsilons[0]) * (K[1] - K[0])) * faktor
    dos = np.zeros((2, len(epsilons)))
    for k in prange(Nk):
        for alpha in range(2):
            dos[alpha] += delta_approximation(epsilons - energije[alpha,k] + mu, sigma) 
    return dos / Nk

def to_scalar_if_single(x):
    x = np.asarray(x)
    if x.size == 1:
        return float(x.item())
    return x