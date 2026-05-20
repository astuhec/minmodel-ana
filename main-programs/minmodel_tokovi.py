import numpy as np
from sympy import var, lambdify, diff
from sympy.parsing.mathematica import parse_mathematica
from numba import njit, prange
import warnings
from numba.core.errors import NumbaPerformanceWarning
from scipy.special import expit

from minmodel_helpers import U_k, delta_approximation

''' derivative of Fermi-Dirac distribution ''' 
@njit
def fd_1(omega, T):
    #return expit(-omega/T) *(1 - expit(-omega/T))/T
    return -1/(4*T)/np.cosh(omega/(2*T))**2

''' bare velocity: just derivative of h_k w.r.t. k '''
def V_bare(m_a, m_b, K):
    v = np.zeros((2, 2, len(K)))
    v[0,0] = -K/m_a
    v[1,1] = +K/m_b
    return v

''' velocity: according to Hellmann-Feynmann's theorem '''
def V_dressed(V_bare, m_a, m_b, gap, K, Delta, epsE=1e-10):
    if np.abs(Delta) < epsE:
        return V_bare
    else:
        uk = U_k(m_a, m_b, gap, K, Delta)
        v = np.einsum('jix,jkx,kmx->imx', uk.conj(), V_bare, uk)
        return v

''' Boltzmann transport function'''
def phi_Boltzmann(K, omegas, velocity, energije, mu, Gamma, faktor=0.2):
    Nk = len(K)
    phi = np.zeros(len(omegas))

    v_max = np.max(np.abs(velocity))
    sigma = np.sqrt(v_max * (omegas[1] - omegas[0]) * (K[1] - K[0])) * faktor

    for i, omega in enumerate(omegas):
        for alpha in range(2):
            phi[i] += np.sum(delta_approximation(omega - energije[alpha] + mu, sigma) * velocity[alpha,alpha]**2)
    
    tau = 1/(2*Gamma)
    return phi / Nk * tau

''' spectral function at a single momentum point'''
@njit
def spektralna_k(omega, mu, energije_k, Gamma):
    N_orb = len(energije_k)
    A = np.zeros((N_orb, N_orb))
    for orb in range(N_orb):
        A[orb, orb] = 1/np.pi * Gamma / ( (omega - (energije_k[orb] - mu))**2 + Gamma**2 )
    return A

''' Kubo transport function '''
@njit(parallel=False, cache=True)
def phi_Kubo(K, omegas, velocity, energije, mu, Gamma):
    Nk = len(K)
    phi = np.zeros(len(omegas))
    for i in prange(len(omegas)):
        omega = omegas[i]
        for j in [0,Nk//2]:
            A = spektralna_k(omega, mu, energije[:,j], Gamma)
            vel = velocity[:,:,j]
            for a in range(2):
                for b in range(2):
                    phi[i] += vel[a,b] * A[b,b] * vel[b,a] * A[a,a]

        for j in range(1,Nk//2):
            A = spektralna_k(omega, mu, energije[:,j], Gamma)
            vel = velocity[:,:,j]
            for a in range(2):
                for b in range(2):
                    phi[i] += 2 * vel[a,b] * A[b,b] * vel[b,a] * A[a,a]
    return phi.real / Nk * np.pi

''' Boltzmann transport coefficients'''
@njit(parallel=False, cache=True)
def Kn_Boltzmann(K, velocity, energije, mu, T, Gamma):
    K0, K1 = 0., 0.
    Nk = len(K)
    for alpha in range(2):
        K0 += np.sum( velocity[alpha,alpha]**2 * (-fd_1(energije[alpha] - mu, T)) )
        K1 += np.sum( (energije[alpha] - mu) * velocity[alpha,alpha]**2 * (-fd_1(energije[alpha] - mu, T)) )
    tau = 1/(2*Gamma)
    K0 = K0 * tau / Nk
    K1 = K1 * tau / Nk
    return K0, K1

''' integral with trapezoidal approximation '''
def integral_omega(integrand, omega):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(integrand.real, omega)
    else:
        return np.trapz(integrand.real, omega)

''' Kubo transport coefficients'''
def Kn_Kubo(phi, omegas, T):
    mfd1 = -fd_1(omegas, T)
    K0 = integral_omega(phi * mfd1, omegas)
    K1 = integral_omega(omegas * phi * mfd1, omegas)
    return K0, K1