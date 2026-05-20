import numpy as np
import json5
from scipy.optimize import brentq

import minmodel_helpers as helpers
import minmodel_tokovi as tokovi

def load_config(path):
    with open(path, 'r', encoding="utf-8") as f:
        return json5.load(f)
    
class model:
    def __init__(self, input_file,
                 m_a=None, m_b=None, gap=None, Delta=None, V0=None,):
        config = load_config(input_file)
        self.config = config
    
        self.option = config.get("option")
        self.Nk = config.get("Nk")
        dk = config.get("dk")
        self.L = int(2*np.pi / dk)
        self.K = np.arange(-self.Nk//2,self.Nk//2) * 2*np.pi/self.L

        print(f'=' * 80 + '\n' + 'Started 2-band calculation' + '\n' + f'=' * 80, flush=True)
        print(f'Initialized 1d lattice with Nk={self.Nk} momenta points.', flush=True)
        
        self.m_a = 1/(2*config.get("t1")) if m_a==None else m_a
        self.m_b = 1/(2*config.get("t2")) if m_b==None else m_b
        self.gap = config.get("gap") if gap==None else gap
        self.Delta = config.get("Delta") if Delta==None else Delta
        self.V0 = config.get("V0") if V0==None else V0

        self.tol = config.get("tol")
        self.maxiter = config.get("maxiter")
        
        self.v_bare = tokovi.V_bare(self.m_a, self.m_b, self.K)
        self.energije = helpers.Energies(self.m_a, self.m_b, self.gap, self.Delta, self.K, self.tol)
        self.energije_bare = helpers.Energies_bare(self.m_a, self.m_b, self.gap, self.K)
        self.mu = 0.5 * (np.min(self.energije[1]) + np.max(self.energije[0]))
    
        # physical data
        self.Ts = []
        self.mus = []
        self.Deltas = []
        self.gaps = []

        self.masses_vale = []
        self.ks_vale = []
        self.edges_vale = []

        self.masses_cond = []
        self.ks_cond = []
        self.edges_cond = []
        
        self.K0_boltz = []
        self.K1_boltz = []
        self.K0_kubo = []
        self.K1_kubo = []

        # numerical data
        self.mu_errs = []
        self.Delta_errs = []

    def find_muDelta(self, T, mu_low, mu_up, Delta0=1.) -> None:
        if self.option == 'fixed':
            mu_bisect = brentq(helpers.bisect_condition_mu_fixedDelta, mu_low, mu_up, args=(T, self.K, self.m_a, self.m_b, self.gap, self.Delta), xtol=1e-10, rtol=1e-10)
            Delta = self.Delta
            err = 0.
        elif self.option == 'consistent':
            mu_bisect = brentq(helpers.bisect_condition_mu_consistentDelta, mu_low, mu_up, args=(T, self.K, self.m_a, self.m_b, self.gap, self.Delta, self.V0, self.maxiter), xtol=1e-10, rtol=1e-10)
            Delta, err = helpers.F_full(self.m_a, self.m_b, self.gap, Delta0, self.K, self.V0, mu_bisect, T, self.tol, self.maxiter, np.max(np.hstack([self.tol, self.Delta_errs])))
        self.mu = mu_bisect
        self.Delta = Delta
        self.energije = helpers.Energies(self.m_a, self.m_b, self.gap, self.Delta, self.K, self.tol)    
        mu_err = helpers.condition_mu(self.m_a, self.m_b, self.gap, self.K, self.Delta, self.mu, T)
    
        self.mu_errs.append(mu_err)
        self.Delta_errs.append(err)

    def velocities(self):
        self.v_dressed = tokovi.V_dressed(self.v_bare, self.m_a, self.m_b, self.gap, self.K, self.Delta, self.tol)

    def transport_functions(self, epsilons, Gamma, dict_form=None, faktor=0.2):
        self.velocities()
        phi = tokovi.phi_Kubo(self.K, epsilons, self.v_dressed, self.energije, self.mu, Gamma)
        if dict_form == None:
            return phi
        elif dict_form == True:
            phi_boltz = tokovi.phi_Boltzmann(self.K, epsilons, self.v_dressed, self.energije, self.mu, Gamma, faktor=faktor)
            results = {'phi_kubo' : phi,
                       'epsilons' : epsilons,
                       'phi_boltz' : phi_boltz}
            return results

    def evaluate_Kn(self, Nomega, eps, Gammas, T) -> None:
        omega_max = np.sqrt(np.abs(np.arccosh(1/(eps*4*T))) * 2 * T)
        epsilons = np.linspace(-omega_max, omega_max, Nomega)

        Ngamma = len(Gammas)
        k0_kubos = np.zeros(Ngamma)
        k1_kubos = np.zeros(Ngamma)
        k0_boltzs = np.zeros(Ngamma)
        k1_boltzs = np.zeros(Ngamma)
        for g, Gamma in enumerate(Gammas):
            v_dressed = tokovi.V_dressed(self.v_bare, self.m_a, self.m_b, self.gap, self.K, self.Delta, self.tol)
            phi = tokovi.phi_Kubo(self.K, epsilons, v_dressed, self.energije, self.mu, Gamma)
            k0_kubo, k1_kubo = tokovi.Kn_Kubo(phi, epsilons, T)
            k0_kubos[g] = k0_kubo
            k1_kubos[g] = k1_kubo
            k0_boltz, k1_boltz = tokovi.Kn_Boltzmann(self.K, v_dressed, self.energije, self.mu, T, Gamma)
            k0_boltzs[g] = k0_boltz
            k1_boltzs[g] = k1_boltz
        self.K0_kubo.append(k0_kubos)
        self.K1_kubo.append(k1_kubos)
        self.K0_boltz.append(k0_boltzs)
        self.K1_boltz.append(k1_boltzs)
    
    def run_Tdependence(self) -> None:
        dmu = self.config.get("dmu")
        eps = self.config.get("eps")
        Nomega = self.config.get("Nomega")
        Gammas = self.config.get("Gammas")
        
        T_low = self.config.get("T_low")
        T_high = self.config.get("T_high")
        T_len = self.config.get("T_len")
        Ts = np.linspace(T_low, T_high, T_len)
        print('-' * 80, flush=True)
        print('Started to find temperature dependence of transport coefficients.', flush=True)
        for i, T in enumerate(Ts):
            msg = f'Progress: {i/len(Ts)}'
            print('\r' + msg, end='', flush=True)
            self.Ts.append(T)

            mu_low = self.mu - dmu
            mu_upp = self.mu + dmu
            self.find_muDelta(T, mu_low, mu_upp)
            self.Deltas.append(self.Delta)
            self.mus.append(self.mu)

            self.evaluate_Kn(Nomega, eps, Gammas, T)

            self.gaps.append(helpers.Gap(self.energije))

            k_vale, k_cond, m_vale, m_cond, edge_vale, edge_cond = helpers.Extrema(self.m_a, self.m_a, self.gap, self.Delta, self.K, 20. * np.max(np.hstack([self.tol, self.Delta_errs])))
            self.ks_vale.append(k_vale)
            self.ks_cond.append(k_cond)
            self.masses_vale.append(m_vale)
            self.masses_cond.append(m_cond)
            self.edges_vale.append(edge_vale)
            self.edges_cond.append(edge_cond)

        print('-' * 80, flush=True)
        print('Finished temperature dependence of transport coefficients.', flush=True)

    def collect_transport(self):
        K0_kubo = np.swapaxes(self.K0_kubo, 0, 1)
        K1_kubo = np.swapaxes(self.K1_kubo, 0, 1)
        K0_boltz = np.swapaxes(self.K0_boltz, 0, 1)
        K1_boltz = np.swapaxes(self.K1_boltz, 0, 1)
        
        Seebeck_kubo = - K1_kubo / K0_kubo / np.array(self.Ts)
        Seebeck_boltz = -K1_boltz / K0_boltz / np.array(self.Ts)

        transport_data = {'cond_kubo' : K0_kubo,
                          'cond_boltz' : K0_boltz,
                          'Seebeck_kubo' : Seebeck_kubo,
                          'Seebeck_boltz' : Seebeck_boltz}
        return transport_data

    def collect_physical_data(self):
        physical_data = {'Ts' : np.array(self.Ts),
                         'mus' : np.array(self.mus),
                         'Deltas' : np.array(self.Deltas),
                         'gaps' : np.array(self.gaps)}
        return physical_data
        
    def collect_valence_data(self):
        vale_data = {'masses' : np.array(self.masses_vale),
                     'edges' : np.array(self.edges_vale),
                     'ks' : np.array(self.ks_vale)}
        return vale_data
        
    def collect_conduction_data(self):
        cond_data = {'masses' : np.array(self.masses_cond),
                     'edges' : np.array(self.edges_cond),
                     'ks' : np.array(self.ks_cond)}
        return cond_data

    def collect_numerical_data(self):
        num_data = {'Delta_errs' : np.array(self.Delta_errs),
                    'mu_errs' : np.array(self.mu_errs)}
        return num_data