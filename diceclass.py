import pandas as pd
import numpy as np
import pyomo.environ as pe
from dice2016 import (
    get_b, get_sig0, get_lam, get_l, get_ga, get_al, get_gsig, get_sigma,
    get_pbacktime, get_cost1, get_etree, get_cumetree, get_rr, get_forcoth,
    get_optlrsav, get_cpricebase, dice2016
)

class DICE2016Model:
    def __init__(self, param_df=None):
        if param_df is None:
            param_df = pd.read_csv('dice2016_parameters.csv')
        self.params = dict(zip(param_df.key, param_df.value))
        self.params['nt'] = int(self.params['nt'])
        self.params['logfile'] = bool(self.params['logfile'])
        self._define_dependent_parameters()

    def _define_dependent_parameters(self):
        p = self.params
        p['b11'], p['b21'], p['b22'], p['b32'], p['b33'] = get_b(p['b12'], p['b23'], p['mateq'], p['mueq'], p['mleq'])
        p['sig0'] = get_sig0(p['e0'], p['q0'], p['miu0'])
        p['lam'] = get_lam(p['fco22x'], p['t2xco2'])
        p['l'] = get_l(p['pop0'], p['popadj'], p['popasym'], p['nt'])
        p['ga'] = get_ga(p['ga0'], p['dela'], p['tstep'], p['nt'])
        p['al'] = get_al(p['a0'], p['ga'], p['nt'])
        p['gsig'] = get_gsig(p['gsigma1'], p['dsig'], p['tstep'], p['nt'])
        p['sigma'] = get_sigma(p['sig0'], p['gsig'], p['tstep'], p['nt'])
        p['pbacktime'] = get_pbacktime(p['pback'], p['gback'], p['nt'])
        p['cost1'] = get_cost1(p['pbacktime'], p['sigma'], p['expcost2'])
        p['etree'] = get_etree(p['eland0'], p['deland'], p['nt'])
        p['cumetree'] = get_cumetree(p['etree'], p['tstep'], p['nt'])
        p['rr'] = get_rr(p['prstp'], p['tstep'], p['nt'])
        p['forcoth'] = get_forcoth(p['fex0'], p['fex1'], p['nt'])
        p['optlrsav'] = get_optlrsav(p['dk'], p['elasmu'], p['prstp'], p['gama'])
        p['cpricebase'] = get_cpricebase(p['cprice0'], p['gcprice'], p['tstep'], p['nt'])

    def run_model(self):
        self.model = dice2016(**self.params)
        self.results = self._extract_results()

    def _extract_results(self):
        results = {}
        time_periods = np.array([t for t in self.model.time_periods])
        results_df = pd.DataFrame(index=time_periods)
        results_df.index.name = 'time'
        
        results_df['MAT'] = np.array([pe.value(self.model.MAT[t]) for t in self.model.time_periods])
        results_df['CCATOT'] = np.array([pe.value(self.model.CCATOT[t]) for t in self.model.time_periods])
        results_df['PPM'] = results_df['MAT'] / 2.124
        results_df['ATFRAC'] = (results_df['MAT'] - 588) / (results_df['CCATOT'] + 0.000001)
        results_df['ATFRAC2010'] = (results_df['MAT'] - self.params['mat0']) / (0.00001 + results_df['CCATOT'] - results_df['CCATOT'][0])
        results_df['E'] = np.array([pe.value(self.model.E[t]) for t in self.model.time_periods])
        results_df['TATM'] = np.array([pe.value(self.model.TATM[t]) for t in self.model.time_periods])
        results_df['Y'] = np.array([pe.value(self.model.Y[t]) for t in self.model.time_periods])
        results_df['DAMFRAC'] = np.array([pe.value(self.model.DAMFRAC[t]) for t in self.model.time_periods])
        results_df['CPC'] = np.array([pe.value(self.model.CPC[t]) for t in self.model.time_periods])
        results_df['CPRICE'] = np.array([pe.value(self.model.CPRICE[t]) for t in self.model.time_periods])
        results_df['MIU'] = np.array([pe.value(self.model.MIU[t]) for t in self.model.time_periods])
        results_df['L'] = np.array([pe.value(self.params['l'][t]) for t in range(self.params['nt'])])
        results_df['AL'] = np.array([pe.value(self.params['al'][t]) for t in range(self.params['nt'])])
        results_df['YGROSS'] = np.array([pe.value(self.model.YGROSS[t]) for t in self.model.time_periods])
        results_df['GA'] = np.array([pe.value(self.params['ga'][t]) for t in range(self.params['nt'])])
        results_df['K'] = np.array([pe.value(self.model.K[t]) for t in self.model.time_periods])
        results_df['S'] = np.array([pe.value(self.model.S[t]) for t in self.model.time_periods])
        results_df['I'] = np.array([pe.value(self.model.I[t]) for t in self.model.time_periods])
        results_df['YNET'] = np.array([pe.value(self.model.YNET[t]) for t in self.model.time_periods])
        results_df['DAMAGES'] = np.array([pe.value(self.model.DAMAGES[t]) for t in self.model.time_periods])
        results_df['ABATECOST'] = np.array([pe.value(self.model.ABATECOST[t]) for t in self.model.time_periods])
        results_df['SIGMA'] = np.array([pe.value(self.params['sigma'][t]) for t in range(self.params['nt'])])
        results_df['FORC'] = np.array([pe.value(self.model.FORC[t]) for t in self.model.time_periods])
        results_df['FORCOTH'] = np.array([pe.value(self.params['forcoth'][t]) for t in range(self.params['nt'])])
        results_df['PERIODU'] = np.array([pe.value(self.model.PERIODU[t]) for t in self.model.time_periods])
        results_df['C'] = np.array([pe.value(self.model.C[t]) for t in self.model.time_periods])
        results_df['ETREE'] = np.array([pe.value(self.params['etree'][t]) for t in range(self.params['nt'])])
        results_df['CCA'] = np.array([pe.value(self.model.CCA[t]) for t in self.model.time_periods])
        results_df['MU'] = np.array([pe.value(self.model.MU[t]) for t in self.model.time_periods])
        results_df['ML'] = np.array([pe.value(self.model.ML[t]) for t in self.model.time_periods])
        
        #results = results_df.to_dict(orient='list')
        #results['UTILITY'] = pe.value(self.model.UTILITY)
        return results_df

    def get_results(self):
        return self.results

# Example usage:
# model = DICE2016Model()
# model.run_model()
# results = model.get_results()