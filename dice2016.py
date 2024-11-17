"""Python implementation of DICE2016R3-112018-v8"""

import pandas as pd
import numpy as np
import pyomo.opt as po
import pyomo.environ as pe
import logging

# Conventions:
# Parameter names: lower case
# Variable names: upper case
# Equation names: lower case

# All variables, parameters and equations use the same naming and unit convention as in
# DICE2016*. For the long names of parameter values, refer to the
# dice2016_parameters.csv file. For long names of variables, refer to the output.csv
# file.

# *two exceptions:
# the cardinality of set `t` is here `nt`, and `t` is a iterative value over this set.
# t ranges from 0 to nt-1 (numpy convention), rather than 1 to nt (in GAMS DICE).

# One additional parameter is defined in the dice_parameters.csv that are not part of
# the model:
# logfile = 0 | 1   set to 0 to not dump logfile, or 1 to dump logfile). May 
#                   help debugging.

# Define parameters that are functions of other parameters
def get_b(b12, b23, mateq, mueq, mleq):
    b11 = 1 - b12
    b21 = b12 * mateq/mueq
    b22 = 1 - b21 - b23
    b32 = b23 * mueq/mleq
    b33 = 1 - b32
    return(b11, b21, b22, b32, b33)


def get_sig0(e0, q0, miu0):
    return e0 / (q0 * (1 - miu0))


def get_lam(f2xco2, t2xco2):
    return f2xco2 / t2xco2


def get_l(pop0, popadj, popasym, nt):
    l = np.zeros(nt)
    l[0] = pop0
    for t in range(1, nt):
        l[t] = l[t-1] * (popasym / l[t-1])**popadj
    return l


def get_ga(ga0, dela, tstep, nt):
    ga = ga0 * np.exp(-dela * tstep * np.arange(nt))
    return ga


def get_al(al0, ga, nt):
    al = np.zeros(nt)
    al[0] = al0
    for t in range(1, nt):
        al[t] = al[t-1] / (1 - ga[t-1])
    return al


def get_gsig(gsigma1, dsig, tstep, nt):
    gsig = np.zeros(nt)
    gsig[0] = gsigma1
    for t in range(1, nt):
        gsig[t] = gsig[t-1] * ((1 + dsig) ** tstep)
    return gsig


def get_sigma(sig0, gsig, tstep, nt):
    sigma = np.zeros(nt)
    sigma[0] = sig0
    for t in range(1, nt):
        sigma[t] = sigma[t-1] * np.exp(gsig[t-1] * tstep)
    return sigma


def get_pbacktime(pback, gback, nt):
    pbacktime = pback * (1 - gback)**(np.arange(nt))
    return pbacktime


def get_cost1(pbacktime, sigma, expcost2):
    return pbacktime * sigma / expcost2 / 1000


def get_etree(eland0, deland, nt):
    return eland0 * (1 - deland)**(np.arange(nt))


def get_cumetree(etree, tstep, nt):
    cumetree = np.zeros(nt)
    cumetree[0] = 100
    for t in range(1, nt):
        cumetree[t] = cumetree[t-1] + etree[t-1] * (tstep/3.666)
    return cumetree


def get_rr(prstp, tstep, nt):
    return 1 / ((1 + prstp)**(tstep * (np.arange(nt))))


# TODO: don't hard code period 17 as regime transition here
def get_forcoth(fex0, fex1, nt):
    forcoth = np.ones(nt) * fex1
    forcoth[:17] = fex0 + (fex1 - fex0)/17 * np.arange(17)
    return forcoth


# TODO: don't hard code 0.004 here
def get_optlrsav(dk, elasmu, prstp, gama):
    return (dk + 0.004) / (dk + 0.004 * elasmu + prstp) * gama
    

def get_cpricebase(cprice0, gcprice, tstep, nt):
    return cprice0 * (1 + gcprice) ** (tstep * np.arange(nt))


# Main model definition
def dice2016(**kwargs):
    model = pe.ConcreteModel()
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    
    # Define sets
    model.time_periods = pe.Set(initialize=np.arange(kwargs['nt'], dtype=int))

    # Define bounds and initial conditions
    # Where models are initialised with a fixed value in the first period, this is
    # implemented as a bound for t = 0

    # TODO: allow user to vary these bounds in the input CSV
    def k_bounds(model,t):
        if t == 0: 
            return (kwargs['k0'], kwargs['k0'])
        else:
            return (1, np.inf)

    def mat_bounds(model, t):
        if t == 0:
            return(kwargs['mat0'], kwargs['mat0'])
        else:
            return(10, np.inf)

    def mu_bounds(model, t):
        if t == 0:
            return(kwargs['mu0'], kwargs['mu0'])
        else:
            return(100, np.inf)

    def ml_bounds(model, t):
        if t == 0:
            return(kwargs['ml0'], kwargs['ml0'])
        else:
            return(1000, np.inf)

    def c_bounds(model,t):
        return (2, np.inf)

    def tocean_bounds(model, t):
        if t == 0:
            return(kwargs['tocean0'], kwargs['tocean0'])
        else:
            return(-1, 20)

    def tatm_bounds(model,t):
        if t == 0: 
            return (kwargs['tatm0'], kwargs['tatm0'])
        else:
            return (-np.inf, 12)

    def cpc_bounds(model,t):
        return (0.01, np.inf)


    def cca_bounds(model, t):
        if t == 0:
            return (400, 400)
        else:
            return (0, kwargs['fosslim'])

    def k_bounds(model,t):
        if t == 0: 
            return (kwargs['k0'], kwargs['k0'])
        else:
            return (0, np.inf)

    # TODO: abatement fraction is ripe for customization, would even suggest its own
    # input CSV defining user-specified and lower bounds over the model lifetime
    def miu_bounds(model, t):
        if t == 0:
            return (kwargs['miu0'], kwargs['miu0'])
        elif t < 30:
            return (0, 1)
        else:
            return (0, kwargs['limmiu'])

    def s_bounds(model,t):
        if t <= kwargs['nt'] - 10:
            return (-np.inf, np.inf)
        else:
            return (kwargs['optlrsav'], kwargs['optlrsav'])

    # Variables and their scopes
    model.MIU = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=miu_bounds) 
    model.FORC = pe.Var(model.time_periods, domain=pe.Reals)
    # difference here: I define TATM over Reals, because TATM < 0 is possible
    model.TATM = pe.Var(model.time_periods, domain=pe.Reals, bounds=tatm_bounds)
    model.TOCEAN = pe.Var(model.time_periods, domain=pe.Reals, bounds=tocean_bounds)
    model.MAT = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=mat_bounds) 
    model.MU = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=mu_bounds) 
    model.ML = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=ml_bounds) 
    model.E = pe.Var(model.time_periods, domain=pe.Reals) 
    model.EIND = pe.Var(model.time_periods, domain=pe.Reals) 
    model.C = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=c_bounds) 
    model.K = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=k_bounds) 
    model.CPC = pe.Var(model.time_periods, domain=pe.NonNegativeReals, bounds=cpc_bounds) 
    model.I = pe.Var(model.time_periods, domain=pe.NonNegativeReals) 
    model.S = pe.Var(model.time_periods, domain=pe.Reals, bounds=s_bounds) 
    model.RI = pe.Var(model.time_periods, domain=pe.Reals)
    model.Y = pe.Var(model.time_periods, domain=pe.NonNegativeReals) 
    model.YGROSS = pe.Var(model.time_periods, domain=pe.NonNegativeReals) 
    model.YNET = pe.Var(model.time_periods, domain=pe.Reals) 
    model.DAMAGES = pe.Var(model.time_periods, domain=pe.Reals) 
    model.DAMFRAC = pe.Var(model.time_periods, domain=pe.Reals) 
    model.ABATECOST = pe.Var(model.time_periods, domain=pe.Reals) 
    model.MCABATE = pe.Var(model.time_periods, domain=pe.Reals) 
    model.CCA = pe.Var(model.time_periods, domain=pe.Reals, bounds=cca_bounds) 
    model.CCATOT = pe.Var(model.time_periods, domain=pe.Reals) 
    model.PERIODU = pe.Var(model.time_periods, domain=pe.Reals)
    model.CPRICE = pe.Var(model.time_periods, domain=pe.Reals)
    model.CEMUTOTPER = pe.Var(model.time_periods, domain=pe.Reals)
    model.UTILITY = pe.Var(domain=pe.Reals)

    # Objective function
    def obj_rule(model):
        return  model.UTILITY
    model.OBJ = pe.Objective(rule=obj_rule, sense=pe.maximize)

    # Equations	
    def eeq(model,t):
        return (model.E[t] == model.EIND[t] + kwargs['etree'][t])
    model.eeq = pe.Constraint(model.time_periods, rule=eeq)

    def eindeq(model,t):
        return (model.EIND[t] == kwargs['sigma'][t] * model.YGROSS[t] * (1 - model.MIU[t]))
    model.eindeq = pe.Constraint(model.time_periods, rule=eindeq)

    # TODO: fix instances of 3.666 to 3.664 in my future implemetations
    def ccacca(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.CCA[t] == model.CCA[t-1] + model.EIND[t-1] * kwargs['tstep'] / 3.666)
    model.ccacca = pe.Constraint(model.time_periods, rule=ccacca)

    def ccatoteq(model, t):
        return (model.CCATOT[t] == model.CCA[t] + kwargs['cumetree'][t])
    model.ccatoteq = pe.Constraint(model.time_periods, rule=ccatoteq)

    def force(model,t): 
        return (model.FORC[t] == kwargs['fco22x'] * (pe.log10(model.MAT[t]/588.000)/pe.log10(2)) + kwargs['forcoth'][t])
    model.force = pe.Constraint(model.time_periods, rule=force)

    def damfraceq(model,t):
        return (model.DAMFRAC[t] == (kwargs['a1'] * model.TATM[t]) + (kwargs['a2'] * model.TATM[t]**kwargs['a3']))
    model.damfraceq = pe.Constraint(model.time_periods, rule=damfraceq)

    def dameq(model,t):
        return (model.DAMAGES[t] == (model.YGROSS[t] * model.DAMFRAC[t]))
    model.dameq = pe.Constraint(model.time_periods, rule=dameq)

    def abateeq(model,t):
        return (model.ABATECOST[t] == model.YGROSS[t] * kwargs['cost1'][t] * (model.MIU[t]**kwargs['expcost2']))
    model.abateeq = pe.Constraint(model.time_periods, rule=abateeq)

    def mcabateeq(model,t):
        return (model.MCABATE[t] == kwargs['pbacktime'][t] * model.MIU[t]**(kwargs['expcost2']-1))
    model.mcabateeq = pe.Constraint(model.time_periods, rule=mcabateeq)

    # this is a dupe of above in DICE2016 for the "optimal" strategy. But we leave it in.
    def carbpriceeq(model,t):
        return (model.CPRICE[t] == kwargs['pbacktime'][t] * model.MIU[t]**(kwargs['expcost2']-1))
    model.carbpriceeq = pe.Constraint(model.time_periods, rule=carbpriceeq)

    def dmiueq(model, t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.MIU[t] <= model.MIU[t-1] + kwargs['dmiulim'])
    model.dmiueq = pe.Constraint(model.time_periods, rule=dmiueq)

    def mmat(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.MAT[t] == model.MAT[t-1]*kwargs['b11'] + model.MU[t-1] * kwargs['b21'] + model.E[t-1] * kwargs['tstep'] / 3.666)
    model.mmat = pe.Constraint(model.time_periods, rule=mmat)

    def mml(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.ML[t] == model.ML[t-1] * kwargs['b33'] + model.MU[t-1] * kwargs['b23'] )
    model.mml = pe.Constraint(model.time_periods, rule=mml)

    def mmu(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.MU[t] == model.ML[t-1]*kwargs['b32'] + model.MU[t-1] * kwargs['b22'] + model.MAT[t-1] * kwargs['b12'])
    model.mmu = pe.Constraint(model.time_periods, rule=mmu)

    def tmateq(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else: 
            return (
                (
                    model.TATM[t] == model.TATM[t-1] + kwargs['c1'] * 
                    ((model.FORC[t] - (kwargs['fco22x'] / kwargs['t2xco2']) * model.TATM[t-1]) -
                    (kwargs['c3'] * (model.TATM[t-1] - model.TOCEAN[t-1])))
                )
            )
    model.tmateq = pe.Constraint(model.time_periods, rule=tmateq)

    def toceaneq(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.TOCEAN[t] == model.TOCEAN[t-1] + kwargs['c4'] * (model.TATM[t-1] - model.TOCEAN[t-1]))
    model.toceaneq = pe.Constraint(model.time_periods, rule=toceaneq)

    def ygrosseq(model,t):
        return (model.YGROSS[t] == kwargs['al'][t] * (kwargs['l'][t]/1000)**(1 - kwargs['gama']) * (model.K[t] ** kwargs['gama']))
    model.ygrosseq = pe.Constraint(model.time_periods, rule=ygrosseq)

    def yneteq(model,t):
        return (model.YNET[t] == model.YGROSS[t] * (1-model.DAMFRAC[t]))
    model.yneteq = pe.Constraint(model.time_periods, rule=yneteq)

    def yy(model,t):
        return (model.Y[t] == model.YNET[t] - model.ABATECOST[t])
    model.yy = pe.Constraint(model.time_periods, rule=yy)

    def cc(model,t):
        return (model.C[t] == model.Y[t] - model.I[t])
    model.cc = pe.Constraint(model.time_periods, rule=cc)

    def cpce(model,t):
        return (model.CPC[t] == 1000 * model.C[t] / kwargs['l'][t])
    model.cpce = pe.Constraint(model.time_periods, rule=cpce)

    def seq(model,t):
        return (model.I[t] ==  model.S[t] * model.Y[t])
    model.seq = pe.Constraint(model.time_periods, rule=seq)

    def kk(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.K[t] <= (1 - kwargs['dk'])**kwargs['tstep'] * model.K[t-1] + kwargs['tstep'] * model.I[t-1])
    model.kk = pe.Constraint(model.time_periods, rule=kk)

    def rieq(model,t):
        if t == 0:
            return pe.Constraint.Skip
        else:
            return (model.RI[t-1] == (1 + kwargs['prstp']) * (model.CPC[t] / model.CPC[t-1])**(kwargs['elasmu'] / kwargs['tstep']) - 1)
    model.rieq = pe.Constraint(model.time_periods, rule=rieq)

    def cemutotpereq(model,t):
        return (model.CEMUTOTPER[t] == model.PERIODU[t] * kwargs['l'][t] * kwargs['rr'][t])
    model.cemutotpereq = pe.Constraint(model.time_periods, rule=cemutotpereq)

    def periodueq(model,t):
        return (model.PERIODU[t] == ((model.C[t] * 1000 / kwargs['l'][t])**(1 - kwargs['elasmu']) - 1) / (1 - kwargs['elasmu']) - 1)
    model.periodueq = pe.Constraint(model.time_periods, rule=periodueq)

    def util(model):
        return (model.UTILITY == kwargs['tstep'] * kwargs['scale1'] * pe.summation(model.CEMUTOTPER) + kwargs['scale2'])
    model.util = pe.Constraint(rule=util)

    # Run the model
    solver = pe.SolverFactory('ipopt')
    results = solver.solve(model, tee=kwargs['logfile'], keepfiles=kwargs['logfile'], logfile="output.log")
    if (results.solver.status != po.SolverStatus.ok):
        logging.warning('Check solver not ok?')
    if (results.solver.termination_condition != po.TerminationCondition.optimal):  
        logging.warning('Check solver optimality?')

    print('Optimal solution value:', model.OBJ())
    return model

    
if __name__ == "__main__":
    # Read in model parameters
    param_df = pd.read_csv('dice2016_parameters.csv')
    params = dict(zip(param_df.key, param_df.value))
    params['nt'] = int(params['nt'])
    params['logfile'] = bool(params['logfile'])

    # Define dependent parameters
    params['b11'], params['b21'], params['b22'], params['b32'], params['b33'] = get_b(
        params['b12'], params['b23'], params['mateq'], params['mueq'], params['mleq']
    )
    params['sig0'] = get_sig0(params['e0'], params['q0'], params['miu0'])
    params['lam'] = get_lam(params['fco22x'], params['t2xco2'])
    params['l'] = get_l(params['pop0'], params['popadj'], params['popasym'], params['nt'])
    params['ga'] = get_ga(params['ga0'], params['dela'], params['tstep'], params['nt'])
    params['al'] = get_al(params['a0'], params['ga'], params['nt'])
    params['gsig'] = get_gsig(params['gsigma1'], params['dsig'], params['tstep'], params['nt'])
    params['sigma'] = get_sigma(params['sig0'], params['gsig'], params['tstep'], params['nt'])
    params['pbacktime'] = get_pbacktime(params['pback'], params['gback'], params['nt'])
    params['cost1'] = get_cost1(params['pbacktime'], params['sigma'], params['expcost2'])
    params['etree'] = get_etree(params['eland0'], params['deland'], params['nt'])
    params['cumetree'] = get_cumetree(params['etree'], params['tstep'], params['nt'])
    params['rr'] = get_rr(params['prstp'], params['tstep'], params['nt'])
    params['forcoth'] = get_forcoth(params['fex0'],params['fex1'],params['nt'])
    params['optlrsav'] = get_optlrsav(params['dk'], params['elasmu'], params['prstp'], params['gama'])
    params['cpricebase'] = get_cpricebase(params['cprice0'], params['gcprice'], params['tstep'], params['nt'])

    # Call the model run function with parameters defined
    model = dice2016(**params)

    # derived values in output
    mat = np.array(pe.value(model.MAT[:]))
    ccatot = np.array(pe.value(model.CCATOT[:]))
    ppm = (mat / 2.124)
    atfrac = ((mat - 588) / (ccatot + 0.000001))
    atfrac2010 = ((mat - params['mat0']) / (0.00001 + ccatot - ccatot[0]))

    # get the social cost of carbon from marginals in the dual model
    # (no, I don't understand either what I just said, or the equation)
    scc_components = {}
    for c in model.component_objects(pe.Constraint, active=True):
        if c.name in ['eeq', 'cc']:
            scc_components[c.name] = np.zeros(params['nt'])
            for index in c:
                scc_components[c.name][index] = model.dual[c[index]]
    scc = -1000 * scc_components['eeq'] / (0.00001 + scc_components['cc'])

    # define an output DataFrame and dump to CSV
    data_out = np.ones((params['nt'], 37)) * np.nan
    data_out[:, 0] = np.arange(2015, params['tstep'] * params['nt'] + 2015, params['tstep'])
    data_out[:, 1] = pe.value(model.EIND[:])
    data_out[:, 2] = pe.value(ppm)
    data_out[:, 3] = pe.value(model.TATM[:])
    data_out[:, 4] = pe.value(model.Y[:])
    data_out[:, 5] = pe.value(model.DAMFRAC[:])
    data_out[:, 6] = pe.value(model.CPC[:])
    data_out[:, 7] = pe.value(model.CPRICE[:])
    data_out[:, 8] = pe.value(model.MIU[:])
    data_out[:, 9] = scc
    data_out[:, 10] = pe.value(model.RI[:])
    data_out[:, 11] = pe.value(params['l'][:])
    data_out[:, 12] = pe.value(params['al'][:])
    data_out[:, 13] = pe.value(model.YGROSS[:])
    data_out[:, 14] = pe.value(params['ga'][:])
    data_out[:, 15] = pe.value(model.K[:])
    data_out[:, 16] = pe.value(model.S[:])
    data_out[:, 17] = pe.value(model.I[:])
    data_out[:, 18] = pe.value(model.YNET[:])
    data_out[:, 19] = pe.value(model.DAMAGES[:])
    data_out[:, 20] = pe.value(model.DAMFRAC[:])
    data_out[:, 21] = pe.value(model.ABATECOST[:])
    data_out[:, 22] = pe.value(params['sigma'[:]])
    data_out[:, 23] = pe.value(model.FORC[:])
    data_out[:, 24] = pe.value(params['forcoth'][:])
    data_out[:, 25] = pe.value(model.PERIODU[:])
    data_out[:, 26] = pe.value(model.C[:])
    data_out[0, 27] = pe.value(model.UTILITY)
    data_out[:, 28] = pe.value(params['etree'][:])
    data_out[:, 29] = pe.value(model.CCA[:])
    data_out[:, 30] = pe.value(model.CCATOT[:])
    data_out[:, 31] = pe.value(model.MAT[:])
    data_out[:, 32] = pe.value(model.E[:])
    data_out[:, 33] = pe.value(model.MU[:])
    data_out[:, 34] = pe.value(model.ML[:])
    data_out[:, 35] = pe.value(atfrac)
    data_out[:, 36] = pe.value(atfrac2010)

    varnames = [
        "Year",
        "Industrial Emissions GTCO2 per year", 
        "Atmospheric concentration C (ppm)",
        "Atmospheric Temperature ", 
        "Output Net Net) ", 
        "Climate Damages fraction output",
        "Consumption Per Capita ", 
        "Carbon Price (per t CO2)" , 
        "Emissions Control Rate" ,
        "Social cost of carbon",
        "Interest Rate " , 
        "Population" , 
        "TFP" , 
        "Output gross,gross" , 
        "Change tfp" , 
        "Capital" , 
        "s",
        "I" , 
        "Y gross net", 
        "damages" ,
        "damfrac" , 
        "abatement" ,
        "sigma" ,
        "Forcings" ,
        "Other Forcings" ,
        "Period utilty" ,
        "Consumption" , 
        "Objective",
        "Land emissions" , 
        "Cumulative ind emissions" , 
        "Cumulative total emissions" ,
        "Atmospheric concentrations Gt", 
        "Total Emissions GTCO2 per year",
        "Atmospheric concentrations upper", 
        "Atmospheric concentrations lower", 
        "Atmospheric fraction since 1850" , 
        "Atmospheric fraction since 2010"
    ]

    df_out = pd.DataFrame(data=data_out.T, index=varnames, columns=np.arange(params['nt'], dtype=int))
    df_out.to_csv('results.csv')