from CAV_UE import CAV_UE, CAV_UE_xy
import numpy as np
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
import pandas as pd
import matplotlib.pyplot as plt
from sko.tools import set_run_mode
from bayes_opt import BayesianOptimization
from conf import *
def plain_UE():
    cav_path_distribution_ini = np.array([0.0]*2)
    cav_path_distribution_ini[0] = 0.6 
    cav_path_distribution_ini[1] = 0
    system_time = CAV_UE(cav_path_distribution_ini=cav_path_distribution_ini)
    print(system_time)

def GA_CAV_included_best_UE(control_portion):
    constraint_ueq = [
        lambda x: sum(x) - 5*control_portion,
    ]
    # ga = SA(func=CAV_UE, x0=[0]*15, n_dim=15, size_pop=50, max_iter=200, lb=[0]*15, ub=[5]*15, precision=1e-7, constraint_eq=constraint_eq)
    # ga = PSO(func=CAV_UE, n_dim=15, size_pop=50, max_iter=200, lb=[0]*15, ub=[5]*15, precision=1e-7, constraint_eq=constraint_eq)
    set_run_mode(CAV_UE, 'multiprocessing')
    ga = GA(func=CAV_UE, n_dim=2, size_pop=50, max_iter=800, lb=[0]*2, ub=[5*control_portion]*2, precision=1e-5, constraint_ueq=constraint_ueq)
    best_x, best_y = ga.run()
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    return best_y

# plain_UE()

def BO_CAV_included_best_UE():
    # Bounded region of parameter space
    pbounds = {'myx': (0, 5*control_ratio), 'myy': (0, 5*control_ratio)}
    optimizer = BayesianOptimization(
        f=CAV_UE_xy,
        pbounds=pbounds,
        random_state=1,
    )
    optimizer.maximize(
        init_points=20,
        n_iter=30,
    )
    return optimizer.max["target"]

def GA_best_result():
    best_y_list = []
    global control_ratio
    for control_ratio in [0.01, 0.02, 0.05, 0.08, 0.1, 0.5, 0.7, 1.0]:
        best_y_list.append(GA_CAV_included_best_UE(control_ratio))
    print(best_y_list)

BO_CAV_included_best_UE()