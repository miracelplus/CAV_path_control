import pandas as pd
import cvxpy as cp
import numpy as np
from conf import *
road_num = 15
csv_data = pd.read_csv('./network/toy_network.csv')
t0_list = csv_data["t0"].values
ca_list = csv_data["ca"].values
ca_inverse_power_list = (1/ca_list)**4
demand = 5

def CAV_UE(cav_path_distribution_ini):
    x = cp.Variable(road_num)
    cav_path_distribution = np.zeros((15,))
    cav_path_distribution[8] = cav_path_distribution_ini[0]
    cav_path_distribution[10] = cav_path_distribution_ini[1]
    UE_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x+cav_path_distribution) + cp.multiply(t0_list, cp.multiply(0.03*cp.power(x+cav_path_distribution, 5),ca_inverse_power_list))))
    constraints = [x >= 0, cp.sum(x+cav_path_distribution) == demand]
    prob = cp.Problem(UE_objective, constraints)
    result = prob.solve()
    if x.value is None:
        System_time = 100
    else:
        System_time = np.sum(t0_list * (x.value+cav_path_distribution) + t0_list * 0.15 * (x.value+cav_path_distribution)**5 * ca_inverse_power_list)
    return System_time

def CAV_UE_xy(myx, myy):
    x = cp.Variable(road_num)
    cav_path_distribution = np.zeros((15,))
    cav_path_distribution[8] = myx
    cav_path_distribution[10] = myy
    UE_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x+cav_path_distribution) + cp.multiply(t0_list, cp.multiply(0.03*cp.power(x+cav_path_distribution, 5),ca_inverse_power_list))))
    constraints = [x >= 0, cp.sum(x+cav_path_distribution) == demand]
    prob = cp.Problem(UE_objective, constraints)
    result = prob.solve()
    if x.value is None:
        System_time = 100
    else:
        System_time = np.sum(t0_list * (x.value+cav_path_distribution) + t0_list * 0.15 * (x.value+cav_path_distribution)**5 * ca_inverse_power_list)
    if myx+myy > control_ratio*demand:
        System_time = System_time + 1000*(myx+myy-control_ratio*demand)
    return -System_time


def CAV_UE_SO(mode="UE"):
    x = cp.Variable(road_num)
    UE_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x) + cp.multiply(t0_list, cp.multiply(0.03*cp.power(x, 5),ca_inverse_power_list))))
    SO_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x) + cp.multiply(t0_list, cp.multiply(0.15*cp.power(x, 5),ca_inverse_power_list))))
    constraints = [x >= 0, cp.sum(x) == demand]
    if mode == "UE":
        prob = cp.Problem(UE_objective, constraints)
    else:
        prob = cp.Problem(SO_objective, constraints)
    result = prob.solve()
    return result
