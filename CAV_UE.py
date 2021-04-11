from numpy.core.defchararray import multiply
import pandas as pd
import cvxpy as cp
import numpy as np
road_num = 15
csv_data = pd.read_csv('./network/toy_network.csv')
t0_list = csv_data["t0"].values
ca_list = csv_data["ca"].values
ca_inverse_power_list = (1/ca_list)**4

A = np.array(ca_list)
h = np.array([0]*road_num)
G = -np.eye(road_num)
A = np.array([1]*road_num)
b = 5
x = cp.Variable(road_num)

# UE objective
UE_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x) + cp.multiply(t0_list, cp.multiply(0.03*cp.power(x, 5),ca_inverse_power_list))))
# SO objective
SO_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x) + cp.multiply(t0_list, cp.multiply(0.15*cp.power(x, 5),ca_inverse_power_list))))
constraints = [x >= 0, cp.sum(x) == b]
prob = cp.Problem(UE_objective, constraints)

result = prob.solve()
print(x.value)
System_time = np.sum(t0_list * x.value + t0_list * 0.15 * x.value**5 * ca_inverse_power_list)
print("a")

def CAV_UE(cav_path_distribution):
    UE_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x) + cp.multiply(t0_list, cp.multiply(0.03*cp.power(x, 5),ca_inverse_power_list))))
    SO_objective = cp.Minimize(cp.sum(cp.multiply(t0_list, x) + cp.multiply(t0_list, cp.multiply(0.15*cp.power(x, 5),ca_inverse_power_list))))
    constraints = [x >= cav_path_distribution, cp.sum(x) == b]
    prob = cp.Problem(UE_objective, constraints)
    result = prob.solve()
    print(x.value)
    
