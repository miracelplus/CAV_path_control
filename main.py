import pandas as pd 
csv_data = pd.read_csv('./network/toy_network.csv')
t0_list = csv_data["t0"].values
ca_list = csv_data["ca"].values
print("a")