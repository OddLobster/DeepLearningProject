#%%
import pandas as pd
import numpy as np

#%%
# analyze original data
# filepath = "../data/metr-la.h5"

# df = pd.read_hdf(filepath)

# df.columns
# len(df.index)

# read pickle file
df_pickle = pd.read_pickle("./sensor_graph/adj_mx.pkl")

#%%
# Tokyo Dataset
PATH = "../tokyo/"
months = [10, 11, 12]
agg_data = []

for m in months:
    print(f"working on {m}th month")
    # 3 months, 10min timestamps, 2841 sensors
    data = pd.read_csv(PATH + f"expy-tky_2021{m}.csv")

    # check if 2841 sensors it occurs the same ==> it is.
    recur_flag = True
    for i in range(len(data.index)//2841-2):
        if not np.array_equal(data["linkid"][i*2841:(i+1)*2841].to_numpy(), data["linkid"][(i+1)*2841:(i+2)*2841].to_numpy()):
            recur_flag = False
    
    if not recur_flag:
        print("no recurrence")
        break

    # reshaping the dataframe (num sensors = 2841)
    npdata = data[["speed"]].to_numpy()
    npdata = npdata.reshape((-1, 2841))

    indices = pd.unique(data["timestamp"]).astype('datetime64[ns]')
    columns = data["linkid"][:2841]

    newdata = pd.DataFrame(npdata, indices, columns)

    agg_data.append(newdata)

#%%
# concat 3 months
final_data = pd.concat(agg_data, axis=0)
final_data = final_data * 0.621371
#%%
# save it to hd5
final_data.to_hdf(PATH + "metr-tokyo.h5", key='df', mode="w")



#%%%%%%%%%%%%%%%%%%%%%%%
# read graph_sensor_locations.csv
PATH = "../tokyo/"
df_sensors = pd.read_csv(PATH + "graph_sensor_locations_tokyo.csv")
df_lasensors = pd.read_csv(PATH + "..\data\sensor_graph\graph_sensor_locations.csv")

# generate graph_sensor_ids_tokyo.csv
np.savetxt(PATH+"graph_sensor_ids_tokyo.txt", df_sensors["sensor_id"].to_numpy()[None, :], delimiter=",", fmt="%d")

#%%
# generate distances_tokyo.csv
df_dist = pd.read_csv(PATH + "..\data\sensor_graph\distances_la_2012.csv")
#%%
import geopy
1201054,1201066,2610.9

825515	825513

df_lasensors
df_lasensors.loc[df_lasensors['1201054']
geopy.distance.geodesic(coords_1, coords_2)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%5
# open adjacent npz file
data = np.load('expy-tky_adjdis.npy')