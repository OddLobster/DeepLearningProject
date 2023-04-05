#%%
import pandas as pd
import numpy as np

# target dataset shape
# filepath = "data/metr-la.h5"

# df = pd.read_hdf(filepath)

# df.columns
# len(df.index)


#%%
# Tokyo Dataset
months = [10, 11, 12]
agg_data = []

for m in months:
    print(f"working on {m}th month")
    # 3 months, 10min timestamps, 2841 sensors
    data = pd.read_csv(f"expy-tky_2021{m}.csv")

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

    indices = pd.unique(data["timestamp"])
    columns = data["linkid"][:2841]

    newdata = pd.DataFrame(npdata, indices, columns)

    agg_data.append(newdata)

#%%
# concat 3 months
final_data = pd.concat(agg_data, axis=0)
final_data = final_data * 0.621371

#%%
# save it to hd5
final_data.to_hdf("Tokyo.h5", key='df')

