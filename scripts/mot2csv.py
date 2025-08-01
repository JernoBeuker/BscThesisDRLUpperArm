import numpy as np
import pandas as pd


"""Script to convert opensim inverse kinematics output (.mot) to format for PPO (this repository)"""


is_radians = False
data = False
data_dict = {}
dir_name = "ik"
filename = "ik_camargo_13_760_17_566"
headers = []

with open(f"{dir_name}/{filename}.mot") as f:
    for line in f:
        if line.startswith("inDegrees=yes"):
            is_radians = False
        if data:
            values = [float(value) for value in line.strip().split()]
            for i, header in enumerate(headers):
                data_dict[header].append(values[i])
        if line.startswith("time"):
            headers = line.strip().split()
            for header in headers:
                data_dict[header] = []
            data = True

# center the training data to 0,Y,0
for pelvis in ["pelvis_tx", "pelvis_tz"]:
    if data_dict[pelvis][0] != 0:
        value_0 = data_dict[pelvis][0]
        for i in range(len(data_dict[pelvis])):
            data_dict[pelvis][i] -= value_0

# convert to radians
if not is_radians:
    for key, value in data_dict.items():
        if key in ["time", "pelvis_tx", "pelvis_ty", "pelvis_tz"]:
            continue
        data_dict[key] = np.radians(data_dict[key])

# calculate velocities
time = data_dict["time"][1]
for header in headers:
    if header in ["time"]:
        continue
    data_dict[f"{header}_vel"] = np.subtract(np.append(data_dict[header][1:], 0), data_dict[header]) / time

# write the csv
df = pd.DataFrame(data_dict)
df.to_csv(path_or_buf=f"{dir_name}/{filename}.csv", index=False)

nRows = df.shape[0]
mot_header = f"{filename}_edited\nversion=1\nnRows={nRows}\nnColumns=18\ninDegrees=no\n\nUnits are S.I. units (second, meters, Newtons, ...)\nIf the header above contains a line with 'inDegrees', this indicates whether rotational values are in degrees (yes) or radians (no).\n\nendheader\n"

with open(f"{dir_name}/{filename}_edited.mot", "w") as f:
    f.write(mot_header)
    f.write("\t\t".join(headers))
    f.write("\n")
    for i in range(nRows):
        row = []
        for header in headers:
            row.append(data_dict[header][i])
        f.write("\t\t")
        row_as_str = ["%.4f" % number for number in row]
        f.write("\t\t".join(row_as_str))
        f.write("\n")
