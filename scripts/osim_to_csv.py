import pandas as pd
import csv

def to_csv(motion_file_path: str, csv_file_path: str, normalized: str, delimiter=",", time_step=0.005, header_info_len=10) -> None:
    """method to transform opensim inverse kinematics or inverse dynamics file to csv format
       @param motion_file_path: path of opensim output file
       @param csv_file_path: path to save new csv
       @param normalized: path to save normalized csv (start at t=0)
       @param delimiter: delimiter to add in created csv file (,)
       @param time_step: timestep used in csv file
       @param header_info_len: length of the header in opensim files """

    with open(motion_file_path, "r") as motion_file:
        motion_data = [line.strip().split(delimiter) for line in motion_file]

    with open(csv_file_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerows(motion_data[header_info_len:-1])  # skip headings produced by opensim

    df = pd.read_csv(csv_file_path, delim_whitespace=True)

    df['time'] = [x * time_step for x in range(len(df))]
    df.to_csv(normalized, index=False, sep=',')


if __name__ == '__main__':
    ik_mot_file = "ADD PATH"
    id_sto_file = "ADD PATH"

    ik_csv_file = "ADD PATH"
    id_csv_file = "ADD PATH"

    #Care opensim sometimes doesn't follow the same positive/negative as we expect in the field (knee).
    #Make sure these align in your inverse kinematics file.
    to_csv(motion_file_path=ik_mot_file, csv_file_path=ik_csv_file, normalized=ik_csv_file, time_step=0.005, header_info_len=10)
    to_csv(motion_file_path=id_sto_file, csv_file_path=ik_csv_file, normalized=ik_csv_file, time_step=0.005, header_info_len=6)