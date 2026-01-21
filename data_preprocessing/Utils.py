import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
import time
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.lines as mlines
import glob
import matplotlib.patches as patches
import re

### BASIC FUNCTIONS

# get recordingsId list from list of file names
def get_recordingsId_list(file_name_list):
    """
    This function takes a list of file names and extracts the recording IDs from them.
    """
    recordingId_list = file_name_list.copy()
    # remove 2 number from start of string
    string = recordingId_list[0][2:]
    for i in range(len(recordingId_list)):
        recordingId_list[i]=recordingId_list[i].replace(string, '')
    return recordingId_list

# load dfs from csv files from specific folde
def load_dfs(file_name_list, root_dir, folder_name):
    """
    This function loads dataframes from a list of file names.
    """
    os.chdir(root_dir)
    # change to data dir
    os.chdir(folder_name)
    dfs = []
    for i in range(len(file_name_list)):
        df=pd.read_csv(file_name_list[i])
        dfs.append(df)
    # change back to parent directory
    os.chdir('..')
    return dfs

# create a folder to save data in a specific root directory
def create_folder_to_save_data(root_dir, folder_name):
    """
    This function creates a folder to save data.
    """
    # change to data dir
    os.chdir(root_dir)
    # check if folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder Created:\n {folder_name} ")
    else:
        print(f"Folder: '{folder_name}' already exists")
    # change back to parent directory
    os.chdir('..')

# Save dfs to csv
def save_dfs(root_dir, dfs, recordingId_list, folder_name, file_name_suffix):
    """
    save each df in dfs to csv file.
    provide the recordingId list and suffix
    provide folder name to save in under the root_dir 
    """
    os.chdir(root_dir)
    # create folder to save files
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # change to working dir
    os.chdir(folder_name)
    # initialize a list of the files to return
    file_name_data_list=[]
    # iterate dfs and save each df to csv
    for x in tqdm( range(len(dfs)) , desc="Saving CSV files" ):
        # save as CSV file
        file_name=str(recordingId_list[x]) + file_name_suffix
        dfs[x].to_csv(file_name, index=False)
        # append the file name to the list
        file_name_data_list.append(file_name)
    
    print(f"files saved to: {os.getcwd()}")
    print("list of files saved:")
    print(file_name_data_list)
    # change back to parent directory
    os.chdir('..')
    return file_name_data_list

### Step 12

def normalized_distance_from_center(point_xy_m, circle_center_xy_m, radius_m):
    """
    Returns:
        -1 if point is outside the circle,
        0 if point is at the center,
        1 if point is on the edge,
        value between 0 and 1 if inside.
        OR CHANGE MANUALLY DEF FOR:
        0-Outside
        1-Inside
    """
    distance = np.linalg.norm(np.array(point_xy_m) - np.array(circle_center_xy_m))
    if distance > radius_m:
        # return -1  # Outside the circle
        return 0  # Outside the circle
    else:
        # return distance / radius_m  # 0 (center) to 1 (edge) [m]/[m]=[no units]
        return 1  # Inside the circle
    
# function to get the "pairs" column [row, class, ttc] from the csv file as a list of lists
def get_lists_of_lists_from_cell(cell_value): 
    """
    this function takes a string from a cell in the "pairs" 
    column and converts it into a list of lists.
    """   
    # Remove all whitespace and quotes
    cell_value = cell_value.replace(" ", "").replace("'", "").replace('"', '')
    # Split by comma, bracket, or whitespace to get all words/numbers
    tokens = re.split(r'[\[\],]+', cell_value)
    tokens = [t for t in tokens if t]  # remove empty strings
    # Convert tokens: try to int, then float, else keep as string, handle 'inf'
    def convert_token(x):
        if x.lower() == 'inf':
            return np.inf
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return x
    tokens = [convert_token(t) for t in tokens]
    # Group every 3 tokens into a list
    list_of_lists = [tokens[i:i+3] for i in range(0, len(tokens), 3)]
    return list_of_lists
