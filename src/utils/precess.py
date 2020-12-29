import os
import json

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)


def to_json():
    filename='道路.json'
    with open(filename,'w') as file_obj:
        json.dump(df.iloc[12],file_obj)