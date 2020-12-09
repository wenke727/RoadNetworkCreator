import json
import pandas as pd
import os

file = '/home/pcl/Data/lane_detection/label_data_0531.json'

df = pd.read_json( file, lines=True )


res = []
for f in ["label_data_0313.json", "label_data_0601.json", "label_data_0531.json"]:
    df = pd.read_json( os.path.join("/home/pcl/Data/lane_detection", f), lines=True )
    res.append(df)

df = pd.concat(res)


df.reset_index(drop=True, inplace=True)

df.loc[:, 'num'] = df.lanes.apply(lambda x: len(x))

df.num.value_counts()

