import json
import pandas as pd
import os


folder = '/home/pcl/Data/TuSimple/LaneDetection'
lst = [ x for x in os.listdir(folder) if x.__contains__('json') ]

res = []
for f in lst:
    df = pd.read_json( os.path.join("/home/pcl/Data/TuSimple/LaneDetection", f), lines=True )
    res.append(df)

df = pd.concat(res)


df.reset_index(drop=True, inplace=True)

df.loc[:, 'num'] = df.lanes.apply(lambda x: len(x))

df.num.value_counts()

