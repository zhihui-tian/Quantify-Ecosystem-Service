"""
For:
Author: zhihui
Date: 2023/09/22
"""

import numpy as np
import geopandas as gpd
import pandas as pd


gdf=gpd.read_file("C:/Users/zhihui.tian/Desktop/FL/truth_data.shp")
gdf= gdf.loc[(gdf["lctype"] != '`7') & (gdf["lctype"] != "9'")]
class_names=gdf['lctype'].unique()
print('class names',class_names)
class_ids=np.arange(class_names.size)+1
print('class ids',class_ids)
df=pd.DataFrame({'labels':class_names,'id':class_ids})
df.to_csv("C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/class_lookup.csv")
print("gdf without ids",gdf.head())
gdf['id']=gdf['lctype'].map(dict(zip(class_names,class_ids)))
print('gdf with ids',gdf.head())

gdf_train=gdf.sample(frac=0.7)
gdf_test=gdf.drop(gdf_train.index)
print('gdf shape',gdf.shape,'training shape',gdf_train.shape,'test',gdf_test.shape)
gdf_train.to_file("C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/train.shp")
gdf_test.to_file("C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/test.shp")
