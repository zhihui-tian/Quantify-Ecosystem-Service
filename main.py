"""
For:
Author: zhihui
Date: 2023/09/22
"""

import numpy as np
import scipy
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, slic
import time
import scipy
from osgeo import gdal,ogr
from sklearn.ensemble import RandomForestClassifier
from pysnic.algorithms.snic import snic
from pysnic.algorithms.polygonize import polygonize
from pysnic.algorithms.ramerDouglasPeucker import RamerDouglasPeucker


Input_fn="C:/Users/zhihui.tian/Downloads/3/3_planet.tif"

driverTiff= gdal.GetDriverByName('GTiff')
naip_ds=gdal.Open(naip_fn)
nbands=naip_ds.RasterCount

band_data=[]
print('bands',naip_ds.RasterCount,naip_ds.RasterYSize,'rows','columns',
      naip_ds.RasterXSize)
for i in range(1,nbands+1):
    band=naip_ds.GetRasterBand(i).ReadAsArray()
    band_data.append(band)
band_data=np.dstack(band_data)
print(band_data.shape)

img=exposure.rescale_intensity(band_data)
# do segmentation
seg_start=time.time()
segments, distance_map, centroids=snic(img,5000,0.1)  
segments=np.array(segments)
print('segments complete',time.time()-seg_start)

def segment_features(segment_pixels):
    features=[]
    npixels, nbands= segment_pixels.shape
    for b in range(nbands):
        stats=scipy.stats.describe(segment_pixels[:,b])
        band_stats=list(stats.minmax)+list(stats)[2:]
        if npixels==1:
            band_stats[3]=0.0
        features += band_stats
    return features
obj_start=time.time()
segment_ids=np.unique(segments)
objects=[]
object_ids=[]
for id in segment_ids:
    segment_pixels=img[segments == id]
    #print('pixel for id',id,segment_pixels.shape)
    object_features=segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)

print('created',len(objects),'objects with',len(objects[0]),'variables in',time.time()-obj_start,'seconds')
# save segments to raster
segments_fn='C:/Users/zhihui.tian/Downloads/3/segments3_planet_5000.tif'
segments_ds=driverTiff.Create(segments_fn, naip_ds.RasterXSize,naip_ds.RasterYSize,
                              1,gdal.GDT_Float32)
segments_ds.SetGeoTransform(naip_ds.GetGeoTransform())
segments_ds.SetProjection(naip_ds.GetProjectionRef())
segments_ds.GetRasterBand(1).WriteArray(segments)
segments_ds=None



train_fn='C:/Users/zhihui.tian/Desktop/1/naip/m_2908222_sw_17_060_20211129/train.shp'
train_ds=ogr.Open(train_fn)
lyr=train_ds.GetLayer()
driver=gdal.GetDriverByName('MEM')
target_ds=driver.Create('',naip_ds.RasterXSize, naip_ds.RasterYSize,1,gdal.GDT_UInt16)
target_ds.SetGeoTransform(naip_ds.GetGeoTransform())
target_ds.SetProjection(naip_ds.GetProjection())
options=['ATTRIBUTE=id']
gdal.RasterizeLayer(target_ds,[1],lyr,options=options)
ground_truth=target_ds.GetRasterBand(1).ReadAsArray()


classes=np.unique(ground_truth)[1:]
print('class values',classes)

segments_per_class={}
for klass in classes:
    segments_of_class=segments[ground_truth == klass]
    segments_per_class[klass]=set(segments_of_class)
    print("Training segments for class",klass,":",len(segments_of_class))

intersection=set()
accum=set()

for class_segments in segments_per_class.values():
    intersection |=accum.intersection(class_segments)
    accum |=class_segments



train_img = np.copy(segments)
threshold = train_img.max() + 1

for klass in classes:
    class_label=threshold+klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img==segment_id]=class_label

train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

training_objects = []
training_labels = []

for k in classes:
    class_train_object = [value for i, value in enumerate(objects) if segment_ids[i] in segments_per_class[k]]
    training_labels += [k] * len(class_train_object)
    # add training_objects
    training_objects += class_train_object
    print('Training objecs for class', k, ':', len(class_train_object))

model = RandomForestClassifier(n_jobs=-1)
model.fit(training_objects, training_labels)
print('Fitting Random Forest Classifier')




predicted_scores=np.array([])
for i in range(len(objects)):
    a = np.array(objects[i])
    a[np.isnan(a)] = 0
    predicted_proba1 = model.predict_proba(np.array(a).reshape(-1,1).T)
    score=[105,97,36,30,62,46,46,3,51]
    predicted_score=np.dot(predicted_proba1,score)
    predicted_scores=np.append(predicted_scores,predicted_score)

print('Predicting Classifications')
clf = np.copy(segments_test)


for segment_id, k in zip(segment_ids, predicted_scores):
    clf[clf == segment_id] = k



print('Prediction applied to numpy array')


mask = np.sum(img, axis=2)
mask[mask > 0.0] = 1.0
mask[mask == 0.0] = -1.0
clf = np.multiply(clf, mask)
clf[clf < 0] = -9999.0


clf_ds = driverTiff.Create('C:/Users/zhihui.tian/Downloads/3/Classified_new_weight_planet_5000.tif', naip_test_ds.RasterXSize, naip_test_ds.RasterYSize,
                          1, gdal.GDT_Float32)
clf_ds.SetGeoTransform(naip_test_ds.GetGeoTransform())
clf_ds.SetProjection(naip_test_ds.GetProjectionRef())
clf_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
clf_ds.GetRasterBand(1).WriteArray(clf)
clf_ds = None

