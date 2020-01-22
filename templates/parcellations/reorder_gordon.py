'''
Sorts parcels according to hemi - RSN - y(post->ant)
'''
import os
import pandas as pd
import nibabel as nb
import numpy as np
import shutil

parcels_filename = 'Gordon_2014_Parcels/Parcels_MNI_111.nii'
labels_filename = 'Gordon_2014_Parcels/Parcels.xlsx'

sorted_parcels_filename = 'Gordon_2014_Parcels/Parcels_MNI_111_sorted.nii.gz'
sorted_labels_filename = 'Gordon_2014_Parcels/Parcels_sorted.xlsx'

df = pd.read_excel(labels_filename, index='ParcelID')

# get centroids as float numbers
cent = df['Centroid (MNI)'].values
x, y, z = [], [], []
for c in cent:
    _x, _y, _z = c.split()
    x.append(float(_x))
    y.append(float(_y))
    z.append(float(_z))
df['x'] = x
df['y'] = y
df['z'] = z

df = df.sort(['Hem', 'Community', 'y'])

df['ParcelID_sorted'] = range(1, len(df)+1)
df.to_excel(sorted_labels_filename)


parc_img = nb.load(parcels_filename)
parc_data = parc_img.get_data()

id_mapping_list = df[['ParcelID', 'ParcelID_sorted']].values
id_mapping = {}
for k, v in id_mapping_list:
    id_mapping[k] = v

sorted_data = np.zeros_like(parc_data)
for old_id, new_id in id_mapping.items():
    sorted_data[parc_data==old_id] = new_id


sorted_img = nb.Nifti1Image(sorted_data, parc_img.get_affine(), parc_img.get_header())
sorted_img.to_filename(sorted_parcels_filename)

