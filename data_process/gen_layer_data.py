import os

import numpy as np
from scipy.io import loadmat
from natsort import natsorted
src_dir = 'path/to/OCTA-500/GroundTruth/GT_Layers'
dst_root = 'path/to/layer_code'
names = natsorted(os.listdir(src_dir))

for name in names:
    data = loadmat(os.path.join(src_dir, name))
    print(data.keys())
    layers = data['Layer']
    print(layers.shape)
    _, d,w = layers.shape

    surfaces = np.zeros([8, d, w], int)
    surfaces[1:7] = layers
    surfaces[0,:,:]=0
    surfaces[7,:,:]=640
    os.makedirs(os.path.join(dst_root, name[:-4]), exist_ok=True)
    for i in range(d):
        np.save(os.path.join(dst_root, name[:-4], str(i+1)+'.npy'), surfaces[:,i,:])
    # exit()