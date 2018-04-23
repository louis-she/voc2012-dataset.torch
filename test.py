import numpy as np
import dataset
from itertools import product
np.set_printoptions(threshold=np.nan)

ds = dataset.VOC2012Segmentation('./VOCdevkit/VOC2012')
mask = ds.get_mask('2008_002240')

h, w, _ = mask.shape
classed_mask = np.zeros( (h,w) )
for pos in product(range(h), range(w)):
    classed_mask[pos] = np.argwhere(np.all( ds.color_map == mask[pos], axis=1 ))[0][0]

print(classed_mask)