from os.path import join
from itertools import product
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
import numpy as np
import cv2

class VOC2012(Dataset):
    def __init__(self, voc2012_basedir, mode, input_shape=(224, 224)):
        super().__init__()
        self.voc2012_basedir = voc2012_basedir
        self.mode = mode
        self.image_dir = join(self.voc2012_basedir, 'JPEGImages')
        self.input_shape = input_shape
        self.labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor', 'void']
        self.annotations_dir = join(self.voc2012_basedir, 'Annotations')

    def get_image(self, image_id):
        path = join(self.image_dir, "{}.jpg".format(image_id))
        return cv2.imread(path)[:,:,::-1]

class VOC2012ClassSegmentation(VOC2012):
    def __init__(self, voc2012_basedir, mode='train', mask_shape=(56, 56)):
        super().__init__(voc2012_basedir, mode)
        self.color_map = self.create_color_map()
        self.class_mask_dir = join(self.voc2012_basedir, 'SegmentationClass')
        self.image_ids = self.get_image_ids()
        self.mask_shape = mask_shape

    def get_image_ids(self):
        segmentation_file = join(self.voc2012_basedir, 'ImageSets',
                'Segmentation', '{}.txt'.format(self.mode))

        with open(segmentation_file) as f:
            return f.read().splitlines()

    def __len__(self):
        return len(self.train_image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        return self.getitem(image_id)

    def getitem(self, image_id):
        image = self.get_image(image_id)
        scales = ( self.input_shape[0] / image.shape[0], self.input_shape[1] / image.shape[1] )
        image = self.resize(image)
        bboxes, labels = self.get_bboxes(image_id, scales)

        original_mask = self.get_mask(image_id)
        class_mask = self.convert_color_mask_to_class_mask(original_mask)
        bbox_masks = self.get_masks_by_bboxes_and_labels(class_mask, bboxes, labels)
        return image, bboxes, labels, bbox_masks

    def get_masks_by_bboxes_and_labels(self, mask, bboxes, labels):
        masks = []
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            label = labels[i]
            # crop the bbox
            copy_mask = mask.copy()
            box_area = copy_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            # keep the value which is equal to label
            # which means set other value all to zero
            mask_index = box_area == label
            box_area[~mask_index] = 0

            # resize the mask to fixed size for feeding to the network
            # this step is reversible if we have the corresponding bounding box
            # and original image shape
            box_area_resized = cv2.resize(box_area, self.mask_shape)

            masks.append(box_area_resized)
        return np.stack(masks)

    def get_bboxes(self, image_id, scales=(1,1)):
        """get bounding boxes from annotations for given image id,
        the bounding box coordinate is [y1, x1, y2, x2]
        """
        annotation_file = join(self.annotations_dir, '{}.xml'.format(image_id))
        root = ET.parse(annotation_file).getroot()
        bboxes = []
        labels = []
        for obj in root.findall('object'):
            coordinates = []
            for index, name in enumerate(['ymin', 'xmin', 'ymax', 'xmax']):
                coordinates.append( int(int(obj.find('bndbox').find(name).text) * scales[index%2]) )
            label = obj.find('name').text
            bboxes.append(coordinates)
            labels.append(self.labels.index(label))
        return np.stack(bboxes), np.stack(labels)

    def extract_bboxes(self, masks):
        """get bounding boxes from masks
        """
        boxes = np.zeros([masks.shape[0], 4], dtype=np.int32)
        for i in range(masks.shape[0]):
            m = masks[0]
            # Bounding box.
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
            else:
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([x1, y1, x2, y2])
        return boxes.astype(np.int32)

    def resize(self, image):
        return cv2.resize(image, self.input_shape)

    def convert_color_mask_to_class_mask(self, mask):
        h, w, _ = mask.shape
        classed_mask = np.zeros( (h,w) )
        for pos in product(range(h), range(w)):
            classed_mask[pos] = np.argwhere(np.all( self.color_map == mask[pos], axis=1 ))[0][0]
        return classed_mask

    def get_mask(self, image_id):
        path = join(self.class_mask_dir, "{}.png".format(image_id))
        return cv2.imread(path)[:,:,::-1]

    def create_color_map(self, N=256, normalized=False):
        """See https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
        """

        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        dtype = 'float32' if normalized else 'uint8'
        cmap = np.zeros((N, 3), dtype=dtype)
        for i in range(N):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        cmap = np.concatenate( [cmap[:len(self.labels)-1], [cmap[-1]] ])
        cmap = cmap/255 if normalized else cmap
        return cmap