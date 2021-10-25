#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, concatenate
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np
import pandas as pd
import json
import os.path
import tifffile as tiff
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize'] = (8, 6)
import numpy as np
import imageio


# # Create u-net model
# 

# In[3]:


# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

def unet_model(n_classes=5, im_sz=160, n_channels=8, n_filters_start=32, growth_factor=2, upconv=True,
               class_weights=[0.2, 0.3, 0.1, 0.1, 0.3]):
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    conv1 = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv1)
    conv1 = Dropout(0.3)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    n_filters *= growth_factor
    conv2 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv2)
    conv2 = Dropout(0.3)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    n_filters *= growth_factor
    conv3 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv3)
    conv3 = Dropout(0.4)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    n_filters *= growth_factor
    conv4 = BatchNormalization()(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(pool3)
    conv4 = Dropout(0.4)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    n_filters *= growth_factor
    conv5 = BatchNormalization()(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(pool4)
    conv5 = Dropout(0.5)(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv5)

    n_filters //= growth_factor
    if upconv:
        up6 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    else:
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = BatchNormalization()(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(up6)
    conv6 = Dropout(0.4)(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv6)

    n_filters //= growth_factor
    if upconv:
        up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    else:
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(up7)
    conv7 = Dropout(0.4)(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv7)

    n_filters //= growth_factor
    if upconv:
        up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    else:
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(up8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(conv8)

    n_filters //= growth_factor
    if upconv:
        up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    else:
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = BatchNormalization()(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu',kernel_initializer='he_normal', padding='same')(up9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', kernel_initializer='he_normal',padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    def weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(class_weights))

    model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy)
    return model



# if __name__ == '__main__':
#     model = unet_model()
#     print(model.summary())
#     #plot_model(model, to_file='unet_model.png', show_shapes=True)
# 

# # Get random patch
# you can randomly rotate images before getting a patch.

# In[28]:


def rotate_img(img, r):
    # channels in img are last!!!
    # r is a transformation type (an integer from 0 to 7)
    reverse_x = r % 2 == 1         # True for r in [1,3,5,7]
    reverse_y = (r // 2) % 2 == 1  # True for r in [2,3,6,7]
    swap_xy = (r // 4) % 2 == 1    # True for r in [4,5,6,7]
    if reverse_x:
        img = img[::-1, :, :]
    if reverse_y:
        img = img[:, ::-1, :]
    if swap_xy:
        img = img.transpose([1, 0, 2])
    return img


# In[29]:


def get_rand_patch(img, mask, sz=160, augment=True):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz and img.shape[0:2] == mask.shape[0:2]
    xc = random.randint(0, img.shape[0] - sz)
    yc = random.randint(0, img.shape[1] - sz)
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]
    if augment:
        j = random.randint(0, 7)
        patch_img = rotate_img(patch_img, j)
        patch_mask = rotate_img(patch_mask, j)
    return patch_img, patch_mask


def get_patches(x_dict, y_dict, n_patches, sz=160):
    x = list()
    y = list()
    total_patches = 0
    while total_patches < n_patches:
        img_id = random.sample(x_dict.keys(), 1)[0]
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = get_rand_patch(img, mask, sz)
        x.append(img_patch)
        y.append(mask_patch)
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)


# # Train

# In[30]:


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x



N_BANDS = 8
N_CLASSES = 5  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.2, 0.3, 0.1, 0.1, 0.3]
N_EPOCHS = 500
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 100
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(2) for i in range(1, 25)]  # all availiable ids: from "01" to "24"


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread('./data/mband/{}.tif'.format(img_id)).transpose([1, 2, 0]))
        mask = tiff.imread('./data/gt_mband/{}.tif'.format(img_id)).transpose([1, 2, 0]) / 255
        train_xsz = int(3/4 * img_m.shape[0])  # use 75% of image as train and 25% for validation
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net()


# In[ ]:


#model.save_weights('model_weights.h5')


# In[ ]:


#model.load_weights('model_weights.h5')


# In[33]:


def predict(img, model, patch_sz=160, border_sz=20, n_classes=5, augment=True):
    ## model is a trained CNN    
    # border is a place around center in the patch where predictions are usually bad in u-nets
    img_height = img.shape[0]
    img_width = img.shape[1]
    n_channels = img.shape[2]

    # make extended img so that it contains integer number of crossed-by-border patches
    center_sz = patch_sz - 2 * border_sz
    npatches_vert = int(math.ceil((img_height - 2*border_sz)/center_sz))
    npatches_horizon = int(math.ceil((img_width - 2*border_sz)/center_sz))
    extended_height = 2*border_sz + center_sz * npatches_vert
    extended_width = 2*border_sz + center_sz * npatches_horizon
    ext_img = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_img[:img_height, :img_width, :] = img
    for i in range(img_height, extended_height):
        ext_img[i, :, :] = ext_img[2*img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_img[:, j, :] = ext_img[:, 2*img_width - j - 1, :]

    # now assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vert):
        for j in range(0, npatches_horizon):
            x0, x1 = i * center_sz, (i + 1) * center_sz + 2 * border_sz
            y0, y1 = j * center_sz, (j + 1) * center_sz + 2 * border_sz
            if augment:
                for r in range(8):
                    patches_list.append(rotate_img(ext_img[x0:x1, y0:y1, :], r))
            else:
                patches_list.append(ext_img[x0:x1, y0:y1, :])
    patches_arr = np.asarray(patches_list) # np.transpose(patches_list, (0, 1, 2, 3))
    # predictions:
    patches_predict = model.predict(patches_arr, batch_size=4)
    confidence_map_patch = np.full(shape=(patch_sz, patch_sz, n_classes), fill_value=0.1)  # low confidence for borders
    confidence_map_patch[border_sz:border_sz+center_sz, border_sz:border_sz+center_sz, :] = 1 # high confidence for center
    confidence_map_img = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    prd = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(0, patches_predict.shape[0]):  # for all predicted patches
        if augment:
            r = k % 8   # patch transformation type (0..7)
            i = k // 8 // npatches_horizon  # patch x-coordinate
            j = k // 8 % npatches_horizon   # patch y-coordinate
            x0, x1 = i * center_sz, (i + 1) * center_sz + 2 * border_sz
            y0, y1 = j * center_sz, (j + 1) * center_sz + 2 * border_sz
            confidence_map_img[x0:x1, y0:y1, :] += confidence_map_patch
            prd[x0:x1, y0:y1, :] += rotate_img(patches_predict[k, :, :, :], r) * confidence_map_patch
        else:
            i = k // npatches_horizon
            j = k % npatches_horizon
            x0, x1 = i * center_sz, (i + 1) * center_sz + 2 * border_sz
            y0, y1 = j * center_sz, (j + 1) * center_sz + 2 * border_sz
            confidence_map_img[x0:x1, y0:y1, :] += confidence_map_patch
            prd[x0:x1, y0:y1, :] += patches_predict[k, :, :, :] * confidence_map_patch
    prd /= confidence_map_img
    return prd[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict


if __name__ == '__main__':
    model = get_model()
    model.load_weights(weights_path)
    test_id = 'test'
    img = normalize(tiff.imread('data/mband/{}.tif'.format(test_id)).transpose([1,2,0]))   # make channels last
    mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])  # make channels first
    map = picture_from_mask(mask, 0.5)

    tiff.imsave('result.tif', (255*mask).astype('uint8'))
    tiff.imsave('map.tif', map)
