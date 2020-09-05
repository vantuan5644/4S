# %%

# %%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import cv2
import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import sklearn
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
from skimage import transform as trans

# %%
# %%

sys.path.append(os.path.abspath('..'))

# %%

from models.mtcnn_detector import MtcnnDetector


def get_model(ctx, model):
    image_size = (112, 112)
    # Import ONNX model
    sym, arg_params, aux_params = import_model(model)
    # Define and binds parameters to the network
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


# %% md

### Download pre-trained face detection models

# %%

mtcnn_weights_path = '../weights/mtcnn-model'

# %%

for i in range(4):
    mx.test_utils.download(dirname=mtcnn_weights_path,
                           url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-0001.params'.format(
                               i + 1))
    mx.test_utils.download(dirname=mtcnn_weights_path,
                           url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}-symbol.json'.format(
                               i + 1))
    mx.test_utils.download(dirname=mtcnn_weights_path,
                           url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.caffemodel'.format(
                               i + 1))
    mx.test_utils.download(dirname=mtcnn_weights_path,
                           url='https://s3.amazonaws.com/onnx-model-zoo/arcface/mtcnn-model/det{}.prototxt'.format(
                               i + 1))

# %% md

### Configure face detection model for preprocessing

# %%

# Determine and set context
if len(mx.test_utils.list_gpus()) == 0:
    ctx = mx.cpu()
else:
    ctx = mx.gpu(0)
# Configure face detector
det_threshold = [0.6, 0.7, 0.8]
model_folder = os.path.abspath(mtcnn_weights_path)
detector = MtcnnDetector(model_folder=model_folder, ctx=ctx, num_worker=1, accurate_landmark=True,
                         threshold=det_threshold)


# %%

def preprocess(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    # Assert input shape
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # Do alignment using landmark points
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped

    # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret


def get_input(detector, face_img):
    # Pass input images through face detector
    ret = detector.detect_face(face_img, det_type=0)
    if ret is None:
        return None
    bbox, points = ret
    if bbox.shape[0] == 0:
        return None
    bbox = bbox[0, 0:4]
    points = points[0, :].reshape((2, 5)).T
    # Call preprocess() to generate aligned images
    nimg = preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned


#


# %%

def get_feature(model, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    model.forward(db, is_train=False)
    embedding = model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding


# %% md

### Download input images and prepare ONNX model

# %%

# Download first image
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/player1.jpg')
# Download second image
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/player2.jpg')
# Download onnx model
mx.test_utils.download('https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100.onnx',
                       dirname='../weights/arcface')
# Path to ONNX model
model_name = '../weights/arcface/resnet100.onnx'

# %%

# Load ONNX model
model = get_model(ctx, model_name)

# %% md

# %%

# Load first image
img1 = cv2.imread('player1.jpg')
# Display first image
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.show()

# %%
import time

start = time.time()
# Preprocess first image
pre1 = get_input(detector, img1)
print(f"MTCNN run in {time.time() - start}")
# Display preprocessed image
plt.imshow(np.transpose(pre1, (1, 2, 0)))
plt.show()
# Get embedding of first image

start = time.time()
out1 = get_feature(model, pre1)
print(f"Arcface run in {time.time() - start}")
# %%

# Load second image
img2 = cv2.imread('player2.jpg')
# Display second image
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.show()

# %%

# Preprocess second image
pre2 = get_input(detector, img2)
# Display preprocessed image
plt.imshow(np.transpose(pre2, (1, 2, 0)))
plt.show()
# Get embedding of second image
out2 = get_feature(model, pre2)

# %%

# Compute squared distance between embeddings
dist = np.sum(np.square(out1 - out2))
# Compute cosine similarity between embedddings
sim = np.dot(out1, out2.T)
# Print predictions
print('Distance = %f' % (dist))
print('Similarity = %f' % (sim))

# %%
