import argparse
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

mtcnn_path = os.path.join(ROOT_DIR, 'models/mtcnn')
insightface_path = os.path.join(ROOT_DIR, 'models/insightface/models/model-r100-ii/model') + ',0'
dataset_path = os.path.join(ROOT_DIR, 'datasets/custom_dataset_small')
pkl_path = os.path.join(ROOT_DIR, 'outputs/embeddings.pkl')
gender_age_model = os.path.join(ROOT_DIR, 'models/gamodel-r50/model') + ',0'
ethnics_model = os.path.join(ROOT_DIR, 'gender_age_ethnic/face_model.pkl')

origin_suffix = '_o'
aligned_suffix = '_a'
embedding_suffix = '_e'
uploaded_suffix = '_u'

# Argument of insightface model
image_size = '112,112'
model = insightface_path
gpu = -1
num_workers = 4
det = 0
flip = 0
threshold = 0.64
emb_storage_method = 'npy'