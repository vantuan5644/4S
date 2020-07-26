import itertools
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imutils import paths
from sklearn.metrics import classification_report

import configs
from src.face_detection.mtcnn_mxnet import MTCNN
from src.make_embeddings import MakeEmbeddings


def emb_distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


# def verification_inference(src, neg, plot=False, threshold=1.22):
#     src_path = np.random.choice(os.listdir(src))
#     src_img = cv2.imread(os.path.join(src, src_path), cv2.IMREAD_COLOR)
#
#     pos_path = os.listdir(src)
#     pos_path.remove(src_path)
#     pos_path = np.random.choice(pos_path)
#     pos_img = cv2.imread(os.path.join(src, pos_path), cv2.IMREAD_COLOR)
#
#     neg_path = np.random.choice(os.listdir(neg))
#     neg_img = cv2.imread(os.path.join(neg, neg_path), cv2.IMREAD_COLOR)
#
#     # bbox, landmark = model.detect(img, threshold=0.5, scale=1.0)
#     src_emb = model.get_embedding(src_img)
#     pos_emb = model.get_embedding(pos_img)
#     neg_emb = model.get_embedding(neg_img)
#     return src_emb, pos_emb, neg_emb


def cosine_similarity(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# def generate_embedding_vectors(path='/datasets/custom_dataset'):
#     class_names = os.listdir(path)
#     embeddings_list = []
#     class_name_list = []
#     file_path_list = []
#     for class_name in class_names:
#         class_path = os.path.join(path, class_name)
#         for file_name in os.listdir(class_path):
#             file_path = os.path.join(class_path, file_name)
#             img = cv2.imread(file_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = np.transpose(img, (2, 0, 1))
#
#             embeddings_list.append(model.get_embedding(img))
#             class_name_list.append(class_name)
#             file_path_list.append(file_path)
#             print(file_path)
#
#     data = {"embeddings": embeddings_list, "class_name": class_name_list, "file_path": file_path_list}
#     f = open('dataset_embs.pkl', "wb")
#     f.write(pickle.dumps(data))
#     f.close()


def calculate_threshold(pkl_file=configs.pkl_path):
    with open(pkl_file, 'rb') as f:
        dataset_embs = pickle.load(f)

    intra_class = []
    inter_class = []
    names = dataset_embs['id']
    embeddings = dataset_embs['embedding']
    for i, name_1 in enumerate(names):
        for j, name_2 in enumerate(names):
            if i != j and name_1 == name_2:
                intra_class.append(cosine_similarity(embeddings[i], embeddings[j]))
            elif name_1 != name_2:
                inter_class.append(cosine_similarity(embeddings[i], embeddings[j]))

    plt.hist(intra_class, bins=100)
    plt.hist(inter_class, bins=100)
    plt.show()


def get_list_of_files(dir_path):
    # create a list of file and sub directories
    # names in the given directory
    list_of_file = os.listdir(dir_path)
    all_files = list()
    # Iterate over all the entries
    for entry in list_of_file:
        # Create full path
        full_path = os.path.join(dir_path, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(full_path):
            all_files = all_files + get_list_of_files(full_path)
        else:
            all_files.append(full_path)

    return all_files


def prediction(emb_1, emb_2, threshold=configs.threshold):
    # 1: same class
    # 0: different classes
    return 1 if cosine_similarity(emb_1, emb_2) <= threshold else 0


def generate_combinations(path=configs.dataset_path):
    combinations = []

    if configs.emb_storage_method == 'pkl':
        files_list = paths.list_images(path, contains=configs.aligned_suffix)
        # class_name = os.path.split(os.path.split(file_path)[0])[1]

        pkl_file = configs.pkl_path
        with open(pkl_file, 'rb') as f:
            dataset_embs = pickle.load(f)

        for file_path_1, file_path_2 in itertools.combinations(files_list, 2):
            class_name_1 = os.path.split(os.path.split(file_path_1)[0])[1]
            class_name_2 = os.path.split(os.path.split(file_path_2)[0])[1]

            label = 1 if class_name_1 == class_name_2 else 0

            emb_1 = dataset_embs['embedding'][dataset_embs['image_path'].index(file_path_1)]
            emb_2 = dataset_embs['embedding'][dataset_embs['image_path'].index(file_path_2)]

            pred = prediction(emb_1, emb_2)
            combinations.append({'file_path_1': file_path_1,
                                 'file_path_2': file_path_2,
                                 'label': label,
                                 'pred': pred})
    elif configs.emb_storage_method == 'npy':
        files_list = paths.list_files(path, validExts=('.npy'))
        for file_path_1, file_path_2 in itertools.combinations(list(files_list), 2):
            class_name_1 = os.path.split(os.path.split(file_path_1)[0])[1]
            class_name_2 = os.path.split(os.path.split(file_path_2)[0])[1]

            label = 1 if class_name_1 == class_name_2 else 0

            emb_1 = np.load(file_path_1)
            emb_2 = np.load(file_path_2)

            pred = prediction(emb_1, emb_2)
            combinations.append({'file_path_1': file_path_1,
                                 'file_path_2': file_path_2,
                                 'cosine_similarity': cosine_similarity(np.load(file_path_1), np.load(file_path_2)),
                                 'label': label,
                                 'pred': pred})

    combinations = pd.DataFrame(combinations)
    combinations.to_csv('combinations.csv', index=False)


def face_distance_to_conf(face_distance, face_match_threshold=configs.threshold):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


if __name__ == '__main__':
    total_images, successfully_aligned = MTCNN().dataset_alignment(configs.dataset_path)
    print(f"[INFO] Nof total images: {total_images}, successfully aligned: {successfully_aligned}")

    MakeEmbeddings().dataset_processing(configs.dataset_path)

    # calculate_threshold()

    # generate_combinations()
    # result = pd.read_csv('combinations.csv')
    # print(classification_report(result['label'], result['pred']))
