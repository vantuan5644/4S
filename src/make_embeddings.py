import os
import pickle

import cv2
import numpy as np
from imutils import paths

import configs
from src.utils import face_model


class MakeEmbeddings:

    def __init__(self):
        self.model = face_model.FaceModel(load_mtcnn=False)

    def get_emb_from_aligned_img(self, image_path):
        _vec = configs.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))

        image = cv2.imread(image_path)
        assert (image.shape[0] == image_size[0]) and (image.shape[1] == image_size[1])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        face_embedding = self.model.get_embeddings(image)

        return face_embedding

    def dataset_processing(self, dataset_path):
        image_paths = list(paths.list_images(dataset_path, contains=configs.aligned_suffix))

        successfully_processed = 0
        embedding_list = []
        id_list = []
        image_path_list = []

        for (i, image_path) in enumerate(image_paths):
            # print("[INFO] Processing image {}/{}, img_path = {}".format(i + 1, len(image_paths), image_path))
            id = image_path.split(os.path.sep)[-2]
            img_file_name = os.path.splitext(os.path.split(image_path)[1])[0].replace(configs.aligned_suffix, '')
            emb_path = img_file_name + configs.embedding_suffix + '.npy'
            emb_path = os.path.join(os.path.split(image_path)[0], emb_path)

            if not os.path.isfile(emb_path):
                emb = self.get_emb_from_aligned_img(image_path)
                if configs.emb_storage_method == 'pkl':
                    id_list.append(id)
                    embedding_list.append(emb)
                    image_path_list.append(os.path.abspath(image_path))
                    successfully_processed += 1
                elif configs.emb_storage_method == 'npy':
                    # Dump to file
                    np.save(emb_path, emb)
                    successfully_processed += 1
                    print(f"[INFO] Successfully converted, embedding path = {emb_path}")
            else:
                print(f"[INFO] Already converted, embedding path = {emb_path}")
                successfully_processed += 1

        print(f"Successfully converted {successfully_processed} / {len(image_paths)} face images")

        # Dump to file
        if configs.emb_storage_method == 'pkl':
            data = {"embedding": embedding_list, "id": id_list, "image_path": image_path_list}
            f = open(configs.pkl_path, "wb")
            f.write(pickle.dumps(data))
            f.close()


if __name__ == '__main__':
    MakeEmbeddings().dataset_processing(configs.dataset_path)
