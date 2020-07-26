import argparse
import os
import sys

import cv2
import numpy as np
from skimage import io

module_path = os.path.abspath(os.getcwd() + '/..')
if module_path not in sys.path:
    sys.path.append(module_path)

import configs
from src.inference import prediction, cosine_similarity, face_distance_to_conf
from src.model_exceptions import ExceptionType, ModelException
from src.utils import face_preprocess
from src.utils.face_model import FaceModel


class ImagesVerification(FaceModel):

    def __init__(self, params):
        super().__init__(load_mtcnn=True)

        self.origin_img = io.imread(params.origin_img)
        self.upload_img = io.imread(params.upload_img)

        self.embeddings_origin = np.array([])
        self.embeddings_upload = np.array([])

        try:
            aligned_images_origin, embeddings_origin = self.face_detection_steps(self.origin_img)
        except Exception as e:
            raise ModelException(ExceptionType.FACE_DETECTION_ERROR.value, "Can't find any faces in the original image")

        try:
            aligned_images_upload, embeddings_upload = self.face_detection_steps(self.upload_img)
        except Exception as e:
            raise ModelException(ExceptionType.FACE_DETECTION_ERROR.value, "Can't find any faces in the uploaded image")

        if len(aligned_images_origin) > 1:
            raise ModelException(ExceptionType.FACE_DETECTION_ERROR.value, "Too many faces in the original image")

        elif len(aligned_images_upload) > 1:
            raise ModelException(ExceptionType.FACE_DETECTION_ERROR.value, "Too many faces in the uploaded image")

        elif len(aligned_images_origin) == len(aligned_images_origin) == 1:
            self.embeddings_origin = embeddings_origin[0]
            self.embeddings_upload = embeddings_upload[0]

    def face_detection_steps(self, image):
        # Face detect and process tight cropping
        aligned_image = self.detect_face_and_align_image(image=image)
        if aligned_image is None:
            raise ModelException(ExceptionType.FACE_DETECTION_ERROR.value, "Can't find any faces in the input image")
        else:
            aligned_images = []
            embeddings_list = []
            for i, face in enumerate(aligned_image):
                if face is not None:
                    # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    aligned_images.append(face)
                    embeddings_list.append(self.get_emb_from_aligned_img(face))
            return aligned_images, embeddings_list

    def detect_face_and_align_image(self, image):
        ret = self.detector.detect_face(image, det_type=configs.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        multiple_faces = []
        nof_faces = bbox.shape[0]
        for i in range(nof_faces):
            bounding_box = bbox[i, 0:4]
            facial_landmarks = points[i, :].reshape((2, 5)).T
            # print(bbox)
            # print(points)
            nimg = face_preprocess.preprocess(image, bounding_box, facial_landmarks, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            multiple_faces.append(nimg)
        return multiple_faces

    def get_emb_from_aligned_img(self, aligned_image):
        image = aligned_image
        image = np.transpose(image, (2, 0, 1))
        face_embedding = self.get_embeddings(image)
        return face_embedding

    @staticmethod
    def embeddings_verification(src_emb, dst_emb):
        return prediction(src_emb, dst_emb), cosine_similarity(src_emb, dst_emb)

    def get_result(self):

        if self.embeddings_origin.size == 0 or self.embeddings_upload.size == 0:
            result_dict = {"same_person": 0,
                           "multiple_faces": True,
                           }
            return result_dict
        else:
            verification_results = self.embeddings_verification(self.embeddings_origin, self.embeddings_upload)

            result_dict = {"same_person": verification_results[0],
                           "cosine_similarity": verification_results[1],
                           "confidence": face_distance_to_conf(face_distance=verification_results[1])}

        return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face images verification')
    parser.add_argument("-o", "--origin-img", default=None, type=str, required=True,
                        help="URL of original face image")
    parser.add_argument("-u", "--upload-img", default=None, type=str, required=True,
                        help="URL of upload face image")
    parser = parser.parse_args()

    result = ImagesVerification(parser).get_result()

    print(f"[VERIFICATION] {result}")
