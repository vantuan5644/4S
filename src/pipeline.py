import argparse
import base64
import io
import os
import shutil
import sys
import time
from fnmatch import fnmatch

import cv2
import numpy as np
from PIL import Image

module_path = os.path.abspath(os.getcwd() + '/..')
if module_path not in sys.path:
    sys.path.append(module_path)

import configs
from src.inference import prediction, cosine_similarity, face_distance_to_conf
from src.model_exceptions import ExceptionType, ModelException
from src.utils import face_preprocess
from src.utils.face_model import FaceModel


class VerificationPipeline(FaceModel):
    """
    Convert a known ID's face image that have been encoded in base64 to an embedding vector.
    :return Verification accuracy
    """

    def __init__(self, params):
        super().__init__(load_mtcnn=True)
        self.uid = params.uid
        self.user_register = params.user_register
        self.input_img_path = params.input_img_path
        self.dump_to_files = params.dump_to_files
        self.base_64_string = params.base_64_string
        self.embeddings = np.array([])
        if self.uid is None:
            raise ModelException(ExceptionType.INPUT_ERROR, f"Input the user UID, current value = {self.uid}")
        # Get image from base64 string
        if self.base_64_string is not None:
            self.image = self.base64_to_img(self.base_64_string)
        elif self.input_img_path is not None:
            self.image = cv2.imread(self.input_img_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        else:
            raise ModelException(ExceptionType.INPUT_ERROR, f"Input the path to image or base64 encoded string")

        # Check whether user path existed or not
        user_path = os.path.join(configs.dataset_path, self.uid)
        self.user_path = user_path

        # Unique image_file_name
        # current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        millis = round(time.monotonic() * 1000)
        image_file_name = f"{self.uid}_{millis}"
        self.image_file_name = image_file_name

        if self.user_register:
            if os.path.isdir(user_path):
                raise ModelException(ExceptionType.REGISTRATION_ERROR, f"Current UID {self.uid} is existed")
            else:
                os.makedirs(user_path)
                self.image_path = os.path.join(user_path, f"{image_file_name}{configs.origin_suffix}.png")
                self.dump_to_file(self.image_path, self.image)

                try:
                    aligned_img_path, aligned_image, emb_path, embeddings = self.face_detection_steps(multiple=True)
                    if len(aligned_img_path) == 1:
                        self.dump_to_file(aligned_img_path[0], aligned_image[0])
                        self.dump_to_file(emb_path[0], embeddings[0])
                    else:
                        if os.path.exists(self.user_path):
                            shutil.rmtree(self.user_path)
                        raise ModelException(ExceptionType.REGISTRATION_ERROR.value, "Too many faces in register image")

                except Exception as e:
                    if os.path.exists(self.user_path):
                        shutil.rmtree(self.user_path)
                    raise e

        else:
            if not os.path.isdir(user_path):
                raise ModelException(ExceptionType.VERIFICATION_ERROR, f"Current UID {self.uid} is not existed")
            else:
                self.image_path = os.path.join(user_path, f"{image_file_name}{configs.uploaded_suffix}.png")
                aligned_img_path, aligned_image, emb_path, embeddings = self.face_detection_steps(multiple=True)
                self.embeddings = embeddings[0]
                if len(aligned_img_path) == 1:
                    # Verification with predefined threshold
                    results = self.embeddings_verification(embeddings[0])
                    if results[0] == 1:
                        # Same person
                        if self.dump_to_files:
                            self.dump_to_file(self.image_path, self.image)
                            self.dump_to_file(aligned_img_path[0], aligned_image[0])
                            self.dump_to_file(emb_path[0], embeddings[0])
                else:
                    for i, face in enumerate(aligned_image):
                        results = self.embeddings_verification(embeddings[i])
                        if results[0] == 1:
                            # Same person
                            if self.dump_to_files:
                                self.dump_to_file(self.image_path, self.image)
                                self.dump_to_file(aligned_img_path[i], aligned_image[i])
                                self.dump_to_file(emb_path[i], embeddings[i])
                                self.embeddings = embeddings[i]

    def get_result(self):
        self.face_detection_steps(multiple=True)
        return self.embeddings

    def face_detection_steps(self, multiple=False):
        # Face detect and process tight cropping
        aligned_image = self.detect_face_and_align_image(multiple=multiple)
        if aligned_image is None:
            raise ModelException(ExceptionType.FACE_DETECTION_ERROR.value, "Can't find any faces in the input image")
        else:
            if not multiple:
                aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
                aligned_img_path = os.path.join(self.user_path, f"{self.image_file_name}{configs.aligned_suffix}.png")
                # Get the embeddings vector
                embeddings = self.get_emb_from_aligned_img(aligned_image)
                emb_path = os.path.join(self.user_path, f"{self.image_file_name}{configs.embedding_suffix}.npy")
                return aligned_img_path, aligned_image, emb_path, embeddings
            else:
                aligned_img_paths = []
                aligned_images = []
                emb_paths = []
                embeddings_list = []
                for i, face in enumerate(aligned_image):
                    if face is not None:
                        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        aligned_images.append(face)
                        embeddings_list.append(self.get_emb_from_aligned_img(face))
                        aligned_img_paths.append(
                            os.path.join(self.user_path, f"{self.image_file_name}_{i}{configs.aligned_suffix}.png"))
                        emb_paths.append(
                            os.path.join(self.user_path, f"{self.image_file_name}_{i}{configs.embedding_suffix}.npy"))
                return aligned_img_paths, aligned_images, emb_paths, embeddings_list

    @staticmethod
    def dump_to_file(file_path, file):
        if ".npy" in file_path:
            np.save(file_path, file)
        else:
            cv2.imwrite(file_path, cv2.cvtColor(file, cv2.COLOR_BGR2RGB))

    @staticmethod
    def img_to_base64(img_path, decode='utf-8'):
        with open(img_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read())
        return encoded_string.decode(decode)

    @staticmethod
    def base64_to_img(base64_string):
        img_data = base64.b64decode(str(base64_string))
        image = Image.open(io.BytesIO(img_data))
        return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    def detect_face_and_align_image(self, multiple):
        ret = self.detector.detect_face(self.image, det_type=configs.det)
        if ret is None:
            return None
        bbox, points = ret
        if bbox.shape[0] == 0:
            return None
        if not multiple:
            bbox = bbox[0, 0:4]
            points = points[0, :].reshape((2, 5)).T
            # print(bbox)
            # print(points)
            nimg = face_preprocess.preprocess(self.image, bbox, points, image_size='112,112')
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            # aligned = np.transpose(nimg, (2, 0, 1))
            # return aligned
            return nimg
        else:
            multiple_faces = []
            nof_faces = bbox.shape[0]
            for i in range(nof_faces):
                bounding_box = bbox[i, 0:4]
                facial_landmarks = points[i, :].reshape((2, 5)).T
                # print(bbox)
                # print(points)
                nimg = face_preprocess.preprocess(self.image, bounding_box, facial_landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                multiple_faces.append(nimg)
            return multiple_faces

    def get_emb_from_aligned_img(self, aligned_image):
        image = aligned_image
        image = np.transpose(image, (2, 0, 1))
        face_embedding = self.get_embeddings(image)
        return face_embedding

    def embeddings_verification(self, dst_emb):

        pattern = "*.npy"
        emb_list = []
        for path, subdirs, files in os.walk(self.user_path):
            for name in files:
                if fnmatch(name, pattern):
                    emb_list.append(os.path.join(path, name))
        src_emb = np.random.choice(emb_list)
        # TODO: Use a fixed source embeddings
        src_emb = np.load(src_emb)
        return prediction(src_emb, dst_emb), cosine_similarity(src_emb, dst_emb)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='User facial image registration or verification')
    parser.add_argument("--uid", default=None, type=str, required=True,
                        help="User unique ID")
    parser.add_argument("--base-64-string", default=None, type=str,
                        help="Base64 encode of the input image")
    parser.add_argument("--input-img-path", default=None, type=str,
                        help="Input img absolute path")
    parser.add_argument("--user-register", default=False, type=str2bool, required=True,
                        help="True for Registration, False for Verification")
    parser.add_argument("--dump-to-files", default=False, type=str2bool,
                        help="Enable images storage")
    parser = parser.parse_args()

    pipeline = VerificationPipeline(parser)

    # _, _, _, embeddings = pipeline.face_detection_steps()
    embeddings = pipeline.get_result()
    if not parser.user_register:
        verification_results = pipeline.embeddings_verification(embeddings)

        result = {"same_person": verification_results[0],
                  "cosine_similarity": verification_results[1],
                  "confidence": face_distance_to_conf(face_distance=verification_results[1])}

        print(f"[VERIFICATION] {result}")
    else:
        print(f'[REGISTRATION] Successfully registered UID = {parser.uid}')
