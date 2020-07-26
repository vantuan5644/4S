import os
import cv2
import mxnet as mx
from imutils import paths

import configs
from src.face_detection.mtcnn_detector import MtcnnDetector
from src.libs.facenet import facenet
from src.utils import face_preprocess


class MTCNN:

    def __init__(self):

        mtcnn_path = configs.mtcnn_path
        num_workers = configs.num_workers
        ctx = configs.gpu

        if ctx == -1:
            ctx = mx.cpu(0)
        else:
            ctx = mx.gpu(configs.gpu)

        _vec = configs.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))

        self.det_minsize = 50
        self.det_threshold = [0.6, 0.7, 0.8]
        # self.det_factor = 0.9
        self.image_size = image_size

        if configs.det == 0:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=num_workers, accurate_landmark=True,
                                     threshold=self.det_threshold)
        else:
            detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=num_workers, accurate_landmark=True,
                                     threshold=[0.0, 0.0, 0.2])
        self.detector = detector

    def detect_face_img(self, face_img, multiple=False):
        ret = self.detector.detect_face(face_img, det_type=configs.det)
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
            nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
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
                nimg = face_preprocess.preprocess(face_img, bounding_box, facial_landmarks, image_size='112,112')
                nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                multiple_faces.append(nimg)
            return multiple_faces

    def dataset_alignment(self, input_dir):
        input_dir = os.path.expanduser(input_dir)
        # assert os.path.isdir(input_dir)

        nof_total_images = 0
        nof_successfully_aligned = 0

        image_paths = list(paths.list_images(input_dir))
        aligned_images = list(paths.list_images(input_dir, contains=configs.aligned_suffix))

        image_paths = [item for item in image_paths if item not in aligned_images]

        for (i, image_path) in enumerate(image_paths):
            nof_total_images += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            aligned_filename = os.path.join(os.path.split(image_path)[0], filename + configs.aligned_suffix + '.png')
            if not os.path.exists(aligned_filename):
                try:
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except (IOError, ValueError, IndexError) as e:
                    error_message = f"[ERROR] {e} on image: {image_path}"
                    print(error_message)
                else:
                    if img.ndim < 2:
                        print(f"[INFO] Unable to align gray image: {image_path}")
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                    img = img[:, :, 0:3]

                    results = self.detect_face_img(img)

                    if results is not None:
                        cropped_temp = results
                        scaled_temp = cv2.resize(cropped_temp, self.image_size)
                        nof_successfully_aligned += 1
                        cv2.imwrite(aligned_filename, scaled_temp)
                        print(f"[INFO] Successfully aligned: {image_path}")

                    else:
                        print(f"[INFO] No face found in this image: {image_path}")
            else:
                print(f"[INFO] Already aligned: {image_path}")
                nof_successfully_aligned += 1
        return nof_total_images, nof_successfully_aligned


# TODO: Enable logging error to file

if __name__ == '__main__':
    MTCNN = MTCNN()
    # input_path = configs.dataset_path
    # total_images, successfully_aligned = MTCNN.dataset_alignment(input_path)
    # print(f"[INFO] Nof total images: {total_images}, successfully aligned: {successfully_aligned}")


    # Multiple faces
    path = '/home/vantuan5644/CBI-dev/goodhuman-ai-face-matching/datasets/test_set/noise/multiple.jpg'
    MTCNN.detect_face_img(cv2.imread(path, cv2.IMREAD_UNCHANGED), multiple=True)