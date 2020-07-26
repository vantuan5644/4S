import mxnet as mx
import numpy as np
import sklearn.preprocessing as preprocessing

import configs
from src.face_detection.mtcnn_detector import MtcnnDetector


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, model_str, layer):
    _vec = model_str.split(',')
    assert len(_vec) == 2
    prefix = _vec[0]
    epoch = int(_vec[1])
    print('loading', prefix, epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    # model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class FaceModel:
    def __init__(self, load_mtcnn=False, load_ga=False):
        self.load_mtcnn = load_mtcnn
        self.load_ga = load_ga
        num_workers = configs.num_workers

        ctx = configs.gpu
        if ctx == -1:
            ctx = mx.cpu(0)
        else:
            ctx = mx.gpu(configs.gpu)

        _vec = configs.image_size.split(',')
        assert len(_vec) == 2
        image_size = (int(_vec[0]), int(_vec[1]))
        self.image_size = image_size

        self.model = None
        # self.ga_model = None
        if len(configs.model) > 0:
            self.model = get_model(ctx, image_size, configs.model, 'fc1')
        # if len(args.ga_model) > 0:
        #     self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

        self.threshold = configs.threshold

        if self.load_mtcnn:
            mtcnn_path = configs.mtcnn_path
            self.det_minsize = 50
            self.det_threshold = [0.6, 0.7, 0.8]
            # self.det_factor = 0.9
            if configs.det == 0:
                detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=num_workers,
                                         accurate_landmark=True,
                                         threshold=self.det_threshold)
            else:
                detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=num_workers,
                                         accurate_landmark=True,
                                         threshold=[0.0, 0.0, 0.2])
            self.detector = detector

        if self.load_ga:
            self.ga_model = get_model(ctx, image_size, configs.gender_age_model, 'fc1')

    def get_embeddings(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding

    def get_ga(self, aligned):
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.ga_model.forward(db, is_train=False)
        ret = self.ga_model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age
