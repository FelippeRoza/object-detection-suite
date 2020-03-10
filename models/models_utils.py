import urllib.request
import shutil
import os
import numpy as np
import tensorflow as tf


def imread(image_path):
    return tf.image.decode_image(open(image_path, 'rb').read(), channels=3)


def load_weights(model, weights_path = None):
    """Load model with trained weights.

    model: model
    weights_path: directory where the model is saved
    """

    if not  weights_path:
        if model.name == 'yolov3':
            weights_path = './data/weights/yolov3_coco.h5'
            url = 'https://pjreddie.com/media/files/yolov3.weights'
    if not os.path.exists(weights_path):
        # if weights are not in folder, download them
        print('Downloading weights from', url)
        temp_weights = './data/weights/yolov3_coco.weights'
        with urllib.request.urlopen(url) as resp, open(temp_weights, 'wb') as out:
            shutil.copyfileobj(resp, out)
        convert_darknet(model.keras_model, temp_weights, save_as = weights_path)

    model.load_weights(weights_path, by_name = True)

    return model


def convert_darknet(model, weights_path, save_as):
    """Load darknet weights in a keras model for yolo/yolo-tiny.

    model: yolo keras model
    weights_path: weights file name
    """

    wf = open(weights_path, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    layers =  ['yolo_darknet', 'yolo_conv_0','yolo_output_0','yolo_conv_1',
               'yolo_output_1', 'yolo_conv_2', 'yolo_output_2',]

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            print("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

    # save .h5 file and remove .weights one
    model.save_weights(save_as)
    print('h5 model saved')
    os.remove(weights_path)
