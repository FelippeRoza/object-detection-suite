# from utils.load_model import load_model
from utils import dataset, visualize
import tensorflow as tf
# import cv2
from models import yolov3, models_utils

# load model
model = yolov3.YoloV3(classes = 80)
model = models_utils.load_weights(model)

#load class mapping for correspondent dataset on which model was traned
class_names = dataset.class_mapping(dataset = 'coco')

# load image
image_path = './data/images/sample.jpg'
image = models_utils.imread(image_path)

# perform object detection
det = model.detect(image)

# draw detections and save image
out_img = visualize.draw_from_detection(image, det, class_names)
tf.keras.preprocessing.image.save_img('./data/output/output.png', out_img)
