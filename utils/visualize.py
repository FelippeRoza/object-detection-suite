import cv2
import numpy as np


def draw_outputs(img, xmin, ymin, xmax, ymax, score, class_name):
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    img = cv2.putText(img, '{} {:.4f}'.format(class_name, score),
        (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_from_df(img, img_name, df):
    img = np.array(img)
    for index, row in df[df['image_name'] == img_name].iterrows():
        img = draw_outputs(img, row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                            row['scores'], row['object'])
    return img

def draw_from_detection(img, det, class_name):
    img = np.array(img)
    for detection in det:
        img = draw_outputs(img, detection[0][0], detection[0][1], detection[0][2],
                            detection[0][3], detection[2], class_name[detection[1]])
    return img
