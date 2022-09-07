import math
import pickle
from detectron2.modeling.poolers import ROIPooler
import numpy as np
from joblib.numpy_pickle_utils import xrange
from tensorflow.keras.applications import Xception
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import build_backbone
from imutils.object_detection import non_max_suppression

CFG_PATH = 'C:/Users/Charlool/Desktop/cs_studies_carmeliol/' \
           'deep_learning_proj/detectron2/configs/COCO-Detection/rpn_R_50_FPN_1x.yaml'


def group_list(l, group_size):
    """
    :param l:           list
    :param group_size:  size of each group
    :return:            Yields successive group-sized lists from l.
    """
    for i in xrange(0, len(l), group_size):
        yield l[i:i + group_size]


class ObjectsExtractor:
    def __init__(self):
        cfg = get_cfg()
        cfg.merge_from_file(CFG_PATH)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/rpn_R_50_FPN_1x.yaml")
        self.alpha = 0.55
        # self.backbone = build_backbone(cfg)
        # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)
        # ------------ Model--------------- #
        self.model = Xception(weights='imagenet')
        # self.pooler_resolution = 14
        # self.canonical_level = 4
        # self.canonical_scale_factor = 2 ** self.canonical_level
        # self.pooler_scales = (1 / self.canonical_scale_factor,)
        # self.sampling_ratio = 0
        # self.roi_pooler = ROIPooler(
        #     output_size=self.pooler_resolution,
        #     scales=self.pooler_scales,
        #     sampling_ratio=self.sampling_ratio,
        #     pooler_type="ROIPool"
        # )


    def get_objects(self, img):
        # cnn_features = self.backbone(img[..., ::-1])
        # cnn_features_p3 = [cnn_features['p3']]
        outputs = self.predictor(img[..., ::-1])
        rois = []
        boxes = []
        roi = img.copy()
        for box in outputs['proposals'].proposal_boxes.to('cpu'):
            x1, y1, x2, y2 = box
            w, h = abs(x1 - x2), abs(y1 - y2)
            if w / float(img.shape[0]) < 0.1 or h / float(img.shape[1]) < 0.1:
                continue
            roi = img[int(y1): int(y2), int(x1):int(x2)]

            # Resize it to fit the input requirements of the model
            roi = cv2.resize(roi, (299, 299))
            # Further preprocessing
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            rois.append(roi)
            boxes.append(box)

        # Convert ROIS list to arrays for predictions
        # region_feature_matrix = self.roi_pooler(cnn_features_p3, boxes)
        #
        # rf_C = region_feature_matrix.shape[1]
        # rf_W = region_feature_matrix.shape[2]
        # rf_H = region_feature_matrix.shape[3]
        input_array = np.array(rois)
        print("Input array shape is ;", input_array.shape)

        # ---------- Make Predictions -------#
        input_arrays = group_list(input_array, 16)
        start = True
        for array in input_arrays:
            if not start:
                preds = np.append(preds, self.model.predict(array))
            else:
                start = False
                preds = self.model.predict(array)
        # preds = self.model.predict(input_array)
        preds = imagenet_utils.decode_predictions(preds.reshape((len(input_array), 1000)), top=1)

        # Initiate the dictionary
        objects = {}
        for (i, pred) in enumerate(preds):

            # extract the prediction tuple
            # and store it's values
            iD = pred[0][0]
            label = pred[0][1]
            prob = pred[0][2]

            if prob >= self.alpha:
                # grab the bounding box associated
                # with the prediction and
                # convert the coordinates
                box = boxes[i]

                # create a tuble using box and probability
                value = objects.get(label, [])

                # append the value to the list for the label
                value.append((box, prob))

                # Add this tuple to the objects dictionary
                # that we initiated
                objects[label] = value

        # Loop through the labels
        # for each label apply the non_max_suppression
        cropped_objects = []
        for label in objects.keys():
            # clone the original image so that we can
            # draw on it
            img_copy = img.copy()
            boxes = np.array([pred[0].numpy() for pred in objects[label]])
            proba = np.array([pred[1] for pred in objects[label]])
            boxes = non_max_suppression(boxes, proba)

            # Now unpack the co-ordinates of the bounding box
            (startX, startY, endX, endY) = boxes[0]
            cropped_object = img[startY: endY, startX:endX]
            cropped_objects.append(cropped_object)
            # plt.imshow(cropped_object[..., ::-1])
            # plt.show()

            # # Draw the bounding box
            # cv2.rectangle(img_copy, (startX, startY),
            #               (endX, endY), (0, 255, 0), 2)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            #
            # # Put the label on the image
            # cv2.putText(img_copy, label, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0), 2)
            #
            # # Show the image
            # img_copy = cv2.resize(img_copy, (500, 500))
            # cv2.imshow("Regional proposal object detection", img_copy)
            # cv2.waitKey(0)
        return cropped_objects


# xxx = ObjectsExtractor()
# img = cv2.imread("./N0000500.jpg")
# img1 = cv2.imread("./P00004.jpg")
# img2 = cv2.imread("./P00083.jpg")
# # xxx.get_objects(img)
# # xxx.get_objects(img1)
# yyy = xxx.get_objects(img2)
# dictionary = {'yay': [yyy, 1]}
# with open('saved_dictionary.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)


# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)
# for gg in loaded_dict['yay'][0]:
#     plt.imshow(gg[..., ::-1])
#     plt.show()
