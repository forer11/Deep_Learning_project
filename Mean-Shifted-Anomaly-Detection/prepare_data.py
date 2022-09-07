from __future__ import print_function, division
import os
import pickle
from struct import Struct

import tensorflow as tf
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image
import math

# Ignore warnings
import warnings

from object_extractor import ObjectsExtractor

warnings.filterwarnings("ignore")

IMG_PATH = 'C:/Users/Charlool/Desktop/cs_studies_carmeliol/deep_learning_proj/Data/SIXray/dataset/JPEGImage/'
yayness = 'C:/Users/Charlool/Desktop/cs_studies_carmeliol/deep_learning_proj/Data/SIXray/dataset/ImageSet' \
          '/New_folder'

CSV_BASE_PATH = 'C:/Users/Charlool/Desktop/cs_studies_carmeliol/deep_learning_proj/Data/SIXray/dataset/ImageSet' \
                '/train_test_easy/'
CLASSES_PATH = 'C:/Users/Charlool/Desktop/cs_studies_carmeliol/deep_learning_proj/Data/SIXray/dataset/ImageSet/class_txt/'

CLASSES = {'G': 0, 'K': 1, 'W': 2, 'P': 3, 'S': 4, 'N': 5, 'C': 6}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h, (1,))
        left = torch.randint(0, w - new_w, (1,))

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}


class XrayDataset(Dataset):
    """Face Landmarks dataset."""

    @staticmethod
    def get_target(row):
        img_name = row['name']

        gun = row['Gun'] if 'Gun' in row else -1
        knife = row['Knife'] if 'Knife' in row else -1
        pliers = row['Pliers'] if 'Pliers' in row else -1
        scissors = row['Scissors'] if 'Scissors' in row else -1
        wrench = row['Wrench'] if 'Wrench' in row else -1

        if gun + knife + pliers + scissors + wrench > -3:
            return CLASSES['C']

        elif not math.isnan(gun) and gun == 1:
            return CLASSES['G']
        elif not math.isnan(knife) and knife == 1:
            return CLASSES['K']
        elif not math.isnan(pliers) and pliers == 1:
            return CLASSES['P']
        elif not math.isnan(scissors) and scissors == 1:
            return CLASSES['S']
        elif not math.isnan(wrench) and wrench == 1:
            return CLASSES['W']
        else:
            return CLASSES['N']

    def __init__(self, csv_file, root_dir, transform=None, all_objects=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # gun_set, knife_set, pliers_set, scissors_set, wrench_set = set(), set(), set(), set(), set()
        # get_images_set(gun_set, CLASSES_PATH + 'Gun.txt')
        # get_images_set(knife_set, CLASSES_PATH + 'Knife.txt')
        # get_images_set(pliers_set, CLASSES_PATH + 'Pliers.txt')
        # get_images_set(scissors_set, CLASSES_PATH + 'Scissors.txt')
        # get_images_set(wrench_set, CLASSES_PATH + 'Wrench.txt')
        #
        # self.only_gun_set = filter_other_classes(gun_set, [knife_set, pliers_set, scissors_set, wrench_set])
        # self.only_knife_set = filter_other_classes(knife_set, [gun_set, pliers_set, scissors_set, wrench_set])
        # self.only_pliers_set = filter_other_classes(pliers_set, [knife_set, gun_set, scissors_set, wrench_set])
        # self.only_scissors_set = filter_other_classes(scissors_set, [knife_set, pliers_set, gun_set, wrench_set])
        # self.only_wrench_set = filter_other_classes(wrench_set, [knife_set, pliers_set, scissors_set, gun_set])

        if all_objects is None:
            all_objects = []
        self.all_objects = all_objects

        self.xray_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.xray_frame['name'] = self.xray_frame['name'] + '.jpg'
        self.xray_frame['targets'] = self.xray_frame.apply(lambda row: self.get_target(row), axis=1)
        self.data = self.xray_frame['name']
        self.targets = self.xray_frame['targets']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.data[idx]
        img_full_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_full_path)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(image)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample, self.targets[idx], img_name


def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)


def test_transform(xray_dataset):
    scale = Rescale(256)
    crop = RandomCrop(150)
    composed = transforms.Compose([Rescale(256),
                                   RandomCrop(224)])

    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = xray_dataset[500]
    ax = plt.subplot(1, 4, 1)
    plt.tight_layout()
    ax.set_title('normal')
    show_image(sample)

    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)

        ax = plt.subplot(1, 4, i + 2)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_image(**transformed_sample)
    plt.show()


def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch = sample_batched
    batch_size = len(images_batch)
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')


def get_images_set(images_set, path):
    f = open(path, "r")
    for x in f:
        images_set.add(x.rstrip())
    f.close()


def filter_other_classes(image_set, other_sets):
    new_set = set()
    for img in image_set:
        image_not_in_other_sets = True
        for other_set in other_sets:
            if img in other_set:
                image_not_in_other_sets = False
                break
        if image_not_in_other_sets:
            new_set.add(img)
    return new_set


def get_extracted_objects_dict(files):
    all_objects_dict = {}
    for file in files:
        temp_objects_dict = {}
        with open(file + '.pkl', 'rb') as f:
            temp_objects_dict = pickle.load(f)
            all_objects_dict.update(temp_objects_dict)
    return all_objects_dict


def get_files_list(file_name, num_of_files):
    files = []
    for i in range(num_of_files):
        files.append(file_name + str(i))
    return files

# plt.ion()  # interactive mode
# transform_color = transforms.Compose([transforms.Resize(256),
#                                       transforms.CenterCrop(224),
#                                       transforms.ToTensor()])
# xray_dataset = XrayDataset(csv_file=CSV_BASE_PATH + 'test_easy_new.csv', root_dir=IMG_PATH, transform=transform_color)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# # with open('saved_test_data.pkl', 'rb') as f:
# #     all_objects_dict = pickle.load(f)
# test_files = get_files_list('saved_test_data', 17)
# all_objects_dict = get_extracted_objects_dict(test_files)
# objectExtractor = ObjectsExtractor()
# pkl_name = 'saved_test_data'
# num_of_pkl = 18
# objects_dict = {}
# for index, img_name in enumerate(xray_dataset.data):
#     if img_name not in all_objects_dict and img_name not in objects_dict:
#         img_full_path = os.path.join(IMG_PATH, img_name)
#         image = io.imread(img_full_path)
#         extracted_objects = objectExtractor.get_objects(image)
#         objects_dict[img_name] = extracted_objects
#         if index % 50 == 0:
#             with open(pkl_name + str(num_of_pkl) + '.pkl', 'wb') as f:
#                 print('saving.. ' + pkl_name + str(num_of_pkl))
#                 pickle.dump(objects_dict, f)
#             if index % 200 == 0:
#                 num_of_pkl += 1
#                 all_objects_dict.update(objects_dict)
#                 objects_dict = {}
#     print(img_name + '  ' + str(index) + '/' + str(len(xray_dataset.data)))

# dataloader = DataLoader(xray_dataset, batch_size=4,
#                         shuffle=True, num_workers=0)


# for i_batch, sample_batched in enumerate(dataloader):
#     # observe 4th batch and stop.
#     if i_batch == 3:
#         print(sample_batched[0].size())
#         plt.figure()
#         show_landmarks_batch(sample_batched[0])
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break
#
# plt.figure()
# test_transform(xray_dataset)
