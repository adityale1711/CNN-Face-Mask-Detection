import os
import argparse
import pandas as pd

from PIL import Image
from glob import glob
from generateDataFrame import DataFrameGenerator
from sklearn.model_selection import train_test_split

def crop_img(image_path, x_min, y_min, x_max, y_max):
  x_shift = (x_max - x_min) * 0.1
  y_shift = (y_max - y_min) * 0.1

  img = Image.open(image_path)
  cropped = img.crop((x_min - x_shift, y_min - y_shift, x_max + x_shift, y_max + y_shift))

  return cropped

def extract_faces(image_name, image_info):
  faces = []
  df_one_img = image_info[image_info['file'] == image_name[:-4]][['xmin', 'ymin', 'xmax', 'ymax', 'name']]

  for row_num in range(len(df_one_img)):
    x_min, y_min, x_max, y_max, label = df_one_img.iloc[row_num]
    image_path = os.path.join(df_gen.input_data_path, image_name)

    faces.append((crop_img(image_path, x_min, y_min, x_max, y_max), label, f'{image_name[:-4]}_{(x_min, y_min)}'))

  return faces

def save_image(image, image_name, output_data_path, dataset_type, label):
  output_path = os.path.join(output_data_path, dataset_type, label, f'{image_name}.png')

  image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data_path', type=str, required=True, help='The directory path for images data')
    parser.add_argument('--annotation_path', type=str, required=True, help='The directory path for annotations data')

    args, _ = parser.parse_known_args()
    df_gen = DataFrameGenerator(args)

    dataset = [df_gen.parse_annotation(anno) for anno in glob(df_gen.annotations_path + '/*.xml')]
    full_dataset = sum(dataset, [])
    df = pd.DataFrame(full_dataset)

    final_test_image = 'maksssksksss0'
    df_final_test = df.loc[df['file'] == final_test_image]

    df_gen.images.remove(f'{final_test_image}.png')
    df = df.loc[df['file'] != final_test_image]
    df = df.drop(df[df['name'] == 'mask_weared_incorrect'].index)

    labels = df['name'].unique()
    directory = ['train', 'test', 'val']
    output_data_path = '../../splitted_datasets/'

    for label in labels:
        for d in directory:
            path = os.path.join(output_data_path, d, label)
            if not os.path.exists(path):
                os.makedirs(path)

    cropped_images = [extract_faces(img, df) for img in df_gen.images]
    flat_cropped_faces = sum(cropped_images, [])

    with_mask = [(img, image_name) for img, label, image_name in flat_cropped_faces if label == 'with_mask']
    without_mask = [(img, image_name) for img, label, image_name in flat_cropped_faces if label == 'without_mask']

    with_mask_train, with_mask_test = train_test_split(with_mask, test_size=0.2, random_state=42)
    with_mask_test, with_mask_val = train_test_split(with_mask_test, test_size=0.7, random_state=42)

    without_mask_train, without_mask_test = train_test_split(without_mask, test_size=0.2, random_state=42)
    without_mask_test, without_mask_val = train_test_split(without_mask_test, test_size=0.7, random_state=42)

    datasets = [
        (with_mask_train, 'train', 'with_mask'),
        (without_mask_train, 'train', 'without_mask'),
        (with_mask_test, 'test', 'with_mask'),
        (without_mask_test, 'test', 'without_mask'),
        (with_mask_val, 'val', 'with_mask'),
        (without_mask_val, 'val', 'without_mask')
    ]

    for dataset, data_type, mask_status in datasets:
        for image, image_name in dataset:
            save_image(image, image_name, output_data_path, data_type, mask_status)