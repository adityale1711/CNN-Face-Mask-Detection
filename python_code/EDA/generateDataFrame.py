import os
import xml.etree.ElementTree as ET

class DataFrameGenerator:
    def __init__(self, args):
        self.input_data_path = args.input_data_path
        self.annotations_path = args.annotation_path
        self.images = [*os.listdir(self.input_data_path)]
        self.output_data_path = '/'

    def parse_annotation_object(self, annotation_object):
      params = {}
      for param in list(annotation_object):
        if param.tag == 'name':
          params['name'] = param.text
        if param.tag == 'bndbox':
          for coord in list(param):
            if coord.tag == 'xmin':
              params['xmin'] = int(coord.text)
            if coord.tag == 'ymin':
              params['ymin'] = int(coord.text)
            if coord.tag == 'xmax':
              params['xmax'] = int(coord.text)
            if coord.tag == 'ymax':
              params['ymax'] = int(coord.text)

      return params

    def merge(self, dict1, dict2):
      res = {**dict1, **dict2}

      return res

    def parse_annotation(self, path):
      tree = ET.parse(path)
      root = tree.getroot()
      constants = {}
      objects = [child for child in root if child.tag == 'object']
      for element in tree.iter():
        if element.tag == 'filename':
          constants['file'] = element.text[:-4]
        if element.tag == 'size':
          for dim in list(element):
            if dim.tag == 'width':
              constants['width'] = int(dim.text)
            if dim.tag == 'height':
              constants['height'] = int(dim.text)
            if dim.tag == 'depth':
              constants['depth'] = int(dim.text)

      object_params = [self.parse_annotation_object(obj) for obj in objects]
      full_result = [self.merge(constants, ob) for ob in object_params]

      return full_result