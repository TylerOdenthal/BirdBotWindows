"""
# Create train data:

python generate_tfrecord.py --csv_input=B:/BirdBot/BirdModelTraining/data/train_labels.csv --image_dir=B:\BirdBot\BirdModelTraining\images  --output_path=B:/BirdBot/BirdModelTraining/data/train.record

# Create test data:

python generate_tfrecord.py --csv_input=B:/BirdBot/BirdModelTraining/data/test_labels.csv --image_dir=B:\BirdBot\BirdModelTraining\images --output_path=B:/BirdBot/BirdModelTraining/data/test.record

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS

print(FLAGS.image_dir)


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == "American Robin":
        return 1
    elif row_label == "Black-capped Chickadee":
        return 2
    elif row_label == "Steller's Jay":
        return 3
    elif row_label == "House Finch":
        return 4   
    elif row_label == "Red-breasted Nuthatch":
        return 5
    elif row_label == "Black-headed Grosbeak":
        return 6
    elif row_label == "House Sparrow":
        return 7
    elif row_label == "Dark-eyed Junco":
        return 8
    elif row_label == "Pine Siskin":
        return 9
    elif row_label == "Golden-crowned Sparrow":
        return 10
    elif row_label == "Chestnut-backed Chickadee":
        return 11
    elif row_label == "European Starling":
        return 12
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()