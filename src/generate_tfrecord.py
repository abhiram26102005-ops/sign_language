import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from collections import namedtuple


LABEL_MAP = {
    "hello": 1,
    "thanks": 2,
    "yes": 3,
    "no": 4,
    "iloveyou": 5,
    "please": 6,
    "sorry": 7,
    "help": 8,
    "good": 9,
    "clockit": 10

}


def class_text_to_int(row_label):
    return LABEL_MAP[row_label]


def split(df, group):
    data = namedtuple("data", ["filename", "object"])
    grouped = df.groupby(group)

    return [
        data(filename, grouped_df)
        for filename, grouped_df in grouped
    ]


def create_tf_example(group, image_dir):
    image_path = os.path.join(image_dir, group.filename)

    with tf.io.gfile.GFile(image_path, "rb") as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    width, height = image.size

    filename = group.filename.encode("utf8")
    image_format = b"jpg"

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)

        classes_text.append(row["class"].encode("utf8"))
        classes.append(class_text_to_int(row["class"]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        "image/width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        "image/filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        "image/source_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        "image/encoded": tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        "image/format": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        "image/object/bbox/xmin": tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        "image/object/bbox/xmax": tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        "image/object/bbox/ymin": tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        "image/object/bbox/ymax": tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        "image/object/class/text": tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        "image/object/class/label": tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

    return tf_example


def generate_tfrecord(csv_input, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_input)

    grouped = split(examples, "filename")

    for group in grouped:
        tf_example = create_tf_example(group, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()

    print(f"Successfully created TFRecord: {output_path}")


if __name__ == "__main__":
    generate_tfrecord(
        "workspace/annotations/train_labels.csv",
        "workspace/images/train",
        "workspace/annotations/train.record"
    )

    generate_tfrecord(
        "workspace/annotations/test_labels.csv",
        "workspace/images/test",
        "workspace/annotations/test.record"
    )

    print("\nTFRecord generation completed successfully.")