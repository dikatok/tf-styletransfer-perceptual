import argparse
import os
from glob import glob

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph


def parse_args():
    parser = argparse.ArgumentParser(description='Perceptual Losses for Real-Time Style Transfer and Super-Resolution')
    parser.add_argument('--output_path', default='model.pb', help='Content loss weight.', type=str)
    parser.add_argument('--saved_model_dir', default='saved_model', help='Content loss weight.', type=str)
    return parser.parse_args()


def main(_):
    args = parse_args()

    saved_models = sorted(glob(os.path.join(args.saved_model_dir, '*')))

    assert len(saved_models) > 0

    with tf.Session() as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_models[-1])

        graph = tf.get_default_graph()

        convert_variables_to_constants = tf.graph_util.convert_variables_to_constants

        output_graph_def = convert_variables_to_constants(
            sess,
            graph.as_graph_def(),
            ["transferred"])

        input_names = ["images"]
        output_names = ["transferred"]

        transforms = ["strip_unused_nodes", "fold_batch_norms", "fold_constants", "quantize_weights"]

        transformed_graph_def = TransformGraph(
            output_graph_def,
            input_names,
            output_names,
            transforms)

        with tf.gfile.GFile(args.output_path, "wb") as f:
            f.write(transformed_graph_def.SerializeToString())


if __name__ == "__main__":
    tf.app.run(main=main)
