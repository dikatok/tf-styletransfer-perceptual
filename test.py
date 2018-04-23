import argparse
import os

import tensorflow as tf


def _load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="graph")

    return graph


def parse_args():
    parser = argparse.ArgumentParser(description='Perceptual Losses for Real-Time Style Transfer and Super-Resolution')
    parser.add_argument('input_path', help='Path to input video.', type=str)
    parser.add_argument('output_path', help='Path to save output video.', type=str)
    parser.add_argument('--model_path', default='./model.pb', help='Path to frozen model.', type=str)
    return parser.parse_args()


def main(_):
    args = parse_args()

    assert os.path.exists(args.input_path), "Input video does not exist"
    assert os.path.exists(args.model_path), "Model does not exist"

    graph = _load_graph(args.model_path)

    content = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(args.input_path)
    )

    with tf.Session(graph=graph) as sess:
        inputs = graph.get_tensor_by_name('graph/images:0')
        outputs = graph.get_tensor_by_name('graph/transferred:0')

        result = sess.run(outputs, {inputs: [content]})

        tf.keras.preprocessing.image.array_to_img(result[0]).save(args.output_path)


if __name__ == "__main__":
    tf.app.run(main=main)
