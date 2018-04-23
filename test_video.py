import argparse
import os

import cv2
import numpy as np
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

    cap = cv2.VideoCapture(args.input_path)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    save = cv2.VideoWriter(
        args.output_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (int(width), int(height)),
        isColor=True)

    try:
        with tf.Session(graph=graph) as sess:
            inputs = graph.get_tensor_by_name('graph/images:0')
            outputs = tf.image.resize_bilinear(
                tf.cast(graph.get_tensor_by_name('graph/transferred:0'), dtype=tf.float32),
                size=tf.shape(inputs)[1:3])

            while cap.isOpened():
                ret, frame = cap.read()

                if frame is None:
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = sess.run(outputs, {inputs: [frame]})

                save.write(np.uint8(cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR)))
    finally:
        cap.release()
        save.release()


if __name__ == "__main__":
    tf.app.run(main=main)