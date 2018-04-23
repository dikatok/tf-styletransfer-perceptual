import argparse
import os

import cv2
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
    parser.add_argument('--model_path', default='./model.pb', help='Path to frozen model.', type=str)
    parser.add_argument('--capture_height', default=None, help='Capture height.', type=int)
    parser.add_argument('--capture_width', default=None, help='Capture width.', type=int)
    return parser.parse_args()


def main(_):
    args = parse_args()

    assert os.path.exists(args.model_path), "Model does not exist"

    cap = cv2.VideoCapture(0)
    if args.capture_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    if args.capture_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)

    try:
        graph = _load_graph(args.model_path)

        with tf.Session(graph=graph) as sess:
            inputs = graph.get_tensor_by_name('graph/images:0')
            outputs = graph.get_tensor_by_name('graph/transferred:0')

            while cap.isOpened():
                ret, frame = cap.read()

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = cv2.flip(frame, 1)

                if len(frame.shape) < 3:
                    break

                result = sess.run(outputs, {inputs: [frame]})

                cv2.imshow('result', cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR) / 255.)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()


if __name__ == "__main__":
    tf.app.run(main=main)
