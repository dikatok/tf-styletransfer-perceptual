import argparse

import tensorflow as tf

from core.model import create_model_fn
from utils.train_utils import create_estimator_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Perceptual Losses for Real-Time Style Transfer and Super-Resolution')
    parser.add_argument('--model_dir', default='./ckpts', help='Content loss weight.', type=str)
    parser.add_argument('--saved_model_dir', default='saved_model', help='Content loss weight.', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_fn = create_model_fn()

    estimator_fn = create_estimator_fn(
        model_fn=model_fn,
        loss_model_fn=None,
        loss_fn=None,
        data_format="channels_last")

    estimator = tf.estimator.Estimator(
        model_fn=estimator_fn,
        model_dir=args.model_dir)

    def serving_input_fn():
        images = tf.placeholder(
            shape=[None, None, None, 3],
            dtype=tf.float32,
            name="images")
        return tf.estimator.export.ServingInputReceiver({"contents": images}, {"contents": images})

    estimator.export_savedmodel(args.saved_model_dir, serving_input_fn)
