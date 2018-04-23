# Perceptual Losses for Real-Time Style Transfer and Super-Resolution

This is Tensorflow implementation of [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1508.06576).

## Directory Structure
```
.
├── convert_to_saved_model.py
├── core
│   ├── __init__.py
│   ├── layers.py
│   ├── loss.py
│   ├── model.py
├── create_tfrecords.py
├── freeze_model.py
├── README.md
├── requirements.txt
├── test.py
├── test_video.py
├── test_webcam.py
├── train.py
├── utils
│   ├── __init__.py
│   ├── data_utils.py
│   └── train_utils.py
└── vgg16_weights.npz

```

## Installation
- `pip install -r requirements.txt`
#### 1. Train
- Download vgg16 weights file from either [here](https://drive.google.com/open?id=1vpyQ855RCRHkO-9oOlo4JLaccS8oguW0) which contains only the required kernels (~55mb) or the full one from [here](http://www.cs.toronto.edu/~frossard/post/vgg16/) which the previous link is based on (~554mb)
- Download MS COCO 2014 training set from [here](http://cocodataset.org/#download)
- Create training set tfrecord using `create_tfrecords.py` script
#### 2. Test
- Download pre-trained frozen model from [here](https://drive.google.com/open?id=1jNINZkrXwat4O-9V3fGhn5inO6Nh9Mno)

## Usage
#### 1. Train
    Usage:
        python train.py [options] <style_image>
        
    Options:
        --restart_training      Whether to restart the training and remove all checkpoints and summaries
        --train_files           List of tfrecord filenames
        --model_dir             Directory to save checkpoints and summaries
        --vgg_path              Path to vgg numpy weights
        --data_format           Either channels_last or channels_last
        --num_epochs            Number of training epochs
        --batch_size            Batch size
        --shuffle_buffer_size   Buffer size to shuffle data
        --learning_rate         Initial learning rate
        --image_size            Training image size (content and style)
        --content_features      List of vgg features map to use as content features
        --style_features        List of vgg features map to use as style features
        --content_weight        Content loss weight
        --style_weight          Style loss weight
        --log_iter              Iteration interval to save summaries
        --checkpoint_iter       Iteration interval to save checkpoints
#### 2. Convert checkpoint to saved model
    Usage:
        python convert_to_saved_model.py [options]
        
    Options:
        --model_dir         Directory to find checkpoints
        --saved_model_dir   Directory to save saved models
#### 3. Freeze model
    Usage:
        python freeze_model.py [options]
        
    Options:
        --output_path       Path to save frozen model
        --saved_model_dir   Directory to find saved models
#### 4. Test
    Usage:
        python test.py [options] input_path output_path
        
        python test_video.py [options] input_path output_path
        
        python test_webcam.py [options]
    Options:
        --model_path        Path to frozen model
        
        --model_path        Path to frozen model
        
        --model_path        Path to frozen model
        --capture_height    Height of captured webcam frame
        --capture_width     Width of captured webcam frame

## Example
#### 1. Train
    python train.py --vgg_path vgg_small.npz --restart_training style.jpg
#### 2. Test
    python test.py content.jpg output.jpg
    
    python test_video.py content.mp4 output.mp4
    
    python test_webcam.py