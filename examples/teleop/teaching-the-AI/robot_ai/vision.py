"""The eyes.

Everything that turns a picture into numbers the network can eat lives here, and
NOTHING ELSE in this project is allowed to resize or rescale an image. That rule
matters more than it sounds: a policy only ever sees pictures through this file,
so if training and driving disagree about the size, the colours, or the stretch,
the robot is looking at a world it has never seen before and will behave like it.

The pixels take a slightly odd road to get here, and it is worth knowing why:

    the camera gives us            640 x 480   BGR   (that is the real sensor)
    the teleop script stretched it 1280 x 720  RGB   (and that is what got recorded)
    we shrink that down to          320 x 180  RGB   (small enough to train on)

The stretch in the middle is not useful - it was a mistake nobody noticed while
recording - but the recordings are full of it, so we have to repeat it forever.
That is a real lesson about datasets: whatever your data does, warts and all,
your robot has to do too.
"""

import cv2
import numpy as np
import torch

# What the network sees. 320x180 keeps the same shape as the recording (16:9)
# and is small enough that training takes minutes instead of hours.
IMAGE_W, IMAGE_H = 320, 180

# What the recorded videos contain.
RECORDED_W, RECORDED_H = 1280, 720

# What the RealSense camera actually produces.
CAMERA_W, CAMERA_H = 640, 480

# ResNet was trained on millions of photos that were rescaled with these numbers.
# We reuse them so our pictures look "normal" to it.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def from_recording(rgb_image):
    """A frame out of a recorded video (720x1280 RGB) -> what the network sees."""
    if rgb_image.shape[0] != RECORDED_H or rgb_image.shape[1] != RECORDED_W:
        raise ValueError(
            f"expected a {RECORDED_H}x{RECORDED_W} recorded frame, got {rgb_image.shape}"
        )
    return cv2.resize(rgb_image, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)


def from_camera(bgr_image):
    """A live frame from the RealSense (480x640 BGR) -> what the network sees.

    This is the path used when the robot is actually running. It deliberately
    repeats the stretch that the teleop script applied while recording, so the
    live picture and the training picture are the same picture.
    """
    if bgr_image.shape[0] != CAMERA_H or bgr_image.shape[1] != CAMERA_W:
        raise ValueError(
            f"expected a {CAMERA_H}x{CAMERA_W} camera frame, got {bgr_image.shape}"
        )
    stretched = cv2.resize(bgr_image, (RECORDED_W, RECORDED_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(stretched, cv2.COLOR_BGR2RGB)
    return from_recording(rgb)


def to_tensor(image):
    """Picture (H, W, 3) of 0-255 whole numbers -> tensor (3, H, W) of small decimals.

    Three things happen, and every one of them is required:
      1. 0..255  ->  0.0..1.0          (networks hate big numbers)
      2. subtract the mean, divide by the standard deviation  (ResNet expects this)
      3. move the colour channel to the front  (PyTorch's convention: C, H, W)
    """
    if image.dtype != np.uint8:
        raise TypeError(f"expected a uint8 picture, got {image.dtype}")
    x = image.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def to_batch(image, device="cpu"):
    """Same as to_tensor, plus the extra 'batch of 1' dimension the network wants."""
    return to_tensor(image).unsqueeze(0).to(device)
