import numpy as np
import cv2
import os
from typing import Tuple

# Paths to load the model
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")

# Initialize model globally
net = None
pts = None

def initialize_model() -> None:
    """Initialize the colorization model and load necessary files."""
    global net, pts
    
    if net is None:
        print("Loading model...")
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
        pts = np.load(POINTS)
        
        # Load centers for ab channel quantization used for rebalancing
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Colorize a black and white image.
    
    Args:
        image: Input image as numpy array in BGR format
        
    Returns:
        Tuple containing (original image, colorized image) as numpy arrays
    """
    if net is None:
        initialize_model()
        
    # Convert image to float32 and scale to [0, 1]
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    # Resize image for model input
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    # Colorize the image
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    
    # Resize ab channels to match original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    
    # Combine L and ab channels
    lab = cv2.split(lab)[0]
    colorized = np.concatenate((lab[:, :, np.newaxis], ab), axis=2)
    
    # Convert back to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return image, colorized 