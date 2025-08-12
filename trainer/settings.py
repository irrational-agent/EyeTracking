
import os

MAX_OFFSET = 10  # Maximum pixel offset for augmentation.
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 256, 1)
LEARNING_RATE = 0.002
DATASET_DIR = os.path.abspath(r".\dataset")
OUTPUT_DIR = os.path.abspath(r".\res")
EPOCHS = 30
