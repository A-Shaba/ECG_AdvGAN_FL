"""
Preprocessing script for ECG image dataset.
Loads raw ECG images, applies preprocessing (resize, normalization), and saves processed data for model training/testing.
"""

import os
import numpy as np
from PIL import Image
import argparse

RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'processed')
IMG_SIZE = (224, 224)  # Standard size for CNNs

def preprocess_image(img_path, img_size=IMG_SIZE):
	"""Load an image, resize, and normalize to [0, 1]."""
	img = Image.open(img_path).convert('RGB')
	img = img.resize(img_size)
	img_array = np.asarray(img, dtype=np.float32) / 255.0
	return img_array

def process_dataset(raw_dir=RAW_DIR, processed_dir=PROCESSED_DIR, img_size=IMG_SIZE):
	"""Process all images in raw_dir and save as .npy arrays in processed_dir."""
	if not os.path.exists(processed_dir):
		os.makedirs(processed_dir)

	labels = []
	images = []

	# Assume subfolders in raw_dir are class labels
	for label in os.listdir(raw_dir):
		label_path = os.path.join(raw_dir, label)
		if not os.path.isdir(label_path):
			continue
		for fname in os.listdir(label_path):
			if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
				img_path = os.path.join(label_path, fname)
				img_array = preprocess_image(img_path, img_size)
				images.append(img_array)
				labels.append(label)

	images = np.stack(images)
	labels = np.array(labels)

	np.save(os.path.join(processed_dir, 'images.npy'), images)
	np.save(os.path.join(processed_dir, 'labels.npy'), labels)
	print(f"Processed {len(images)} images. Saved to {processed_dir}.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Preprocess ECG image dataset.")
	parser.add_argument('--raw_dir', type=str, default=RAW_DIR, help='Directory with raw ECG images (by class)')
	parser.add_argument('--processed_dir', type=str, default=PROCESSED_DIR, help='Directory to save processed data')
	parser.add_argument('--img_size', type=int, nargs=2, default=IMG_SIZE, help='Image size (H W)')
	args = parser.parse_args()
	process_dataset(args.raw_dir, args.processed_dir, tuple(args.img_size))
