import torch
import os
import sys

model_path = 'dataset/crime_detector_refined.pth'

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}")
    sys.exit(1)

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print("KEYS:", checkpoint.keys())
    if 'class_names' in checkpoint:
        print("Class Names:", checkpoint['class_names'])
    if 'num_classes' in checkpoint:
        print("Num Classes:", checkpoint['num_classes'])
    if 'accuracy' in checkpoint:
        print("Training Accuracy:", checkpoint['accuracy'])
        
except Exception as e:
    print(f"Error loading model: {e}")
