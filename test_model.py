#!/usr/bin/env python3
"""
Script to test the trained model on sample images.
"""

import os
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def test_model_on_images():
    """Test the trained model on sample images from each category."""
    
    # Import the inference class from train_classifier
    sys.path.insert(0, BASE_DIR)
    from train_classifier import CrimeDetectorInference
    
    model_path = os.path.join(BASE_DIR, 'crime_detector_refined.pth')
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run training first: python train_classifier.py --mode train --model vit --epochs 20")
        return
    
    print("\n" + "="*70)
    print("🎯 TESTING TRAINED MODEL")
    print("="*70)
    
    # Load detector
    detector = CrimeDetectorInference(model_path)
    
    # Test on sample images from each category
    test_cases = [
        ('Normal (Burglary)', f'{OUTPUT_DIR}/burglary/normal'),
        ('Burglary (Anomaly)', f'{OUTPUT_DIR}/burglary/anomaly'),
        ('Normal (Fighting)', f'{OUTPUT_DIR}/fighting/normal'),
        ('Fighting (Anomaly)', f'{OUTPUT_DIR}/fighting/anomaly'),
    ]
    
    for category, folder in test_cases:
        print(f"\n📁 Testing: {category}")
        print("-" * 70)
        
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"⚠️  Folder not found: {folder}")
            continue
        
        # Get first 3 images from folder
        images = (list(folder_path.glob('*.jpg')) + 
                 list(folder_path.glob('*.png')) + 
                 list(folder_path.glob('*.jpeg')))[:3]
        
        if not images:
            print(f"⚠️  No images found in {folder}")
            continue
        
        for img_path in images:
            result = detector.predict_image(str(img_path))
            
            print(f"\n  📸 {img_path.name}")
            print(f"  🎯 Prediction: {result['class']}")
            print(f"  📊 Confidence: {result['confidence']:.1%}")
            
            # Show top predictions
            sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
            for cls, prob in sorted_probs:
                bar = "█" * int(prob * 40)
                print(f"     {cls:12} {prob:6.1%} {bar}")

if __name__ == "__main__":
    test_model_on_images()
