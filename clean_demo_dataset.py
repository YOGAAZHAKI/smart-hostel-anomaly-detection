import os
import shutil
import sys
import torch
from PIL import Image
from tqdm import tqdm

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("Error: transformers library is not installed. Please install it first.")
    sys.exit(1)

def main():
    base_dir = r"E:\smart-hostel\smart-hostel-anomaly-detection"
    demo_dataset = os.path.join(base_dir, "output_demo")
    
    if not os.path.exists(demo_dataset):
        print(f"Error: Demo dataset {demo_dataset} not found. Please run run_demo_prep.py first.")
        return
        
    print("\n==================================================")
    print("🧹 CLEANING DEMO DATASET USING CLIP FOUNDATION MODEL")
    print("==================================================")
    
    print("Loading CLIP model (openai/clip-vit-base-patch32)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Define labels and mapping
    labels = [
        "an empty room, normal hostel corridor, lobby, or quiet hallway", 
        "a burglary break-in, thief stealing, forced entry, or picking a lock", 
        "people fighting, physical violence, hitting, punching, or combat"
    ]
    class_folders = {
        0: "normal",        # Class 0: Normal
        1: "burglary",      # Class 1: Burglary (anomaly)
        2: "fighting"       # Class 2: Fighting (anomaly)
    }
    
    # Gather all images in the output_demo directory
    all_images = []
    for root, dirs, files in os.walk(demo_dataset):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, f))
                
    print(f"Found {len(all_images)} images to check.")
    
    # Temporary directory to hold cleaned files
    temp_clean_dir = os.path.join(base_dir, "output_demo_clean")
    if os.path.exists(temp_clean_dir):
        shutil.rmtree(temp_clean_dir)
        
    # Recreate the correct folder structure
    # Note: the trainer expects output_demo/{crime_category}/normal and output_demo/{crime_category}/anomaly
    # But since we are cleaning, we can keep the trainer happy by organizing them as:
    # - burglary/normal (Normal frames from burglary videos)
    # - burglary/anomaly (Actual burglary frames)
    # - fighting/normal (Normal frames from fighting videos)
    # - fighting/anomaly (Actual fighting frames)
    
    os.makedirs(os.path.join(temp_clean_dir, "burglary", "normal"), exist_ok=True)
    os.makedirs(os.path.join(temp_clean_dir, "burglary", "anomaly"), exist_ok=True)
    os.makedirs(os.path.join(temp_clean_dir, "fighting", "normal"), exist_ok=True)
    os.makedirs(os.path.join(temp_clean_dir, "fighting", "anomaly"), exist_ok=True)
    
    print("\nRunning CLIP zero-shot classification to clean labels...")
    corrected_count = 0
    
    for img_path in tqdm(all_images, desc="Cleaning images"):
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Predict
            inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
                pred_class_id = torch.argmax(probs).item()
                
            filename = os.path.basename(img_path)
            
            # Decide where to save based on prediction
            # We determine original category (burglary or fighting) from the file name
            is_burglary_vid = "burglary" in filename.lower()
            
            if pred_class_id == 0:  # Normal
                dest_subfolder = "burglary/normal" if is_burglary_vid else "fighting/normal"
            elif pred_class_id == 1:  # Burglary
                dest_subfolder = "burglary/anomaly"
            else:  # Fighting
                dest_subfolder = "fighting/anomaly"
                
            # Copy to cleaned dataset directory
            dest_path = os.path.join(temp_clean_dir, dest_subfolder, filename)
            shutil.copy2(img_path, dest_path)
            
            # Track if we corrected the class
            orig_subfolder = "normal" if "normal" in img_path.lower() else "anomaly"
            orig_crime = "burglary" if "burglary" in img_path.lower() else "fighting"
            
            predicted_crime = "burglary" if pred_class_id == 1 else ("fighting" if pred_class_id == 2 else "normal")
            
            if orig_subfolder == "anomaly" and predicted_crime == "normal":
                corrected_count += 1
                
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            
    print(f"\nDataset cleaning complete!")
    print(f"Corrected {corrected_count} false-anomaly labels (reclassified as Normal).")
    
    # Swap output_demo with the cleaned version
    print("\nSwapping dataset folders...")
    shutil.rmtree(demo_dataset)
    shutil.move(temp_clean_dir, demo_dataset)
    print("Done! output_demo now contains clean, CLIP-verified images.")

if __name__ == '__main__':
    main()
