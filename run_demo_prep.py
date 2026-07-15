import os
import shutil
import random
import subprocess
import sys

# Force UTF-8 encoding for standard output
sys.stdout.reconfigure(encoding='utf-8')

def prep_demo():
    base_dir = r"E:\smart-hostel\smart-hostel-anomaly-detection"
    src_dataset = os.path.join(base_dir, "output")
    demo_dataset = os.path.join(base_dir, "output_demo")
    python_exe = r"C:\Users\yogaa\AppData\Local\Programs\Python\Python313\python.exe"
    
    print("--- DEMO PREPARATION SCRIPT ---")
    
    # 1. Create demo dataset (subset of 20 images per category)
    categories = [
        "burglary/normal",
        "burglary/anomaly",
        "fighting/normal",
        "fighting/anomaly"
    ]
    
    if os.path.exists(demo_dataset):
        print("Cleaning up old demo dataset...")
        shutil.rmtree(demo_dataset)
        
    print("Creating a small subset dataset (150 images per folder) for quick training...")
    for cat in categories:
        src_path = os.path.join(src_dataset, cat)
        dest_path = os.path.join(demo_dataset, cat)
        os.makedirs(dest_path, exist_ok=True)
        
        if not os.path.exists(src_path):
            print(f"Error: Source directory {src_path} does not exist. Run prepare_dataset.py first!")
            return
            
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not images:
            print(f"Warning: No images in {src_path}")
            continue
            
        # Select 150 random images
        selected = random.sample(images, min(150, len(images)))
        for img in selected:
            shutil.copy2(os.path.join(src_path, img), os.path.join(dest_path, img))
            
    print("Demo dataset created successfully.")
    
    # 2. Train lightweight ResNet model for 3 epochs
    print("\nTraining a lightweight ResNet model for 3 epochs on the demo subset...")
    print("This should take about 3-5 minutes on CPU...")
    
    cmd = [
        python_exe,
        "train_classifier.py",
        "--mode", "train",
        "--model", "resnet",
        "--dataset", "output_demo",
        "--epochs", "5",
        "--batch-size", "16"
    ]
    
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    # Run the training command
    process = subprocess.Popen(cmd, cwd=base_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    
    # Print training progress in real-time
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode == 0:
        print("\n🎉 Demo Model trained and saved successfully as 'crime_detector_refined.pth'!")
        print("\nHow to run the Demo:")
        print("1. To run live Webcam anomaly detection:")
        print(f"   {python_exe} evaluate_custom.py --option webcam")
        print("2. To test it on sample images:")
        print(f"   {python_exe} test_model.py")
    else:
        print(f"\n❌ Training failed with exit code {process.returncode}")

if __name__ == "__main__":
    prep_demo()
