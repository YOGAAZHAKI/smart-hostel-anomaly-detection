import os
import shutil
import glob

def prepare_dataset():
    base_dir = r"E:\smart-hostel\smart-hostel-anomaly-detection"
    frames_dir = os.path.join(base_dir, "FRAMES")
    output_dir = os.path.join(base_dir, "output")

    # Define our mapping from source directories to output structure
    categories = {
        "Burglary_frames": "burglary",
        "Fighting_frames": "fighting"
    }

    # Create base output directories
    for cat in categories.values():
        os.makedirs(os.path.join(output_dir, cat, "normal"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, cat, "anomaly"), exist_ok=True)

    print("Starting dataset preparation...")
    total_copied = 0

    for src_folder, dest_name in categories.items():
        src_path = os.path.join(frames_dir, src_folder)
        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found. Skipping.")
            continue
            
        print(f"\nProcessing {src_folder}...")
        
        # Iterate over each video clip folder (e.g., Burglary001_x264)
        clip_folders = [f.path for f in os.scandir(src_path) if f.is_dir()]
        
        for clip in clip_folders:
            clip_name = os.path.basename(clip)
            # Find all frames in this clip
            frames = glob.glob(os.path.join(clip, "*.jpg"))
            if not frames:
                continue
                
            # Sort frames numerically (frame_0.jpg, frame_1.jpg, etc.)
            frames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            # Split: First 30% normal, remaining 70% anomaly
            split_idx = int(len(frames) * 0.3)
            normal_frames = frames[:split_idx]
            anomaly_frames = frames[split_idx:]
            
            # Copy Normal frames
            for frame in normal_frames:
                dest = os.path.join(output_dir, dest_name, "normal", f"{clip_name}_{os.path.basename(frame)}")
                shutil.copy2(frame, dest)
                total_copied += 1
                
            # Copy Anomaly frames
            for frame in anomaly_frames:
                dest = os.path.join(output_dir, dest_name, "anomaly", f"{clip_name}_{os.path.basename(frame)}")
                shutil.copy2(frame, dest)
                total_copied += 1
                
            print(f"  - {clip_name}: {len(normal_frames)} normal, {len(anomaly_frames)} anomaly frames copied.")

    print(f"\nDone! Successfully organized {total_copied} frames into the output/ directory.")

if __name__ == "__main__":
    prepare_dataset()
