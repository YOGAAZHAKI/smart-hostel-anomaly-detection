import os
import sys
import cv2
import argparse
import time
import numpy as np
import smtplib
import threading
from email.message import EmailMessage
from pathlib import Path

# Add dataset directory to path so we can import train_classifier
sys.path.append(os.path.join(os.path.dirname(__file__), 'dataset'))

try:
    from train_classifier import CrimeDetectorInference
except ImportError:
    print("Error: Could not import CrimeDetectorInference from dataset.train_classifier")
    sys.exit(1)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class AnomalyEvaluator:
    def __init__(self, model_path, output_dir='detected_anomalies', threshold=0.7):
        self.detector = CrimeDetectorInference(model_path)
        self.output_dir = output_dir
        self.threshold = threshold
        ensure_dir(output_dir)
        
        # State for smart saving
        self.last_save_time = 0
        self.last_saved_frame = None
        self.min_save_interval = 1.0 # Minimum seconds between saves (disk)
        self.diff_threshold = 15.0   # Minimum pixel difference (0-255)
        
        # Email Alert Config
        self.email_user = 'vigneshkofficial06@gmail.com'
        self.email_pass = 'cxaf wvus dtxa dsqq' # App Password
        self.last_email_time = 0
        self.email_cooldown = 30.0 # Minimum seconds between emails
        
        print(f"✅ Anomaly storage directory: {output_dir}")
        print(f"⚙️  Smart Save: Enabled (Interval={self.min_save_interval}s)")
        print(f"📧 Email Alerts: Enabled (Cooldown={self.email_cooldown}s) -> {self.email_user}")

    def send_email_async(self, image_path, label):
        def _send():
            try:
                msg = EmailMessage()
                msg['Subject'] = f'⚠️ Anomaly Detected: {label}'
                msg['From'] = self.email_user
                msg['To'] = self.email_user # Sending to self
                msg.set_content(f"Suspicious activity ({label}) detected at {time.strftime('%Y-%m-%d %H:%M:%S')}.\n\nSee attached image.")

                with open(image_path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(image_path)

                msg.add_attachment(file_data, maintype='image', subtype='jpeg', filename=file_name)

                print("⏳ Sending alert email...")
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login(self.email_user, self.email_pass)
                    smtp.send_message(msg)
                
                print(f"✅ Alert Mail Sent to {self.email_user}!")
            except Exception as e:
                print(f"❌ Failed to send email: {e}")

        # Start in separate thread to not block video
        threading.Thread(target=_send, daemon=True).start()

    def check_and_save_anomaly(self, frame, label, confidence, frame_id=None):
        current_time = time.time()
        
        # 1. Cooldown Check (Disk Save)
        if (current_time - self.last_save_time) < self.min_save_interval:
            return

        # 2. Visual Difference Check (Dedup)
        diff_score = 0.0
        if self.last_saved_frame is not None:
             try:
                 small_curr = cv2.resize(frame, (64, 64))
                 small_last = cv2.resize(self.last_saved_frame, (64, 64))
                 gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
                 gray_last = cv2.cvtColor(small_last, cv2.COLOR_BGR2GRAY)
                 diff_score = np.mean(cv2.absdiff(gray_curr, gray_last))
                 
                 if diff_score < self.diff_threshold:
                     return
             except Exception as e:
                 print(f"Warning in diff check: {e}")
        
        # --- Perform Save ---
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_frame_{frame_id}" if frame_id is not None else ""
        filename = f"{self.output_dir}/{timestamp}{suffix}_{label}_{confidence:.2f}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"🚨 Anomaly Saved: {os.path.basename(filename)} (Diff: {diff_score:.1f})")
        
        self.last_saved_frame = frame.copy()
        self.last_save_time = current_time
        
        # --- Send Email (with separate cooldown) ---
        if (current_time - self.last_email_time) > self.email_cooldown:
            self.send_email_async(filename, label)
            self.last_email_time = current_time

    def process_frame(self, frame):
        # Predict
        result = self.detector.predict_image(frame)
        
        label = result['class']
        confidence = result['confidence']
        probabilities = result.get('probabilities', {})
        
        # Display logic
        color = (0, 255, 0) # Green for Normal
        if label != 'Normal':
            if confidence >= self.threshold:
                color = (0, 0, 255) # Red for high confidence Anomaly
            else:
                color = (0, 165, 255) # Orange for low confidence Anomaly
        
        # Draw annotation
        annotated_frame = frame.copy()
        
        # 1. Main Prediction Box
        text = f"{label}: {confidence:.1%}"
        cv2.rectangle(annotated_frame, (5, 5), (350, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # 2. Detailed Probabilities
        num_classes = len(probabilities)
        box_height = 20 + (num_classes * 25)
        cv2.rectangle(annotated_frame, (5, 60), (300, 60 + box_height), (0, 0, 0), -1)
        
        y_offset = 85
        for cls_name, prob in probabilities.items():
            t_color = (180, 180, 180)
            if cls_name == label:
                t_color = color 
            elif cls_name != 'Normal' and prob > 0.2:
                 t_color = (0, 255, 255) 

            cv2.putText(annotated_frame, f"{cls_name}: {prob:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_color, 1)
            
            bar_len = int(prob * 100)
            cv2.rectangle(annotated_frame, (180, y_offset - 10), (180 + bar_len, y_offset + 2), t_color, -1)
            y_offset += 25
            
        return result, annotated_frame

    def run_webcam(self):
        print("\n🎥 Starting Webcam... Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result, annotated_frame = self.process_frame(frame)
            label = result['class']
            confidence = result['confidence']

            if label != 'Normal' and confidence >= self.threshold:
                self.check_and_save_anomaly(frame, label, confidence)

            cv2.imshow('Crime Detector - Webcam', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def run_video(self, video_path):
        print(f"\n🎬 Processing Video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total Frames: {total_frames}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result, annotated_frame = self.process_frame(frame)
            label = result['class']
            confidence = result['confidence']

            if label != 'Normal' and confidence >= self.threshold:
                frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.check_and_save_anomaly(frame, label, confidence, frame_id)

            cv2.imshow('Crime Detector - Video', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def run_image(self, image_path):
        print(f"\n🖼️ Processing Image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Error: Could not read image.")
            return

        result, annotated_frame = self.process_frame(image)
        label = result['class']
        confidence = result['confidence']

        print(f"🎯 Prediction: {label} ({confidence:.1%})")
        print("Probabilities:", result.get('probabilities'))
        
        if label != 'Normal' and confidence >= self.threshold:
             self.check_and_save_anomaly(image, label, confidence)
        
        cv2.imshow('Crime Detector - Image', annotated_frame)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Evaluate Crime Detector Model')
    parser.add_argument('--option', type=str, choices=['image', 'video', 'webcam'], help='Input source')
    parser.add_argument('--input', type=str, help='Path to input')
    parser.add_argument('--model', type=str, default='dataset/crime_detector_refined.pth', help='Path to .pth model')
    parser.add_argument('--threshold', type=float, default=0.7, help='Confidence threshold')

    args = parser.parse_args()
    
    if not args.option:
        print("\n🔍 Select Input Option:")
        print("1. Image")
        print("2. Video")
        print("3. Webcam")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == '1':
             args.option = 'image'
             args.input = input("Enter image path: ").strip()
        elif choice == '2':
             args.option = 'video'
             args.input = input("Enter video path: ").strip()
        elif choice == '3':
             args.option = 'webcam'
        else:
             print("Invalid choice.")
             return

    model_full_path = os.path.abspath(args.model)
    if not os.path.exists(model_full_path):
         local_path = os.path.join(os.path.dirname(__file__), 'dataset', 'crime_detector_refined.pth')
         if os.path.exists(local_path):
             print(f"⚠️  Model arg not found, found at {local_path}, using that.")
             model_full_path = local_path
         else:
             print(f"❌ Error: Model file not found at {model_full_path}")
             return

    evaluator = AnomalyEvaluator(model_full_path, threshold=args.threshold)

    if args.option == 'webcam':
        evaluator.run_webcam()
    elif args.option == 'video':
        if not args.input:
            print("❌ Error: Input path required for video.")
        else:
            evaluator.run_video(args.input)
    elif args.option == 'image':
        if not args.input:
            print("❌ Error: Input path required for image.")
        else:
            evaluator.run_image(args.input)

if __name__ == "__main__":
    main()
