import os
import subprocess
import sys

# Force UTF-8 encoding for standard output
sys.stdout.reconfigure(encoding='utf-8')

def main():
    python_exe = r"C:\Users\yogaa\AppData\Local\Programs\Python\Python313\python.exe"
    
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print("==================================================")
        print("    AI SMART HOSTEL ANOMALY DETECTION MENU        ")
        print("==================================================")
        print("  [1] Start Streamlit Web Application (Browser)")
        print("  [2] Open Graphical Desktop Interface (Tkinter)")
        print("  [3] Start Live Webcam Detection (OpenCV)")
        print("  [4] Run Terminal Test (Random Sample Images)")
        print("  [5] Exit")
        print("==================================================")
        
        choice = input("Select an option (1, 2, 3, 4, or 5) and press Enter: ").strip()
        
        if choice == '1':
            print("\nStarting Streamlit Web Application...")
            subprocess.run([python_exe, "-m", "streamlit", "run", "app.py"], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        elif choice == '2':
            print("\nOpening Graphical Desktop Interface...")
            subprocess.run([python_exe, "gui_detector.py"], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        elif choice == '3':
            print("\nStarting Webcam Anomaly Detection...")
            print("Press 'q' in the camera window to close it.\n")
            subprocess.run([python_exe, "evaluate_custom.py", "--option", "webcam", "--model", "crime_detector_refined.pth"], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        elif choice == '4':
            print("\nRunning Test Predictions...")
            subprocess.run([python_exe, "test_model.py"], env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            input("\nPress Enter to return to menu...")
        elif choice == '5':
            print("\nExiting. Good luck with your demo!")
            break
        else:
            input("\nInvalid choice. Press Enter to try again...")

if __name__ == '__main__':
    main()
