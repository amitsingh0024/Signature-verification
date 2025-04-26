import os
import subprocess
import sys
import webbrowser
from config.config import PATHS

def launch_tensorboard():
    # Get the absolute path to the runs directory
    runs_dir = os.path.abspath(PATHS['tensorboard'])
    
    # Check if the directory exists
    if not os.path.exists(runs_dir):
        print(f"Error: TensorBoard log directory not found at {runs_dir}")
        print("Please make sure you have run the training script to generate logs.")
        sys.exit(1)
    
    # Launch TensorBoard
    print(f"Launching TensorBoard with log directory: {runs_dir}")
    print("TensorBoard will be available at http://localhost:6006")
    
    # Start TensorBoard in a subprocess
    tensorboard_process = subprocess.Popen(
        ['tensorboard', '--logdir', runs_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Open the TensorBoard URL in the default web browser
    webbrowser.open('http://localhost:6006')
    
    try:
        # Wait for the process to complete
        tensorboard_process.wait()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nShutting down TensorBoard...")
        tensorboard_process.terminate()
        tensorboard_process.wait()

if __name__ == "__main__":
    launch_tensorboard() 