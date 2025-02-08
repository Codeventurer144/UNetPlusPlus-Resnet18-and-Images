import subprocess
import sys

def install_libraries():
    # Installing specific versions of the libraries
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.5.1+cpu"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python==4.11.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==2.2.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas==2.2.3"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml==6.0.2"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations==2.0.2"])

if __name__ == "__main__":
    install_libraries()
    print("Libraries installed successfully.")