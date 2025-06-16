step 1: Create a Virtual Environment

Let's create a virtual environment to keep dependencies isolated:
python3.10 -m venv env

Activate the environment:
source env/bin/activate

upgrade pip
pip install --upgrade pip
pip install --upgrade pip setuptools wheel


Step 2: Install TensorFlow for Mac
If you have an Apple Silicon (M1/M2/M3) Mac, install the optimized version:
pip install tensorflow-macos tensorflow-metal

Step 3: Verify Installation
Once installed, check if TensorFlow is correctly installed:
python -c "import tensorflow as tf; print(tf.__version__)"

Step 4: Install Additional Libraries
Now, install all the other required libraries:
pip install numpy matplotlib opencv-python-headless scikit-learn seaborn imbalanced-learn

Step 5: Run the code
python filename 





