ABSTRACT:
The increasing prevalence of digital authentication has led to a greater need for efficient, strong face anti-spoofing systems that protect privacy and security. Conventional systems often fall short against advanced attacks, such as print and replay, particularly in real-time scenarios. This study presents a face anti-spoofing model that merges MobileViT and CNN architectures, bolstered by multi-scale adaptive attention and data augmentation derived from StyleGAN2, to enhance the model's accuracy and flexibility. By targeting issues related to computational efficiency and privacy, the model employs techniques such as model quantization, knowledge distillation, and model pruning with ONNX execution. These enhancements are designed to ensure high performance in environments with limited resources, resulting in a lightweight yet effective model suitable for EdgeAI implementation. Important factors include privacy and adherence to regulations like Aadhar guidelines; the model integrates secure WebAssembly (WASM) and the Web Crypto API for safeguarding data while utilizing multi-scale replay attack detection and dynamic switching between active and passive detection for reliable spoofing identification. Temporal analysis facilitated by optical flow strengthens detection abilities, whereas model partitioning and progressive loading via a PWA enable compatibility with web platforms. By incorporating EdgeAI for distributing workloads, this solution promotes cost-effectiveness and scalability across various edge settings. Initial findings show significant advancements in speed, precision, and adaptability focused on privacy, establishing this model as a groundbreaking addition to the anti-spoofing research domain. The implications reach further than mere technical improvements, offering a scalable, regulation-compliant solution that emphasizes security and privacy, rendering it an accessible, sustainable method for digital authentication. Therefore, this research not only addresses existing gaps in the field but also establishes new benchmarks for face anti-spoofing technology in secure, real-time settings.

# Face Liveness Detection

This project implements a Face Liveness Detection system using deep learning techniques. It leverages a pre-trained ResNet-18 model to classify input images as either "Live" or "Spoof." The model is trained on a custom dataset and exported to the ONNX format for inference.

## Project Overview

The system is designed to detect whether a face in a given image or video is real (live) or fake (spoofed) by analyzing facial features using a trained convolutional neural network (CNN).

### Key Features
- **Model Training**: Train a deep learning model using a custom dataset.
- **Inference**: Use the trained model (in ONNX format) to perform real-time face liveness detection.
- **Gradio Interface**: A user-friendly web interface to interact with the model for training and testing.

## Installation

1. Clone the Repository
Clone the repository to your local machine using:
```bash
git clone https://github.com/your-username/your-repository-name.git

2. Install Dependencies
Install the required libraries using pip:

pip install -r requirements.txt

3. Set Up Files
Ensure that the following files are present:

model.onnx - The trained model in ONNX format.
dataset/training - The directory containing the training images (if you're retraining the model).

Usage
1. Train the Model
To train the model on your custom dataset, run the following:

python app.py

2. Test the Model
Once the model is trained, you can test it by uploading images or videos in the Test Model tab. The system will classify the uploaded content as either "Live" or "Spoof."
/your-repository-name
│
├── app.py             # Main application file with model training and inference
├── model.onnx         # Trained model in ONNX format
├── dataset/           # Directory containing training dataset
│   ├── training/      # Folder with training images
├── requirements.txt   # List of required Python packages
└── README.md          # This README file

Dependencies
PyTorch: For training the model and model operations.
ONNX: For exporting the trained model and running inference.
Gradio: For creating the web-based user interface.
OpenCV: For image and video preprocessing.
Matplotlib/Seaborn: For visualizations (optional).

To install the dependencies:
pip install torch torchvision gradio opencv-python onnx onnxruntime matplotlib seaborn

Contributing:
Feel free to open issues and contribute to the repository. Pull requests are welcome!



