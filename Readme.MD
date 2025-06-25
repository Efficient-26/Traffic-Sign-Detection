# Traffic Signal Detection

This project implements a Convolutional Neural Network (CNN) for traffic signal detection, classifying various German traffic signs. The model is trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset and deployed using Flask for a simple web inference.

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Model Architecture](https://www.google.com/search?q=%23model-architecture)
  - [Setup and Installation](https://www.google.com/search?q=%23setup-and-installation)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [File Structure](https://www.google.com/search?q=%23file-structure)
  - [Results](https://www.google.com/search?q=%23results)
  - [Contributing](https://www.google.com/search?q=%23contributing)
  - [License](https://www.google.com/search?q=%23license)

-----

## Project Overview

The goal of this project is to accurately identify different traffic signs from images. A custom CNN model is designed and trained using TensorFlow/Keras. After training, the model is saved and integrated into a Flask web application, allowing users to upload an image and receive a prediction of the traffic sign it contains.

-----

## Dataset

The model is trained on the **GTSRB (German Traffic Sign Recognition Benchmark) Challenge** dataset. This dataset contains images of 43 different traffic signs.

The dataset structure expected by the code is:

```
GTSRB_Challenge/
├── train/
│   ├── class_00000/
│   │   ├── 00000_00000.png
│   │   └── ...
│   ├── class_00001/
│   │   ├── 00001_00000.png
│   │   └── ...
│   └── ... (up to 20 classes are handled by the `classes` dictionary)
└── test/
    └── ...
```

**Note**: Your initial data processing script converts all `.png` images in the `train` directory to `.jpg` and reorganizes them into class-specific subdirectories, which is then used by the `ImageDataGenerator`.

-----

## Model Architecture

The CNN model is built using `Sequential` API in Keras and consists of multiple convolutional layers, max-pooling layers, and dense layers. Dropout is included to prevent overfitting.

  - **Input Layer**: Expects images of size `224x224` with 3 color channels (RGB).
  - **Convolutional Layers**:
      - `Conv2D(32, (5, 5), activation='relu')`
      - `MaxPooling2D(pool_size=(2, 2))`
      - `Conv2D(32, (3, 3), activation='relu')`
      - `MaxPooling2D(pool_size=(2, 2))`
      - `Conv2D(32, (5, 5), activation='relu')`
      - `MaxPooling2D(pool_size=(2, 2))`
      - `Conv2D(64, (3, 3), activation='relu')`
      - `MaxPooling2D(pool_size=(2, 2))`
      - `Conv2D(64, (3, 3), activation='relu')`
      - `MaxPooling2D(pool_size=(2, 2))`
  - **Flatten Layer**: Converts 3D feature maps to 1D feature vectors.
  - **Dense Layers**:
      - `Dense(32, activation='relu')`
      - `Dense(64, activation='relu')`
      - `Dense(256, activation='relu')`
  - **Dropout Layer**: `Dropout(0.5)` for regularization.
  - **Output Layer**: `Dense(21, activation='softmax')` for classification into 21 classes (as defined in the `classes` dictionary).

The model is compiled with `categorical_crossentropy` loss and `adam` optimizer.

-----

## Setup and Installation

To run this project, you'll need Python and the following libraries.

1.  **Clone the repository:** (If applicable, assuming this code is part of a larger repo. If not, you can omit this step or instruct to download the files.)

2.  **Unzip the dataset:**
    Make sure you have `GTSRB_Challenge.zip` in your working directory and run the provided unzip command:

    ```bash
    !unzip GTSRB_Challenge.zip
    ```

3.  **Install dependencies:**

    ```bash
    pip install tensorflow numpy matplotlib pillow Flask werkzeug SQLAlchemy
    ```

    *Note: `SQLAlchemy` is listed in your `pip install` commands but isn't used in the provided Python script. You can remove it if it's not actually a dependency for your Flask app.*

4.  **Prepare the dataset (if not already done by the notebook):**
    The initial script handles this automatically by converting `.png` files to `.jpg` and organizing them. Ensure this part of the code runs to prepare your `train` directory correctly.

-----

## Usage

### 1\. Data Preparation and Training

Run the Jupyter Notebook (or the Python script if you've extracted the code) step-by-step:

  - **Unzip and Preprocess Images:** The first few cells handle unzipping the dataset and converting PNG images to JPEG format, structuring them for `ImageDataGenerator`.
  - **Model Definition and Compilation:** The CNN model is defined and compiled.
  - **Data Augmentation and Loading:** `ImageDataGenerator` is used for real-time data augmentation and loading images from the `train` directory.
  - **Model Training:** The model is trained using the `fit` method.
  - **Plotting Results:** Loss and accuracy plots are generated.
  - **Save the Model:** The trained model is saved as `signal_model2.h5`.

### 2\. Running the Flask Web Application

There are two Flask application scripts provided. Choose the one relevant to your deployment environment:

#### a) For Google Colab (or similar Linux/Unix environment):

The first Flask app block (where `MODEL_PATH = '/content/signal_model2.h5'`) is set up for environments like Google Colab.

1.  Ensure you have saved the `signal_model2.h5` model in the `/content/` directory.
2.  Run the Flask application code. You might need to use `ngrok` or similar tools to expose the Flask app running on Colab to the internet.
    ```python
    if __name__ == '__main__':
        app.run(port=5001,debug=True)
    ```

#### b) For Local Windows Machine:

The second Flask app block (where `MODEL_PATH = 'C:/Users/91700/dl deploy/signal_model2.h5'`) is configured for a local Windows machine.

1.  Save the `signal_model2.h5` model to the specified path: `C:\Users\91700\dl deploy\signal_model2.h5`.
2.  Save the Flask application code as a Python file (e.g., `app.py`).
3.  Open your command prompt in the directory where `app.py` is saved.
4.  Run the application:
    ```bash
    python app.py
    ```

The application will typically run on `http://127.0.0.1:5001/` (or `http://localhost:5001/`).

### 3\. Making Predictions

Once the Flask app is running:

1.  Open your web browser and navigate to the Flask application's URL (e.g., `http://127.0.0.1:5001/`).
2.  You should see an `index.html` page (which is expected to be in a `templates` folder alongside your `app.py`). This page will have a file upload interface.
3.  Upload an image of a traffic sign.
4.  The application will process the image and return the predicted traffic sign label.

-----

## File Structure

The project expects the following basic structure:

```
.
├── GTSRB_Challenge.zip
├── Traffic-Signal-Detection.ipynb  # The Jupyter Notebook containing your code
├── signal_model2.h5                # The trained model file
├── app.py                          # Flask application (or similar for deployment)
├── templates/                      # Folder for HTML templates
│   └── index.html                  # Basic HTML file for file upload
└── uploads/                        # Folder to temporarily store uploaded images
```

*Note: You'll need to create the `templates` and `uploads` directories manually in your deployment environment if they don't exist.*

-----

## Results

After training, the model's performance can be evaluated by observing the generated loss and accuracy plots:

  - `LossVal_loss.png`: Shows the training loss and validation loss over epochs.
  - `AccVal_acc.png`: Shows the training accuracy and validation accuracy over epochs.

The `model.summary()` output provides details about the layers and trainable parameters of the CNN.

-----
