# Plant Disease Prediction Using CNN

This project is a web application that predicts plant diseases using a Convolutional Neural Network (CNN). The application is built using TensorFlow for the machine learning model and Streamlit for the user interface.

## Features
- Upload an image of a plant leaf.
- The application preprocesses the image and predicts the disease class.
- Displays the prediction result with a user-friendly interface.

## Project Structure
```
app/
    class_indices.json          # JSON file containing class indices for prediction
    main.py                     # Main application script
    trained_model/
        plant_disease_prediction_model.h5  # Pre-trained CNN model
```

## Requirements
- Python 3.7 or higher
- Required Python libraries:
  - TensorFlow
  - Streamlit
  - Pillow
  - NumPy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<username>/Plant_Disease_Prediction_Using_CNN.git
   cd Plant_Disease_Prediction_Using_CNN
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app/main.py
   ```

## Usage
1. Open the application in your browser (Streamlit will provide a local URL).
2. Upload an image of a plant leaf in JPG, JPEG, or PNG format.
3. Click the "Classify" button to get the prediction.

## Model
The pre-trained model (`plant_disease_prediction_model.h5`) is a CNN trained on a dataset of plant leaf images. The model predicts the class of the disease based on the uploaded image.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- TensorFlow for the machine learning framework.
- Streamlit for the web application framework.
- Dataset used for training the model (if applicable).