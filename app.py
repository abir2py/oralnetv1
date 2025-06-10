# app.py

from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import io
import base64
import pandas as pd
import joblib
import logging

# Configure logging for better visibility of server actions and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Flask application instance
# This line must be at the top level of the module so Flask can find it.
app = Flask(__name__)

# --- PyTorch Model Loading ---
# Using @app.before_first_request or loading globally ensures models are ready
# when the first request comes in. @app.before_first_request is slightly safer
# if model loading is very heavy and might delay app startup.
# For simplicity, loading globally here as it's typical for inference servers.
def load_pytorch_model():
    logging.info("Attempting to load PyTorch model 'resnet_model.pth'...")
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid() # Sigmoid for binary classification probability output
    )
    try:
        # Load the state_dict directly. map_location='cpu' ensures it loads on CPU
        # even if trained on GPU, making it portable for server deployment.
        state_dict = torch.load('resnet_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval() # Set model to evaluation mode (important for dropout/batchnorm layers)
        logging.info("PyTorch model 'resnet_model.pth' loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error("ERROR: PyTorch model 'resnet_model.pth' not found. Please ensure it's in the same directory as app.py.")
        return None
    except Exception as e:
        logging.error(f"ERROR: Failed to load PyTorch model: {e}")
        return None

# --- Scikit-learn Model Loading ---
def load_sklearn_model():
    logging.info("Attempting to load Scikit-learn model 'random_forest_model.pkl'...")
    try:
        # *** CHANGED MODEL FILENAME HERE ***
        model = joblib.load('random_forest_model.pkl')
        logging.info("Scikit-learn model 'random_forest_model.pkl' loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error("ERROR: Scikit-learn model 'random_forest_model.pkl' not found. Please ensure it's in the same directory as app.py.")
        return None
    except Exception as e:
        logging.error(f"ERROR: Failed to load Scikit-learn model: {e}")
        return None

# Load models globally when the Flask app starts.
# These variables will hold your loaded models.
pytorch_model = load_pytorch_model()
sklearn_model = load_sklearn_model()

# --- Prediction Functions ---

def predict_image(image_bytes, model):
    """
    Predicts oral cancer class from an image using the PyTorch model.
    Args:
        image_bytes: Raw bytes of the image.
        model: The loaded PyTorch model.
    Returns:
        A dictionary with predicted class and probabilities, or error info.
    """
    if model is None:
        logging.warning("PyTorch model is not loaded, skipping image prediction.")
        return {'predicted_class': 'Model Not Loaded', 'prob_class_0': 0.0, 'prob_class_1': 0.0, 'error': 'PyTorch model not loaded'}

    try:
        # Open image from bytes, convert to RGB (important for models expecting 3 channels)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Define image transformation pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)), # Resize for ResNet input
            transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
        ])
        input_tensor = transform(image).unsqueeze(0) # Add batch dimension (1, C, H, W)

        # Perform inference
        with torch.no_grad(): # Disable gradient calculation for inference (saves memory and speeds up)
            output = model(input_tensor)
            prob_class_1 = output.item() # Get the scalar probability for OSCC
            prob_class_0 = 1 - prob_class_1 # Probability for Leukoplakia
            predicted_class = 'OSCC' if prob_class_1 > 0.5 else 'Leukoplakia'

        logging.info(f"Image prediction: {predicted_class}, OSCC Prob: {prob_class_1:.4f}")
        return {
            'predicted_class': predicted_class,
            'prob_class_0': prob_class_0,
            'prob_class_1': prob_class_1
        }
    except Exception as e:
        logging.error(f"Error during image prediction: {e}", exc_info=True) # exc_info=True to log traceback
        return {'predicted_class': 'Error', 'prob_class_0': 0.0, 'prob_class_1': 0.0, 'error': str(e)}

def predict_dysplasia(data_dict, model):
    """
    Predicts dysplasia severity from tabular data using the Scikit-learn model.
    Args:
        data_dict: Dictionary of input parameters.
        model: The loaded Scikit-learn model.
    Returns:
        A dictionary with predicted severity code, or error info.
    """
    if model is None:
        logging.warning("Scikit-learn model is not loaded, skipping tabular prediction.")
        return {'predicted_severity': -1, 'error': 'Scikit-learn model not loaded'}

    try:
        # Create a Pandas DataFrame from the input dictionary.
        # Ensure that the column order and names match those used during model training.
        # If your model expects certain features or feature engineering, it must be replicated here.
        input_df = pd.DataFrame([data_dict])

        # Define the expected columns in the order your model was trained on.
        # It's CRITICAL that this list matches your training features exactly.
        expected_columns = [
            'localization', 'larger_size', 'tobacco_use', 'alcohol_consumption',
            'sun_exposure', 'gender', 'skin_color', 'age_group', 'dysplasia_severity'
        ]

        # Reindex the DataFrame to ensure correct column order.
        # `fill_value=0` can be used for missing columns if your model can handle it,
        # but it's generally better to ensure all expected inputs are provided.
        input_df = input_df.reindex(columns=expected_columns) # Don't use fill_value unless you're sure about defaults

        # Check for any NaN values after reindexing if inputs are missing or incorrect
        if input_df.isnull().any().any():
            missing_cols = input_df.columns[input_df.isnull().any()].tolist()
            raise ValueError(f"Missing required tabular data for columns: {missing_cols}")

        prediction = model.predict(input_df)
        logging.info(f"Scikit-learn prediction: {int(prediction[0])}")
        return {'predicted_severity': int(prediction[0])}
    except Exception as e:
        logging.error(f"Error during scikit-learn prediction: {e}", exc_info=True)
        return {'predicted_severity': -1, 'error': str(e)}

# --- Fusion Logic ---
# This function combines the results from both models into a single diagnostic message.
def get_combined_diagnosis(image_prob_oscc, sklearn_severity_code):
    """
    Generates a combined diagnostic message based on image and tabular predictions.
    Args:
        image_prob_oscc: Probability of OSCC from the image model.
        sklearn_severity_code: Numerical code for dysplasia severity from tabular model.
    Returns:
        A string representing the combined diagnosis.
    """
    final_diagnosis_message = "Based on the available information: "

    # Map scikit-learn prediction code to a human-readable label
    prediction_map = {
        0: 'No Dysplasia',
        1: 'Mild Dysplasia',
        2: 'Moderate Dysplasia',
        3: 'Severe Dysplasia',
        4: 'Carcinoma in situ'
    }
    sklearn_severity_label = prediction_map.get(sklearn_severity_code, f"Unknown prediction ({sklearn_severity_code})")

    # Simple rule-based fusion logic (can be replaced by a trained fusion model)
    if image_prob_oscc > 0.7: # High confidence for OSCC from image
        final_diagnosis_message += f"There is a **HIGH SUSPICION OF OSCC** based on the image analysis ({(image_prob_oscc*100):.2f}% probability). "
        if sklearn_severity_label in ['Severe Dysplasia', 'Carcinoma in situ']:
            final_diagnosis_message += f"This is further supported by the patient's clinical parameters indicating **{sklearn_severity_label}**. Immediate medical consultation is highly recommended."
        else:
            final_diagnosis_message += "Clinical parameters suggest a less severe dysplasia, but the strong image indication requires careful follow-up."
    elif image_prob_oscc > 0.5: # Moderate confidence for OSCC
        final_diagnosis_message += f"The image suggests **possible OSCC** ({(image_prob_oscc*100):.2f}% probability). "
        if sklearn_severity_label in ['Severe Dysplasia', 'Carcinoma in situ']:
            final_diagnosis_message += f"Combined with clinical parameters showing **{sklearn_severity_label}**, further investigation is strongly advised."
        else:
            final_diagnosis_message += "While clinical parameters are less severe, the image warrants close monitoring and further evaluation."
    else: # Image primarily suggests Leukoplakia
        final_diagnosis_message += f"The image primarily suggests **Leukoplakia** ({((1-image_prob_oscc)*100):.2f}% probability). "
        if sklearn_severity_label in ['Severe Dysplasia', 'Carcinoma in situ']:
            final_diagnosis_message += f"However, the patient's clinical parameters indicate **{sklearn_severity_label}**. This discrepancy highlights the need for careful histological examination and clinical correlation."
        elif sklearn_severity_label in ['Mild Dysplasia', 'Moderate Dysplasia']:
            final_diagnosis_message += f"Clinical parameters also indicate **{sklearn_severity_label}**. Regular follow-up and monitoring are essential."
        else:
            final_diagnosis_message += f"Clinical parameters indicate **{sklearn_severity_label}**. This overall suggests a lower risk, but routine check-ups are always prudent."

    return final_diagnosis_message


# --- API Endpoint ---
@app.route('/predict_diagnosis', methods=['POST'])
def predict_diagnosis_endpoint():
    """
    Handles POST requests for comprehensive diagnosis.
    Expects JSON with 'imageData' (base64) and 'tabularData'.
    Returns JSON with image prediction, tabular prediction, and combined diagnosis.
    """
    if not request.is_json:
        logging.error("Request received is not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image_data_b64 = data.get('imageData')
    tabular_data = data.get('tabularData')

    if not image_data_b64:
        logging.error("Missing 'imageData' in request.")
        return jsonify({"error": "Missing imageData"}), 400
    if not tabular_data:
        logging.error("Missing 'tabularData' in request.")
        return jsonify({"error": "Missing tabularData"}), 400

    try:
        image_bytes = base64.b64decode(image_data_b64)
    except Exception as e:
        logging.error(f"Failed to decode base64 image data: {e}", exc_info=True)
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    # Get predictions from individual models
    image_result = predict_image(image_bytes, pytorch_model)
    sklearn_result = predict_dysplasia(tabular_data, sklearn_model)

    # Determine if combined diagnosis can be generated
    # Only proceed if both individual predictions did not result in an explicit error.
    combined_diagnosis_message = "Could not generate combined diagnosis due to errors in individual model predictions."
    if 'error' not in image_result and 'error' not in sklearn_result:
        prob_oscc = image_result.get('prob_class_1', 0.0)
        sklearn_severity = sklearn_result.get('predicted_severity', -1)
        combined_diagnosis_message = get_combined_diagnosis(prob_oscc, sklearn_severity)
        logging.info("Combined diagnosis generated successfully.")
    else:
        logging.warning("Combined diagnosis skipped due to errors in individual models.")

    response = {
        'image_prediction': image_result,
        'sklearn_prediction': sklearn_result,
        'combined_diagnosis': combined_diagnosis_message
    }
    return jsonify(response)

# --- Server Run Configuration ---
if __name__ == '__main__':
    # When running directly with 'python app.py', this block executes.
    # 'host='0.0.0.0'' makes the server accessible from other devices on the network.
    # 'port=5000' is the default Flask port.
    # 'debug=True' provides helpful error messages during development.
    # IMPORTANT: DO NOT USE debug=True in a production environment.
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000, debug=False)

