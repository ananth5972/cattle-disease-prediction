from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Initialize the interpreter
interpreter = tf.lite.Interpreter(model_path="modelnewacc.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Function to preprocess the input image
def preprocess_image(image_bytes, input_details):
    image = Image.open(io.BytesIO(image_bytes)).resize((input_details['shape'][1], input_details['shape'][2]))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize image to [0, 1]

    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        image = image / input_scale + input_zero_point

    image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
    return image

# Function to predict the category
def predict(image_bytes):
    test_image = preprocess_image(image_bytes, input_details)

    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])[0]
    confidence = np.max(output)  # Get the confidence of the highest prediction
    predicted_label = np.argmax(output)  # Get the predicted label
    return predicted_label, confidence, output

# Mapping predictions to disease descriptions and treatments
def get_disease_description(predicted_label):
    if predicted_label == 0:
        return ("Foot And Mouth Disease", "Treatment recommendation: Clean the infected area with antiseptic solution, then apply Zinc Oxide ointment or Gentian Violet on the sores. Consult a vet for additional antibiotic treatment like Penicillin or Streptomycin.")
    elif predicted_label == 1:
        return ("Infectious Bovine Keratoconjunctivitis", "Treatment recommendation: Use antibiotic eye ointment like Oxytetracycline or Terramycin. In severe cases, administer intramuscular antibiotics such as Tylosin or LA-200.")
    elif predicted_label == 2:
        return ("Lumpy Skin Disease", "Treatment recommendation: Administer anti-inflammatory drugs and antibiotics like Oxytetracycline to prevent secondary infections. Apply wound care ointments like Iodine or Zinc Oxide on the skin lesions.")
    elif predicted_label == 3:
        return ("Healthy", "No treatment needed.")
    else:
        return ("Unknown Disease", "No recommendation available")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        image_bytes = file.read()
        predicted_label, confidence, output = predict(image_bytes)

        # Convert the output (which might contain uint8) to a list or float
        output = output.tolist()  # Convert to a list if it's an array
        confidence = float(confidence)  # Ensure confidence is a float

        # Get the disease description and treatment recommendation
        disease, treatment = get_disease_description(predicted_label)

        # Return the result as JSON
        return jsonify({
            "predicted_disease": disease,
            "confidence": round(confidence, 2),
            "treatment": treatment,
            "output": output  # Ensure output is serializable
        })

if __name__ == "__main__":
    app.run(debug=True)
