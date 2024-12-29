import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the interpreter
interpreter = tf.lite.Interpreter(model_path="modelnewacc.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# Function to preprocess the input image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((input_details['shape'][1], input_details['shape'][2]))
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize image to [0, 1]
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        image = image / input_scale + input_zero_point
    image = np.expand_dims(image, axis=0).astype(input_details["dtype"])
    return image

# Function to predict the category
def predict(image_bytes):
    test_image = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    predicted_label = np.argmax(output)
    confidence = np.max(output)
    return predicted_label, confidence

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

# Streamlit app layout
st.title('Cattle Disease Prediction')
st.write('Upload an image of your cattle to get a disease prediction.')

# File uploader
file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if file is not None:
    # Display the uploaded image
    st.image(file, caption="Uploaded Image.", use_column_width=True)

    # Predict the disease
    predicted_label, confidence = predict(file.read())
    disease, treatment = get_disease_description(predicted_label)
    
    # Display the result
    st.write(f"Predicted Disease: {disease}")
    st.write(f"Confidence: {confidence:.2f}")
    st.write(f"Treatment: {treatment}")
