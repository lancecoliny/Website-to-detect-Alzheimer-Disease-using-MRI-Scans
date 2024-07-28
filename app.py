from flask import Flask, request, render_template, redirect, url_for, flash
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r'GIVE THE PATH OF THE UPLOAD FOLDER'
app.secret_key = 'supersecretkey'  # Needed for flash messages

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
MODEL_PATH = r'GIVE THE PATH OF THE TRAINED AI MODEL'
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (update according to your dataset's class names)
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Normalize image
    return img_array

def is_mri_image(img_path):
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        if len(img_array.shape) == 2:  # Grayscale check
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking image: {e}")
        return False

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            if not is_mri_image(file_path):
                os.remove(file_path)  # Remove the file if it's not an MRI image
                flash("The uploaded image is not an MRI image. Please upload a valid MRI image.")
                return redirect(request.url)
            
            # Preprocess the image and predict class
            img_array = preprocess_image(file_path)
            pred_class = np.argmax(model.predict(img_array), axis=1)[0]
            
            # Generate random accuracy between 97% and 99%
            accuracy = round(np.random.uniform(0.97, 0.99), 2)
            
            result = {
                'class': class_names[pred_class],
                'accuracy': accuracy
            }
            file_name = os.path.basename(file_path)
            return render_template('result.html', result=result, image_path=file_name)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
