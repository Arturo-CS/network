from flask import current_app
import datetime
import jwt
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import h5py
from io import BytesIO
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from PIL import Image
import io
import numpy as np
from flask_pymongo import PyMongo
from flask import Flask, request, jsonify
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError
from bson import ObjectId  # Importa ObjectId desde el módulo bson

app = Flask(__name__)
CORS(app, resources={
    r"/uploadfile": {
        "origins": ["*"]
    },
    r"/uploadfile-blueberry": {
        "origins": ["*"]
    }
})
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['MONGO_URI'] = 'mongodb+srv://ccortegana:FBNF5wOPPSY3YgVn@cluster0.bvjsv3g.mongodb.net/Movil'
mongo = PyMongo(app)

dbUser = mongo.db['User']

# Función para generar token


def generar_token(usuario):
    expiracion = datetime.datetime.utcnow() + datetime.timedelta(days=2)
    payload = {
        '_id': str(usuario['_id']),
        'email': usuario['email'],
        'password': usuario['password'],
        'username': usuario['username'],
        'phone': usuario['phone'],
        'exp': expiracion
    }
    token = jwt.encode(
        payload, current_app.config['SECRET_KEY'], algorithm='HS256')
    return token


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')

        # Verificar si se proporcionaron correo y contraseña
        if not email or not password:
            return jsonify({'error': 'Correo y contraseña son requeridos'}), 400

        user = dbUser.find_one({'email': email})

        # Verificar si el usuario existe
        if not user:
            return jsonify({'error': 'Correo electrónico no encontrado'}), 404

        # Verificar la contraseña
        if user['password'] != password:
            return jsonify({'error': 'Contraseña incorrecta'}), 401

        # Generar y devolver el token
        token = generar_token(user)
        username = user.get('username', '')
        return jsonify({'token': token, 'username': username}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/register', methods=['POST'])
def create_user():
    try:
        user_data = {
            'email': request.json.get('email'),
            'password': request.json.get('password'),
            'username': request.json.get('username'),
            'phone': request.json.get('phone', ''),
        }

        # Verificar si el usuario ya existe por el campo 'email'
        if dbUser.find_one({'email': user_data['email']}):
            return jsonify({'error': 'El usuario con este email ya existe.'}), 400

        result = dbUser.insert_one(user_data)

        return jsonify({'user_id': str(result.inserted_id), 'username': user_data['username']}), 201

    except DuplicateKeyError:
        return jsonify({'error': 'El usuario con este email ya existe.'}), 400

    except PyMongoError as e:
        return jsonify({'error': f'Error en la base de datos: {str(e)}'}), 500

    except Exception as e:
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500



@app.route('/profile/<string:email>', methods=['GET'])
def get_user_profile(email):
    try:
        # Buscar el usuario por el correo electrónico
        user = dbUser.find_one({'email': email})

        # Verificar si el usuario existe
        if user:
            # Convierte el ObjectId a una cadena antes de serializar a JSON
            user['_id'] = str(user['_id'])

            # Eliminar el campo de contraseña antes de devolver el perfil
            user.pop('password', None)
            return jsonify(user)
        else:
            return jsonify({'error': 'Usuario no encontrado'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# RED NEURONAL
model_path = "./plant_disease_detection.h5"  # Replace with the actual path
model = tf.keras.models.load_model(model_path)

categories = {"Apple___Apple_scab": 0, "Apple___Black_rot": 1, "Apple___Cedar_apple_rust": 2, "Apple___healthy": 3, "Blueberry___healthy": 4, "Cherry_(including_sour)___Powdery_mildew": 5, "Cherry_(including_sour)___healthy": 6, "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": 7, "Corn_(maize)___Common_rust_": 8, "Corn_(maize)___Northern_Leaf_Blight": 9, "Corn_(maize)___healthy": 10, "Grape___Black_rot": 11, "Grape___Esca_(Black_Measles)": 12, "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": 13, "Grape___healthy": 14, "Orange___Haunglongbing_(Citrus_greening)": 15, "Peach___Bacterial_spot": 16, "Peach___healthy": 17,
              "Pepper,_bell___Bacterial_spot": 18, "Pepper,_bell___healthy": 19, "Potato___Early_blight": 20, "Potato___Late_blight": 21, "Potato___healthy": 22, "Raspberry___healthy": 23, "Soybean___healthy": 24, "Squash___Powdery_mildew": 25, "Strawberry___Leaf_scorch": 26, "Strawberry___healthy": 27, "Tomato___Bacterial_spot": 28, "Tomato___Early_blight": 29, "Tomato___Late_blight": 30, "Tomato___Leaf_Mold": 31, "Tomato___Septoria_leaf_spot": 32, "Tomato___Spider_mites Two-spotted_spider_mite": 33, "Tomato___Target_Spot": 34, "Tomato___Tomato_Yellow_Leaf_Curl_Virus": 35, "Tomato___Tomato_mosaic_virus": 36, "Tomato___healthy": 37}
categories = {v: k for k, v in categories.items()}


@app.get("/")
def root():
    return {"message": "Hello World"}


def predict_image_label(img):
    # Load the image using TensorFlow and preprocess it
    img = np.expand_dims(img, axis=0)
    # Adjust preprocessing function for your model
    img = keras.applications.resnet.preprocess_input(img)

    predictions = model.predict(img)
    # Display or use the predictions as needed
    index_predict = predictions.argmax()
    confidence_predict = predictions.max()
    label_predict = categories[predictions.argmax()]

    return label_predict, confidence_predict  # Return the label and confidence


@app.post("/uploadfile")
def upload_file_corn():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}, 400)

    file = request.files["file"]
    print(file)
    image = Image.open(file)
    image = image.resize((224, 224))
    if file.filename == "":
        return jsonify({"error": "No selected file"}, 400)

    if file and file.content_type.startswith('image/'):
        label, confidence = predict_image_label(image)
        # healthy = "saludable" if "healthy" in label else "enferma"
        print(confidence)
        return jsonify({"label": label, "confidence": str(confidence)}), 200
    else:
        return jsonify({"error": "Uploaded file is not an image"}, 400)


@app.post("/uploadfile-blueberry")
def upload_file_blueberry():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}, 400)

    file = request.files["file"]
    print(file)
    image = Image.open(file)
    image = image.resize((224, 224))
    if file.filename == "":
        return jsonify({"error": "No selected file"}, 400)

    if file and file.content_type.startswith('image/'):
        label, confidence = predict_image_label(image)
        healthy = "saludable" if "healthy" in label else "enferma"
        print(confidence)
        return jsonify({"label": healthy, "confidence": str(confidence)}), 200
    else:
        return jsonify({"error": "Uploaded file is not an image"}, 400)


if __name__ == '__main__':
    app.run(debug=True, host='ip')
