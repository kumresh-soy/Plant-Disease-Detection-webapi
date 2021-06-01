from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from PIL import Image
# TensorFlow and tf.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

app = Flask(__name__)
# You can use pretrained model from Keras

model = MobileNetV2(weights='imagenet')

# Model saved with Keras model.save()
MODEL_PATH = 'models/my_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

def model_predict(img, model):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds

Plant_disease_list = [
    'Bael disease',
    'Gauva disease',
    'Gauva healthy',
    'Jamun disease',
    'Jamun healthy',
    'Lemon disease',
    'Lemon healthy',
    'Mango disease',
    'Mango healthy',
    'Pomegranate disease',
    'Pomegranate healthy',]


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    
    img.save("./uploads/image.png")
    preds = model_predict(img, model)

    pred_proba = "{:.3f}".format(np.amax(preds))
    result = np.argmax(preds)
    label = Plant_disease_list[result]

    return jsonify({'result': label, 'score': pred_proba})


if __name__ == '__main__':
    # app.run()
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
