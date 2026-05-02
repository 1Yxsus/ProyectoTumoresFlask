from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg') # Necesario para servidores sin interfaz gráfica
import matplotlib.pyplot as plt
# Cambiamos tensorflow por tflite_runtime
import tflite_runtime.interpreter as tflite

app = Flask(__name__, static_folder='static', template_folder='templates', static_url_path='')
CORS(app)

# Configuración del interprete TFLite
interpreter = tflite.Interpreter(model_path='brain_tumor_cnn.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_with_tflite(img_array):
    img_array = img_array.astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

@app.route('/api/clasificar', methods=['POST'])
def clasificar_api():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No se envió imagen'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocesamiento con PIL (reemplaza a keras.preprocessing)
    img = Image.open(filepath).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = predict_with_tflite(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    probabilities = {class_names[i]: float(f"{prob:.4f}") for i, prob in enumerate(prediction)}

    # Generación de la gráfica
    plt.figure(figsize=(6, 4))
    plt.bar(probabilities.keys(), probabilities.values(), color='skyblue')
    plt.title('Probabilidades por clase')
    plt.ylabel('Confianza')
    plt.tight_layout()
    
    graph_path = os.path.join(app.config['UPLOAD_FOLDER'], 'probabilidades.png')
    plt.savefig(graph_path)
    plt.close()

    return jsonify({
        'prediction': f'Predicción: {predicted_class.upper()}',
        'image_name': filename,
        'probs': probabilities,
        'graph_url': '/static/uploads/probabilidades.png'
    })

# RUTA CRÍTICA: Sirve el index.html de React
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    # Usar el puerto que asigne Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)