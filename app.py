import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_swagger_ui import get_swaggerui_blueprint
import os

# Patch Keras base Layer class to bypass 'quantization_config' compatibility issues in Keras 3
try:
    from keras.layers import Layer
    orig_layer_init = Layer.__init__
    def patched_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        orig_layer_init(self, *args, **kwargs)
    Layer.__init__ = patched_layer_init
    print("Keras Layer base class successfully patched for Keras 3 compatibility.")
except Exception as e:
    print(f"Warning: Failed to patch Keras Layer: {e}")

# Define the Custom Layer required for loading the model
class CustomDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, activation='relu', use_batchnorm=True, **kwargs):
        super(CustomDenseBlock, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_batchnorm = use_batchnorm
        self.dense = tf.keras.layers.Dense(units)
        if self.use_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
        self.act_layer = tf.keras.layers.Activation(activation)

    def call(self, inputs):
        x = self.dense(inputs)
        if self.use_batchnorm: x = self.batchnorm(x)
        return self.act_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'activation': self.activation, 'use_batchnorm': self.use_batchnorm})
        return config

app = Flask(__name__, static_folder='static')

# --- Swagger UI Configuration ---
SWAGGER_URL = '/docs'
API_URL = '/static/swagger.json'
swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Sistem Rekomendasi API - KaloriKu",
        'layout': "BaseLayout",
        'docExpansion': "list",
        'defaultModelsExpandDepth': 2,
        'defaultModelExpandDepth': 2,
        'tryItOutEnabled': True
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

# Global variables for the models
model_rating_path = './best_rating_prediction_model.keras'
model_rec_path = './nlp_recommendation_model.keras'

model_rating = None
model_rec = None

if os.path.exists(model_rating_path):
    model_rating = tf.keras.models.load_model(model_rating_path, custom_objects={'CustomDenseBlock': CustomDenseBlock})
    print("Rating model loaded successfully.")
else:
    print(f"Warning: Rating model not found at {model_rating_path}")

if os.path.exists(model_rec_path):
    model_rec = tf.keras.models.load_model(model_rec_path, custom_objects={'CustomDenseBlock': CustomDenseBlock})
    print("Recommendation model loaded successfully.")
else:
    print(f"Warning: Recommendation model not found at {model_rec_path}")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sistem Rekomendasi API</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-primary: #818cf8;
            --accent-secondary: #34d399;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --glow-color: rgba(129, 140, 248, 0.15);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 800px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 50px var(--glow-color);
            position: relative;
        }

        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }

        .title-area h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--accent-primary), #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .title-area p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .status-badge {
            background: rgba(52, 211, 153, 0.15);
            border: 1px solid var(--accent-secondary);
            color: var(--accent-secondary);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-secondary);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--accent-secondary);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(52, 211, 153, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--accent-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .endpoint-card {
            background: rgba(15, 23, 42, 0.4);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }

        .endpoint-card:hover {
            transform: translateY(-2px);
            border-color: rgba(129, 140, 248, 0.3);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }

        .endpoint-header {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .method {
            background: var(--accent-primary);
            color: #0f172a;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.75rem;
        }

        .path {
            font-family: monospace;
            font-size: 1.1rem;
            font-weight: 600;
            color: #f8fafc;
        }

        .description {
            color: var(--text-secondary);
            font-size: 0.95rem;
            margin-bottom: 1rem;
            line-height: 1.5;
        }

        pre {
            background: #090d16;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 1rem;
            overflow-x: auto;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.85rem;
            color: #38bdf8;
        }

        .tech-stack {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
            margin-top: 2rem;
            border-top: 1px solid var(--card-border);
            padding-top: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .tech-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title-area">
                <h1>Sistem Rekomendasi API</h1>
                <p>Nutritional & Recipe Prediction Services</p>
            </div>
            <div class="status-badge">
                <span class="status-dot"></span>
                <span>Active</span>
            </div>
        </div>

        <h2 class="section-title">API Endpoints</h2>

        <!-- Endpoint 1 -->
        <div class="endpoint-card">
            <div class="endpoint-header">
                <span class="method">POST</span>
                <span class="path">/predict_rating</span>
            </div>
            <p class="description">Predicts food rating based on nutritional inputs, category, and recipe text parameters.</p>
            <p class="description" style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);">Example Request Payload:</p>
            <pre>{
  "Title_processed": "sayur bayam bening",
  "Ingredients_processed": "bayam, bawang merah, bawang putih, temu kunci, garam, air",
  "Steps_processed": "rebus air, masukkan bawang dan temu kunci, masukkan bayam, bumbui",
  "jumlah_kalori_normalized": 0.12,
  "usia_normalized": 0.35,
  "Food_Category": "Sayuran"
}</pre>
        </div>

        <!-- Endpoint 2 -->
        <div class="endpoint-card">
            <div class="endpoint-header">
                <span class="method">POST</span>
                <span class="path">/predict_embedding</span>
            </div>
            <p class="description">Generates high-dimensional embedding vector representations for similarity and recommendations.</p>
            <p class="description" style="font-weight: 600; margin-bottom: 0.5rem; color: var(--text-primary);">Example Request Payload:</p>
            <pre>{
  "Title_processed": "sayur bayam bening",
  "Ingredients_processed": "bayam, bawang merah, bawang putih, temu kunci, garam, air",
  "Steps_processed": "rebus air, masukkan bawang dan temu kunci, masukkan bayam, bumbui",
  "jumlah_kalori_normalized": 0.12,
  "usia_normalized": 0.35,
  "Food_Category": "Sayuran"
}</pre>
        </div>

        <div style="text-align: center; margin-top: 2rem;">
            <a href="/docs" style="display: inline-flex; align-items: center; gap: 0.75rem; background: linear-gradient(135deg, #818cf8, #6366f1); color: #fff; padding: 0.85rem 2rem; border-radius: 12px; text-decoration: none; font-weight: 600; font-size: 1rem; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(129, 140, 248, 0.35);" onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 8px 25px rgba(129,140,248,0.5)'" onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 4px 15px rgba(129,140,248,0.35)'">
                📖 Buka Swagger API Documentation
            </a>
        </div>

        <div class="tech-stack">
            <div class="tech-item">⚡ Flask Backend</div>
            <div class="tech-item">🧠 TensorFlow</div>
            <div class="tech-item">🐳 Docker Space</div>
            <div class="tech-item">🤗 Hugging Face</div>
        </div>
    </div>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    if model_rating is None: return jsonify({"error": "Rating model not available"}), 500
    data = request.json
    try:
        inputs = {
            'title_input_hp': tf.constant([data['Title_processed']], dtype=tf.string),
            'ingredients_input_hp': tf.constant([data['Ingredients_processed']], dtype=tf.string),
            'steps_input_hp': tf.constant([data['Steps_processed']], dtype=tf.string),
            'numerical_input_hp': tf.constant([[data['jumlah_kalori_normalized'], data['usia_normalized']]], dtype=tf.float32),
            'categorical_input_hp': tf.constant([data['Food_Category']], dtype=tf.string)
        }
        pred = model_rating.predict(inputs)
        return jsonify({"predicted_rating": float(pred[0][0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict_embedding', methods=['POST'])
def predict_embedding():
    if model_rec is None: return jsonify({"error": "Recommendation model not available"}), 500
    data = request.json
    try:
        # Inputs untuk model nlp_recommendation_model menggunakan key tanpa suffix '_hp'
        inputs = {
            'title_input': tf.constant([data['Title_processed']], dtype=tf.string),
            'ingredients_input': tf.constant([data['Ingredients_processed']], dtype=tf.string),
            'steps_input': tf.constant([data['Steps_processed']], dtype=tf.string),
            'numerical_input': tf.constant([[data['jumlah_kalori_normalized'], data['usia_normalized']]], dtype=tf.float32),
            'categorical_input': tf.constant([data['Food_Category']], dtype=tf.string)
        }
        pred = model_rec.predict(inputs)
        return jsonify({"predicted_embedding": pred[0].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)
