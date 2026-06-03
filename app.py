import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
import os

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

app = Flask(__name__)

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
    model_rec = tf.keras.models.load_model(model_rec_path)
    print("Recommendation model loaded successfully.")
else:
    print(f"Warning: Recommendation model not found at {model_rec_path}")

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
