import os
import sys
import zipfile
import json
import shutil

# Mock Windows-incompatible dependencies before importing tensorflowjs
from unittest.mock import MagicMock
sys.modules['tensorflow_decision_forests'] = MagicMock()
sys.modules['jax'] = MagicMock()
sys.modules['jax.experimental'] = MagicMock()

try:
    import tensorflow as tf
    import tensorflowjs as tfjs
    import keras
    
    # Patch Keras Layer.__init__ to remove 'quantization_config' which is unsupported in Keras 3.12.2
    # but present in models saved with Keras 3.13.2+
    original_layer_init = keras.layers.Layer.__init__
    def patched_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        original_layer_init(self, *args, **kwargs)
    keras.layers.Layer.__init__ = patched_layer_init
    
    # Define the CustomDenseBlock class matching the exact sub-layer naming in the saved model weights
    class CustomDenseBlock(keras.layers.Layer):
        def __init__(self, units=32, activation='relu', use_batchnorm=True, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.activation = activation
            self.use_batchnorm = use_batchnorm
            
            # Sub-layers matching the weights structure keys: 'dense', 'batchnorm', 'act_layer'
            self.dense = keras.layers.Dense(self.units, activation=None, name="dense")
            if self.use_batchnorm:
                self.batchnorm = keras.layers.BatchNormalization(name="batchnorm")
            self.act_layer = keras.layers.Activation(self.activation, name="act_layer")
            
        def build(self, input_shape):
            self.dense.build(input_shape)
            if self.use_batchnorm:
                bn_input_shape = list(input_shape)[:-1] + [self.units]
                self.batchnorm.build(tuple(bn_input_shape))
            self.act_layer.build(list(input_shape)[:-1] + [self.units])
            super().build(input_shape)
                
        def call(self, inputs):
            x = self.dense(inputs)
            if self.use_batchnorm:
                x = self.batchnorm(x)
            x = self.act_layer(x)
            return x
            
        def get_config(self):
            config = super().get_config()
            config.update({
                "units": self.units,
                "activation": self.activation,
                "use_batchnorm": self.use_batchnorm,
            })
            return config
            
    # Register the class directly in the global custom objects dicts
    keras.saving.get_custom_objects()["CustomDenseBlock"] = CustomDenseBlock
    keras.utils.get_custom_objects()["CustomDenseBlock"] = CustomDenseBlock

    HAS_TF = True
except ImportError as e:
    HAS_TF = False
    print(f"Warning: tensorflow/tensorflowjs not imported yet. Error: {e}")

models = [
    "best_rating_prediction_model.keras",
    "nlp_recommendation_model.keras"
]

def convert_model(model_path):
    print("=" * 60)
    print(f"Processing model: {model_path}")
    print("=" * 60)
    
    # 1. Always extract the basic architecture JSON first
    # This is useful as a reference and is extremely fast.
    try:
        config_out = model_path.replace(".keras", "_config.json")
        with zipfile.ZipFile(model_path, 'r') as archive:
            if 'config.json' in archive.namelist():
                config_data = archive.read('config.json')
                config_json = json.loads(config_data)
                with open(config_out, 'w', encoding='utf-8') as f:
                    json.dump(config_json, f, indent=4)
                print(f"[SUCCESS] Extracted Keras config JSON to: {config_out}")
            else:
                print(f"[INFO] No config.json found in zip for {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to extract config.json: {e}")
        
    if not HAS_TF:
        print("[ERROR] TensorFlow/TensorFlowJS is not available. Skipping full TFJS conversion.")
        return

    # 2. Try converting to TFJS
    tfjs_layers_dir = model_path.replace(".keras", "_tfjs_layers")
    tfjs_graph_dir = model_path.replace(".keras", "_tfjs_graph")
    
    # Try converting as a Keras Layers Model first
    print("\nAttempting conversion to Keras Layers Model (tfjs_layers)...")
    try:
        model = keras.models.load_model(model_path)
        tfjs.converters.save_keras_model(model, tfjs_layers_dir)
        print(f"[SUCCESS] Converted to Keras Layers Model in: {tfjs_layers_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to convert directly as Layers Model: {e}")
        print("This is common if the model contains TextVectorization or custom layers.")
        
        # Clean up failed directory if it exists
        if os.path.exists(tfjs_layers_dir):
            shutil.rmtree(tfjs_layers_dir)
            
        # Fallback: Convert via SavedModel to a TFJS Graph Model
        print("\nAttempting fallback: Export to SavedModel and convert to TFJS Graph Model...")
        temp_saved_model_dir = "temp_saved_model"
        try:
            # Re-load model using standalone Keras 3
            model = keras.models.load_model(model_path)
            
            # Export to TensorFlow SavedModel
            print("Saving model to temporary SavedModel directory...")
            model.save(temp_saved_model_dir)
            
            # Call tensorflowjs_converter to convert SavedModel to Graph Model directly in Python
            print("Converting SavedModel to TFJS Graph Model...")
            import tensorflowjs.converters.converter as tfjs_conv
            
            tfjs_conv.convert([
                "--input_format=tf_saved_model",
                temp_saved_model_dir,
                tfjs_graph_dir
            ])
            print(f"[SUCCESS] Converted to TFJS Graph Model in: {tfjs_graph_dir}")
                
        except Exception as ex:
            print(f"[ERROR] Failed SavedModel conversion workflow: {ex}")
        finally:
            # Clean up temp saved model dir
            if os.path.exists(temp_saved_model_dir):
                shutil.rmtree(temp_saved_model_dir)

if __name__ == "__main__":
    for m in models:
        convert_model(m)
    print("\nAll tasks completed!")
