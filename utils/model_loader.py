import streamlit as st
import joblib
import tensorflow as tf
import zipfile
import json
import os
import tempfile
import shutil
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_keras_model_patched(model_path: str) -> tf.keras.Model:
    """Loads a Keras model, patching serialization differences if necessary."""
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        logger.warning(f"Direct Keras model loading failed: {e}. Attempting custom patch loader...")
        
    # Create a temporary directory to unpack and modify model config
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        config_path = os.path.join(temp_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Recursively remove 'quantization_config' keys that cause deserialization errors
            def clean_config(cfg: Any) -> None:
                if isinstance(cfg, dict):
                    if 'quantization_config' in cfg:
                        del cfg['quantization_config']
                    for k, v in list(cfg.items()):
                        clean_config(v)
                elif isinstance(cfg, list):
                    for item in cfg:
                        clean_config(item)
                        
            clean_config(config)
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
                
        # Zip it back to a temp file
        temp_model_path = os.path.join(temp_dir, 'patched_model.keras')
        with zipfile.ZipFile(temp_model_path, 'w') as zip_ref:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file != 'patched_model.keras':
                        abs_path = os.path.join(root, file)
                        rel_path = os.path.relpath(abs_path, temp_dir)
                        zip_ref.write(abs_path, rel_path)
                        
        model = tf.keras.models.load_model(temp_model_path)
        logger.info("Successfully loaded patched Keras model.")
        return model
    except Exception as patch_err:
        logger.error(f"Patched model loading failed: {patch_err}")
        raise patch_err
    finally:
        shutil.rmtree(temp_dir)

@st.cache_resource
def load_ml_artifacts(artifacts_path: str = "model/mp1_artifacts.pkl") -> Dict[str, Any]:
    """Loads and caches Scikit-learn and XGBoost model artifacts."""
    logger.info(f"Loading ML artifacts from {artifacts_path}...")
    try:
        artifacts = joblib.load(artifacts_path)
        return artifacts
    except Exception as e:
        logger.error(f"Error loading ML artifacts: {e}")
        raise e

@st.cache_resource
def load_ann_model(model_path: str = "model/best_model.keras") -> tf.keras.Model:
    """Loads and caches the Keras Artificial Neural Network model."""
    logger.info(f"Loading ANN model from {model_path}...")
    try:
        return load_keras_model_patched(model_path)
    except Exception as e:
        logger.error(f"Error loading Keras ANN model: {e}")
        raise e
