"""
COLAB — Exportar pesos del modelo para AWS Lambda
----------------------------------------------------
Extrae los pesos de la red Keras como arrays NumPy.
Lambda puede hacer inferencia solo con NumPy, sin TensorFlow.

Agregar esta celda al final del notebook de entrenamiento en Colab.
"""

import numpy as np
import pickle
import tensorflow as tf

# Cargar modelo (si no está en memoria)
# modelo = tf.keras.models.load_model('modelo_lsa.keras')

# Extraer pesos de cada capa Dense (ignora Dropout, no tiene pesos)
pesos = {}
for i, layer in enumerate(modelo.layers):
    w = layer.get_weights()
    if w:  # solo las capas Dense tienen pesos
        pesos[f"W{len(pesos)//2}"] = w[0]   # matriz de pesos
        pesos[f"b{len(pesos)//2}"] = w[1]    # bias

# Guardar pesos
np.savez("modelo_pesos.npz", **pesos)
print(f"✅ Pesos guardados: {list(pesos.keys())}")
for k, v in pesos.items():
    print(f"   {k}: {v.shape}")

# Guardar parámetros del scaler
with open("preproceso.pkl", "rb") as f:
    datos = pickle.load(f)

scaler = datos["scaler"]
clases = datos["clases"]

np.savez("scaler_params.npz",
         mean=scaler.mean_,
         scale=scaler.scale_,
         clases=clases)
print(f"\n✅ Scaler guardado: mean{scaler.mean_.shape}, scale{scaler.scale_.shape}")
print(f"   Clases: {list(clases)}")

# Descargar
from google.colab import files
files.download("modelo_pesos.npz")
files.download("scaler_params.npz")
print("\n📥 Descarga iniciada")