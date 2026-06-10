"""
PASO 2 — Entrenamiento de la Red Neuronal (TensorFlow / Keras)
----------------------------------------------------------------
Construye y entrena un MLP definido capa por capa con Keras.

Genera tres archivos:
  - modelo_lsa.keras      → la red entrenada
  - preproceso.pkl        → StandardScaler + lista de clases
  - curva_entrenamiento.png → gráfico de loss y accuracy por época

Uso:
  python 2_entrenar.py

Requisitos:
  pip install tensorflow scikit-learn matplotlib
"""

import csv
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")          # backend sin ventana (guarda a archivo)
import matplotlib.pyplot as plt

DATASET_CSV   = "dataset.csv"
MODELO_PATH   = "modelo_lsa.keras"
PREPROCESO    = "preproceso.pkl"
CURVA_PNG     = "curva_entrenamiento.png"

# Reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)


def cargar_dataset():
    X, y = [], []
    with open(DATASET_CSV, "r") as f:
        for row in csv.reader(f):
            if not row:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float32), np.array(y)


def construir_modelo(n_entradas: int, n_clases: int) -> tf.keras.Model:
    """
    Define la arquitectura del MLP capa por capa.
    Esta es la red neuronal desarrollada por el equipo.
    """
    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_entradas,)),

        # Capa oculta 1
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),       # regularización: apaga 30% de neuronas

        # Capa oculta 2
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Capa oculta 3
        tf.keras.layers.Dense(32, activation="relu"),

        # Capa de salida: una neurona por clase, softmax = distribución de probabilidad
        tf.keras.layers.Dense(n_clases, activation="softmax"),
    ])

    modelo.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return modelo


def graficar_curvas(history):
    """Guarda la curva de loss y accuracy por época."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Entrenamiento")
    ax1.plot(history.history["val_loss"], label="Validación")
    ax1.set_title("Pérdida (loss) por época")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Entrenamiento")
    ax2.plot(history.history["val_accuracy"], label="Validación")
    ax2.set_title("Precisión (accuracy) por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CURVA_PNG, dpi=120)
    print(f"[INFO] Gráfico de entrenamiento guardado en: {CURVA_PNG}")


def main():
    if not os.path.exists(DATASET_CSV):
        print(f"[ERROR] No se encontró {DATASET_CSV}. Ejecutá primero 1_capturar.py")
        return

    print("[INFO] Cargando dataset...")
    X, y = cargar_dataset()

    clases, conteos = np.unique(y, return_counts=True)
    print(f"\n── Distribución del dataset ──")
    for c, n in zip(clases, conteos):
        barra = "█" * (n // 10)
        print(f"  {c:>6}: {n:>4} muestras  {barra}")
    print(f"  {'TOTAL':>6}: {len(X):>4} muestras\n")

    if len(clases) < 2:
        print("[ADVERTENCIA] Solo hay una clase en el dataset.")
        print("  Capturá también la clase NADA (tecla 0 en 1_capturar.py)\n")
        return

    # Codificar etiquetas (texto → número)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Normalización estadística de las features (rol auxiliar de scikit-learn)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split 80/20
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"[INFO] Entrenamiento: {len(X_train)} muestras")
    print(f"[INFO] Validación:    {len(X_val)} muestras\n")

    # Construir la red
    modelo = construir_modelo(n_entradas=X.shape[1], n_clases=len(clases))

    print("── Arquitectura de la red ──")
    modelo.summary()
    print()

    # Early stopping: detiene si la validación deja de mejorar
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
    )

    print("[INFO] Entrenando...")
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=32,
        callbacks=[early],
        verbose=2,
    )

    # Evaluación final
    print("\n[INFO] Evaluando en conjunto de validación...")
    y_pred = np.argmax(modelo.predict(X_val, verbose=0), axis=1)

    print("\n── Reporte por clase ──")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    print("── Matriz de confusión ──")
    print(f"  Clases: {list(le.classes_)}")
    print(confusion_matrix(y_val, y_pred))

    loss, acc = modelo.evaluate(X_val, y_val, verbose=0)
    print(f"\n── Precisión final en validación: {acc * 100:.1f}% ──")

    if acc < 0.80:
        print("\n[CONSEJO] Precisión baja:")
        print("  • Más muestras por clase (300+)")
        print("  • Más variedad: ángulos, distancias, leve rotación")
        print("  • Verificá que NADA tenga muestras")
    elif acc < 0.92:
        print("\n[CONSEJO] Aceptable, mejorable con más variedad de capturas.")
    else:
        print("\n[OK] Buen resultado. Probá con 3_probar.py")

    # Guardar todo
    modelo.save(MODELO_PATH)
    with open(PREPROCESO, "wb") as f:
        pickle.dump({"scaler": scaler, "clases": le.classes_}, f)

    graficar_curvas(history)

    print(f"\n[INFO] Modelo guardado en:       {MODELO_PATH}")
    print(f"[INFO] Preprocesamiento guardado: {PREPROCESO}")


if __name__ == "__main__":
    main()