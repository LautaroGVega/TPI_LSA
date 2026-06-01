"""
PASO 2 — Entrenamiento del MLP
--------------------------------
Lee dataset.csv (vectores ya normalizados por traslación y escala)
y entrena un MLP con StandardScaler.

Uso:
  python 2_entrenar.py
"""

import csv
import pickle
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

DATASET_CSV = "dataset.csv"
MODELO_PATH = "modelo_lsa.pkl"


def cargar_dataset():
    X, y = [], []
    with open(DATASET_CSV, "r") as f:
        for row in csv.reader(f):
            if not row:
                continue
            y.append(row[0])
            X.append([float(v) for v in row[1:]])
    return np.array(X), np.array(y)


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

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"[INFO] Entrenamiento: {len(X_train)} muestras")
    print(f"[INFO] Validación:    {len(X_val)} muestras\n")

    # Pipeline: StandardScaler + MLP
    # El scaler es importante incluso con vectores normalizados, porque
    # cada una de las 63 features puede tener distribuciones distintas.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes  = (128, 64, 32),
            activation          = "relu",
            solver              = "adam",
            learning_rate_init  = 0.001,
            max_iter            = 1000,
            random_state        = 42,
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 30,
            verbose             = False,
        )),
    ])

    print("[INFO] Entrenando...")
    pipeline.fit(X_train, y_train)

    iteraciones = pipeline.named_steps["mlp"].n_iter_
    print(f"[INFO] Entrenamiento finalizado en {iteraciones} iteraciones.\n")

    y_pred = pipeline.predict(X_val)
    acc    = (y_pred == y_val).mean()

    print("── Reporte por clase ──")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    if len(clases) > 1:
        print("── Matriz de confusión ──")
        print(f"  Clases: {list(le.classes_)}")
        print(confusion_matrix(y_val, y_pred))

    print(f"\n── Precisión final en validación: {acc * 100:.1f}% ──")

    if acc < 0.80:
        print("\n[CONSEJO] Precisión baja:")
        print("  • Más muestras por clase (300+)")
        print("  • Más variedad: ángulos, distancias, leve rotación de mano")
        print("  • Verificá que NADA tenga muestras")
    elif acc < 0.92:
        print("\n[CONSEJO] Aceptable, podés mejorar con más variedad de capturas.")
    else:
        print("\n[OK] Buen resultado. Probá con 3_probar.py")

    with open(MODELO_PATH, "wb") as f:
        pickle.dump({"modelo": pipeline, "encoder": le}, f)

    print(f"\n[INFO] Modelo guardado en: {MODELO_PATH}")


if __name__ == "__main__":
    main()