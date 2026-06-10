"""
Deletreo en tiempo real — Forma oraciones con LSA
----------------------------------------------------
Mantené una seña estable por 2.5 segundos para confirmar la letra.
Las letras confirmadas se acumulan formando palabras y oraciones.

Archivos necesarios:
  - hand_landmarker.task
  - modelo_lsa.keras
  - preproceso.pkl

Uso:
  py -3.12 deletrear.py

Controles (teclado):
  BACKSPACE  → borrar última letra
  ESC        → borrar todo el texto
  Q          → salir
"""

import math
import pickle
import sys
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

HAND_MODEL  = "hand_landmarker.task"
MODELO_PATH = "modelo_lsa.keras"
PREPROCESO  = "preproceso.pkl"
CONEXIONES  = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# ── Configuración ─────────────────────────────────────────────────────────
TIEMPO_CONFIRMACION = 0.5    # segundos que hay que mantener la seña
CONFIANZA_MINIMA    = 0.80   # confianza mínima para considerar la predicción
HISTORIAL_MAX       = 10     # frames para suavizar la predicción


# ═══════════════════════════════════════════════════════════════════════════
#  FUNCIÓN CRÍTICA — idéntica a capturar.py y probar.py
# ═══════════════════════════════════════════════════════════════════════════
def extraer_vector(landmarks) -> list[float]:
    muneca     = landmarks[0]
    base_medio = landmarks[9]

    distancia = math.sqrt(
        (base_medio.x - muneca.x) ** 2 +
        (base_medio.y - muneca.y) ** 2 +
        (base_medio.z - muneca.z) ** 2
    )
    if distancia == 0:
        distancia = 1.0

    vector = []
    for lm in landmarks:
        rel_x = (lm.x - muneca.x) / distancia
        rel_y = (lm.y - muneca.y) / distancia
        rel_z = (lm.z - muneca.z) / distancia
        vector.extend([round(rel_x, 4), round(rel_y, 4), round(rel_z, 4)])
    return vector
# ═══════════════════════════════════════════════════════════════════════════


def crear_detector():
    opciones = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
    )
    return mp_vision.HandLandmarker.create_from_options(opciones)


def dibujar_mano(frame, landmarks):
    h, w = frame.shape[:2]
    puntos = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in CONEXIONES:
        cv2.line(frame, puntos[conn.start], puntos[conn.end], (255, 255, 255), 2)
    for px, py in puntos:
        cv2.circle(frame, (px, py), 5, (0, 220, 0), -1)


def dibujar_barra_progreso(frame, x, y, ancho, alto, progreso, color):
    """Dibuja una barra de progreso con fondo gris y relleno del color dado."""
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (60, 60, 60), -1)
    relleno = int(progreso * ancho)
    if relleno > 0:
        cv2.rectangle(frame, (x, y), (x + relleno, y + alto), color, -1)


def dibujar_texto_formado(frame, texto):
    """Dibuja el texto acumulado en la parte inferior del frame."""
    h, w = frame.shape[:2]

    # Fondo para el área de texto
    area_alto = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - area_alto), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Etiqueta
    cv2.putText(frame, "Texto:", (10, h - area_alto + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    # Texto formado (puede ser largo, mostrar últimos caracteres que entren)
    texto_mostrar = texto if texto else "|"
    max_chars = (w - 20) // 18   # aproximado según tamaño de fuente
    if len(texto_mostrar) > max_chars:
        texto_mostrar = "..." + texto_mostrar[-(max_chars - 3):]

    cv2.putText(frame, texto_mostrar, (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    # ── Cargar modelo ─────────────────────────────────────────────────────
    try:
        modelo = tf.keras.models.load_model(MODELO_PATH)
        print(f"[INFO] Modelo cargado: {MODELO_PATH}")
    except Exception as e:
        print(f"[ERROR] No se pudo cargar {MODELO_PATH}: {e}")
        sys.exit(1)

    try:
        with open(PREPROCESO, "rb") as f:
            datos = pickle.load(f)
        scaler = datos["scaler"]
        clases = datos["clases"]
        print(f"[INFO] Clases: {list(clases)}")
    except FileNotFoundError:
        print(f"[ERROR] No se encontró {PREPROCESO}")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se encontró cámara.")
        sys.exit(1)

    print("[INFO] Cámara iniciada.")
    print("[INFO] Mantené una seña estable 2.5s para confirmar la letra.")
    print("[INFO] BACKSPACE = borrar | ESC = limpiar | Q = salir")

    # ── Estado del deletreo ───────────────────────────────────────────────
    texto_formado     = ""        # oracion que se va formando
    historial         = []        # para suavizar predicciones
    letra_candidata   = None      # letra que se está manteniendo
    tiempo_inicio     = 0.0       # cuándo empezó a mantenerse
    letra_confirmada  = False     # evita repetir la misma letra sin pausa
    cooldown_hasta    = 0.0       # breve pausa después de confirmar

    with crear_detector() as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            resultado = detector.detect(mp_img)

            ahora       = time.time()
            prediccion  = None
            confianza   = 0.0
            color_pred  = (160, 160, 160)
            progreso    = 0.0

            if resultado.hand_landmarks:
                landmarks = resultado.hand_landmarks[0]
                dibujar_mano(frame, landmarks)
                vector = extraer_vector(landmarks)

                vector_scaled = scaler.transform([vector])
                proba = modelo.predict(vector_scaled, verbose=0)[0]
                idx   = np.argmax(proba)
                conf  = proba[idx]

                # Suavizado
                historial.append(idx)
                if len(historial) > HISTORIAL_MAX:
                    historial.pop(0)
                idx_suave = max(set(historial), key=historial.count)

                prediccion = clases[idx_suave]
                confianza  = conf

                # Ignorar NADA como letra (solo resetea el estado)
                if prediccion == "NADA":
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False
                    color_pred = (160, 160, 160)

                elif confianza >= CONFIANZA_MINIMA and ahora > cooldown_hasta:
                    color_pred = (0, 180, 220)   # amarillo: detectando

                    if prediccion == letra_candidata:
                        # Misma letra, acumular tiempo
                        transcurrido = ahora - tiempo_inicio
                        progreso = min(transcurrido / TIEMPO_CONFIRMACION, 1.0)

                        if progreso >= 1.0 and not letra_confirmada:
                            # ¡CONFIRMADA!
                            if prediccion == "ESPACIO":
                                texto_formado += " "
                            else:
                                texto_formado += prediccion

                            letra_confirmada = True
                            cooldown_hasta   = ahora + 0.8   # pausa breve
                            color_pred = (0, 220, 0)         # verde: confirmado
                            print(f"  ✓ Letra confirmada: {prediccion}  →  \"{texto_formado}\"")

                        elif letra_confirmada:
                            color_pred = (0, 220, 0)   # ya confirmada, verde fijo
                            progreso = 1.0
                    else:
                        # Cambió la letra, reiniciar
                        letra_candidata  = prediccion
                        tiempo_inicio    = ahora
                        letra_confirmada = False
                        progreso = 0.0
                else:
                    # Confianza muy baja
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False
            else:
                # Sin mano: resetear todo
                historial.clear()
                letra_candidata  = None
                tiempo_inicio    = 0.0
                letra_confirmada = False

            # ── HUD superior ──────────────────────────────────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # Letra detectada (grande)
            letra_mostrar = prediccion if prediccion else "—"
            cv2.putText(frame, letra_mostrar, (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, color_pred, 5, cv2.LINE_AA)

            # Confianza
            if confianza > 0 and prediccion:
                cv2.putText(frame, f"{confianza * 100:.0f}%", (140, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_pred, 2, cv2.LINE_AA)

            # Estado
            if letra_confirmada:
                estado = "CONFIRMADO"
                color_estado = (0, 220, 0)
            elif letra_candidata and prediccion != "NADA":
                estado = "Mantene..."
                color_estado = (0, 180, 220)
            else:
                estado = "Esperando seña"
                color_estado = (160, 160, 160)

            cv2.putText(frame, estado, (140, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2, cv2.LINE_AA)

            # Barra de progreso de confirmación
            dibujar_barra_progreso(frame, 140, 80, 200, 14, progreso, color_pred)

            # Tiempo restante
            if progreso > 0 and progreso < 1.0:
                restante = TIEMPO_CONFIRMACION * (1 - progreso)
                cv2.putText(frame, f"{restante:.1f}s", (350, 93),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Ayuda
            cv2.putText(frame, "BACKSPACE=borrar  ESC=limpiar  Q=salir",
                        (10, frame.shape[0] - 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

            # ── Texto formado (parte inferior) ────────────────────────────
            dibujar_texto_formado(frame, texto_formado)

            cv2.imshow("LSA — Deletreo en tiempo real", frame)

            # ── Teclado ───────────────────────────────────────────────────
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == 8:      # BACKSPACE
                if texto_formado:
                    texto_formado = texto_formado[:-1]
                    print(f"  ← Borrado  →  \"{texto_formado}\"")
            elif tecla == 27:     # ESC
                texto_formado = ""
                print("  ✕ Texto limpiado")

    cap.release()
    cv2.destroyAllWindows()

    if texto_formado:
        print(f"\n── Texto final ──")
        print(f"  \"{texto_formado}\"")


if __name__ == "__main__":
    main()