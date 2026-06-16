"""
RAPIRO-LSA EC2 Backend — Versión con inferencia real
-------------------------------------------------------
Combina la infraestructura del compañero (DynamoDB, S3, auth, logging)
con la lógica de IA real (MediaPipe tasks API, vector 131, NumPy inference).

Endpoints:
  GET  /health         → estado del servicio
  POST /event          → registrar evento manual
  WS   /ws/frame       → recibir frames del Rapiro por WebSocket
"""

import asyncio
import json
import logging
import math
import os
import time
from collections import defaultdict, deque
from decimal import Decimal
from typing import Any

import boto3
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from urllib.request import Request, urlopen
from urllib.error import URLError
import uvicorn

# ── Imports defensivos ────────────────────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
    CV2_ERROR = None
except Exception as exc:
    cv2 = None
    CV2_AVAILABLE = False
    CV2_ERROR = str(exc)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_ERROR = None
except Exception as exc:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_ERROR = str(exc)

# ── Configuración por variables de entorno ────────────────────────────────
SERVICE_NAME   = "rapiro-lsa-ec2-backend"
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "rapiro-lsa-sessions-sae1")
S3_BUCKET      = os.environ.get("S3_BUCKET", "rapiro-lsa-models-datasets-295552411532-sae1")
S3_EVENTS_PREFIX = os.environ.get("S3_EVENTS_PREFIX", "events/")
AWS_REGION     = os.environ.get("AWS_REGION", "sa-east-1")
API_TOKEN      = os.environ.get("API_TOKEN", "")
WEB_URL        = os.environ.get("WEB_URL", "https://rapiro.onrender.com")

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "models")
HAND_MODEL     = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL     = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
PESOS_FILE     = os.path.join(MODELS_DIR, "modelo_pesos.npz")
SCALER_FILE    = os.path.join(MODELS_DIR, "scaler_params.npz")

TIEMPO_CONFIRMACION = 1.0
CONFIANZA_MINIMA    = 0.80
HISTORIAL_MAX       = 10

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="RAPIRO-LSA EC2 Backend", version="2.0.0")

# ── Estado global ─────────────────────────────────────────────────────────
modelo_data    = {}
hand_detector  = None
face_detector  = None

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION) if DYNAMODB_TABLE else None
table    = dynamodb.Table(DYNAMODB_TABLE) if dynamodb else None
s3       = boto3.client("s3", region_name=AWS_REGION) if S3_BUCKET else None
polly    = boto3.client("polly", region_name=AWS_REGION)


# ── Auth ──────────────────────────────────────────────────────────────────
def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_TOKEN and x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalido")


# ── Modelo NumPy: inferencia sin TensorFlow ───────────────────────────────
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predecir(vector):
    if not modelo_data:
        return "---", 0.0
    p = modelo_data["pesos"]
    x = (np.array(vector, dtype=np.float32) - modelo_data["mean"]) / modelo_data["scale"]
    x = relu(x @ p["W0"] + p["b0"])
    x = relu(x @ p["W1"] + p["b1"])
    x = relu(x @ p["W2"] + p["b2"])
    x = softmax(x @ p["W3"] + p["b3"])
    idx = int(np.argmax(x))
    return modelo_data["clases"][idx], float(x[idx])


# ── MediaPipe: extracción de vector 131 ───────────────────────────────────
def normalizar_mano(landmarks):
    muneca     = landmarks[0]
    base_medio = landmarks[9]
    distancia = math.sqrt(
        (base_medio.x - muneca.x) ** 2 +
        (base_medio.y - muneca.y) ** 2 +
        (base_medio.z - muneca.z) ** 2
    )
    if distancia == 0:
        distancia = 1.0
    valores = []
    for lm in landmarks:
        valores.extend([
            round((lm.x - muneca.x) / distancia, 4),
            round((lm.y - muneca.y) / distancia, 4),
            round((lm.z - muneca.z) / distancia, 4),
        ])
    return valores, distancia, muneca


def extraer_vector(hand_results, nose_x=None, nose_y=None):
    manos = hand_results.hand_landmarks
    mano1_vals, escala1, muneca1 = normalizar_mano(manos[0])

    if len(manos) >= 2:
        mano2_vals, _, muneca2 = normalizar_mano(manos[1])
        dist_manos = [
            round((muneca2.x - muneca1.x) / escala1, 4),
            round((muneca2.y - muneca1.y) / escala1, 4),
            round((muneca2.z - muneca1.z) / escala1, 4),
        ]
    else:
        mano2_vals = [0.0] * 63
        dist_manos = [0.0, 0.0, 0.0]

    if nose_x is not None and nose_y is not None:
        pos_cara = [
            round((muneca1.x - nose_x) / escala1, 4),
            round((muneca1.y - nose_y) / escala1, 4),
        ]
    else:
        pos_cara = [0.0, 0.0]

    return mano1_vals + mano2_vals + dist_manos + pos_cara


def obtener_nariz(face_result):
    if not face_result.detections:
        return None, None
    d = face_result.detections[0]
    if d.keypoints and len(d.keypoints) > 2:
        return d.keypoints[2].x, d.keypoints[2].y
    return None, None


# ── Render: notificación async ────────────────────────────────────────────
def _enviar_web_sync(endpoint, data):
    try:
        body = json.dumps(data).encode("utf-8")
        req = Request(f"{WEB_URL}/api/{endpoint}", data=body,
                     headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=3)
    except (URLError, Exception):
        pass

async def enviar_a_web(endpoint, data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _enviar_web_sync, endpoint, data)


# ── Polly: audio async ───────────────────────────────────────────────────
def _generar_audio_sync(texto):
    try:
        response = polly.synthesize_speech(
            Text=f"La sena detectada fue: {texto}",
            OutputFormat="mp3",
            VoiceId="Lupe",
            LanguageCode="es-US",
        )
        return response["AudioStream"].read()
    except Exception as e:
        logger.error(f"Polly error: {e}")
        return None

async def generar_audio(texto):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generar_audio_sync, texto)


# ── DynamoDB: registro async ─────────────────────────────────────────────
def _guardar_evento_sync(evento):
    if table is None:
        return
    try:
        # Convertir floats a Decimal para DynamoDB
        def decimalize(v):
            if isinstance(v, float):
                return Decimal(str(v))
            if isinstance(v, dict):
                return {k: decimalize(val) for k, val in v.items()}
            if isinstance(v, list):
                return [decimalize(i) for i in v]
            return v
        table.put_item(Item=decimalize(evento))
    except Exception as e:
        logger.error(f"DynamoDB error: {e}")

async def guardar_evento(evento):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _guardar_evento_sync, evento)


# ── Startup: carga única de modelos ───────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global hand_detector, face_detector

    logger.info("Inicializando backend...")

    # Cargar pesos del modelo
    try:
        pesos = dict(np.load(PESOS_FILE))
        scaler = dict(np.load(SCALER_FILE, allow_pickle=True))
        modelo_data["pesos"]  = pesos
        modelo_data["mean"]   = scaler["mean"]
        modelo_data["scale"]  = scaler["scale"]
        modelo_data["clases"] = list(scaler["clases"])
        logger.info(f"Modelo IA cargado. Clases: {modelo_data['clases']}")
    except Exception as e:
        logger.error(f"No se pudo cargar el modelo: {e}")

    # Inicializar MediaPipe (una sola vez)
    if MEDIAPIPE_AVAILABLE:
        try:
            hand_detector = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.7,
                ))
            face_detector = mp_vision.FaceDetector.create_from_options(
                mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=0.5,
                ))
            logger.info("MediaPipe inicializado (global, tasks API).")
        except Exception as e:
            logger.error(f"MediaPipe fallo: {e}")
    else:
        logger.warning(f"MediaPipe no disponible: {MEDIAPIPE_ERROR}")

    logger.info("Backend listo.")


# ── Modelos Pydantic ──────────────────────────────────────────────────────
class EventRequest(BaseModel):
    SessionId: str = Field(default="manual-test-001")
    DetectedSign: str = Field(default="Hola")
    Confidence: float = Field(default=0.9, ge=0, le=1)
    DeviceId: str = Field(default="rapiro-lsa-ec2")
    Mode: str = Field(default="word")
    Source: str = Field(default="EC2 Manual Test")


# ── GET /health ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "cv2_available": CV2_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "modelo_cargado": bool(modelo_data),
        "clases": modelo_data.get("clases", []),
        "dynamodb_table": DYNAMODB_TABLE,
        "s3_bucket": S3_BUCKET,
        "web_url": WEB_URL,
        "errors": {"cv2": CV2_ERROR, "mediapipe": MEDIAPIPE_ERROR},
    }


# ── POST /event (registro manual, compatible con compañero) ──────────────
@app.post("/event")
async def event(payload: EventRequest, _: None = Depends(require_api_key)):
    evento = payload.model_dump()
    evento["Timestamp"] = int(time.time())
    await guardar_evento(evento)
    logger.info(f"Evento manual registrado: {evento['DetectedSign']}")
    return {"message": "Evento registrado", "event": evento}


# ── WebSocket /ws/frame (canal principal con Rapiro) ──────────────────────
@app.websocket("/ws/frame")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"ws-{int(time.time())}"
    logger.info(f"Rapiro conectado | Session: {session_id}")

    # Estado de deletreo por conexión
    texto_formado     = ""
    historial         = []
    letra_candidata   = None
    tiempo_inicio     = 0.0
    letra_confirmada  = False
    cooldown_hasta    = 0.0
    ultimo_web        = 0.0
    ultimo_dynamo     = 0.0

    try:
        while True:
            message = await websocket.receive_bytes()

            if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE or not hand_detector:
                await websocket.send_json({"error": "dependencias no disponibles"})
                continue

            # Decodificar JPEG
            frame_np = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Detectar con modelos globales
            hand_result = hand_detector.detect(mp_img)
            face_result = face_detector.detect(mp_img)

            ahora      = time.time()
            num_manos  = len(hand_result.hand_landmarks)

            prediccion  = "---"
            confianza   = 0.0
            confirmada  = False
            progreso    = 0.0
            audio_bytes = None

            if num_manos > 0:
                nose_x, nose_y = obtener_nariz(face_result)
                vector = extraer_vector(hand_result, nose_x, nose_y)
                prediccion, confianza = predecir(vector)

                # Suavizado
                historial.append(prediccion)
                if len(historial) > HISTORIAL_MAX:
                    historial.pop(0)
                prediccion = max(set(historial), key=historial.count)

                if prediccion == "NADA":
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False

                elif confianza >= CONFIANZA_MINIMA and ahora > cooldown_hasta:
                    if prediccion == letra_candidata:
                        transcurrido = ahora - tiempo_inicio
                        progreso = min(transcurrido / TIEMPO_CONFIRMACION, 1.0)

                        if progreso >= 1.0 and not letra_confirmada:
                            confirmada = True

                            if prediccion == "FINALIZAR" and texto_formado:
                                logger.info(f"FINALIZAR: \"{texto_formado}\"")

                                asyncio.create_task(
                                    enviar_a_web("finalizar", {"texto": texto_formado})
                                )
                                asyncio.create_task(
                                    guardar_evento({
                                        "SessionId": session_id,
                                        "Timestamp": int(time.time()),
                                        "DetectedSign": "FINALIZAR",
                                        "TextoCompleto": texto_formado,
                                        "Confidence": round(confianza, 4),
                                        "Source": "EC2 WebSocket",
                                    })
                                )

                                audio_bytes = await generar_audio(texto_formado)

                                texto_formado    = ""
                                letra_candidata  = None
                                tiempo_inicio    = 0.0
                                letra_confirmada = False
                                cooldown_hasta   = ahora + 2.0
                                historial.clear()

                            elif prediccion == "ESPACIO":
                                texto_formado   += " "
                                letra_confirmada = True
                                cooldown_hasta   = ahora + 0.8
                                logger.info(f"+ ESPACIO -> \"{texto_formado}\"")

                            elif prediccion != "FINALIZAR":
                                texto_formado   += prediccion
                                letra_confirmada = True
                                cooldown_hasta   = ahora + 0.8
                                logger.info(f"+ {prediccion} -> \"{texto_formado}\"")

                        elif letra_confirmada:
                            progreso = 1.0
                    else:
                        letra_candidata  = prediccion
                        tiempo_inicio    = ahora
                        letra_confirmada = False
                else:
                    letra_candidata  = None
                    tiempo_inicio    = 0.0
                    letra_confirmada = False
            else:
                historial.clear()
                letra_candidata  = None
                tiempo_inicio    = 0.0
                letra_confirmada = False

            # Notificar Render cada 0.5s
            if ahora - ultimo_web > 0.5:
                asyncio.create_task(
                    enviar_a_web("letra", {
                        "letra": prediccion,
                        "confianza": confianza,
                        "texto": texto_formado,
                    })
                )
                ultimo_web = ahora

            # Log DynamoDB cada 3s si hay letra confirmada
            if confirmada and ahora - ultimo_dynamo > 3.0:
                asyncio.create_task(
                    guardar_evento({
                        "SessionId": session_id,
                        "Timestamp": int(time.time()),
                        "DetectedSign": prediccion,
                        "TextoAcumulado": texto_formado,
                        "Confidence": round(confianza, 4),
                        "Source": "EC2 WebSocket",
                    })
                )
                ultimo_dynamo = ahora

            # Responder al Pi
            respuesta = {
                "letra": prediccion,
                "confianza": round(confianza, 3),
                "texto": texto_formado,
                "confirmada": confirmada,
                "progreso": round(progreso, 2),
                "manos": num_manos,
            }

            await websocket.send_json(respuesta)

            if audio_bytes:
                await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect:
        logger.info(f"Rapiro desconectado | Session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)