"""
RAPIRO-LSA EC2 Backend v2.1 — Con video en vivo
--------------------------------------------------
Endpoints:
  GET  /health      - Estado del servicio
  GET  /live        - Pagina con video en vivo + deteccion
  GET  /video       - Stream MJPEG del video procesado
  POST /event       - Registro manual de eventos
  WS   /ws/frame    - Canal WebSocket con Rapiro
"""

import asyncio
import json
import logging
import math
import os
import time
from decimal import Decimal
from typing import Any

import boto3
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from urllib.request import Request, urlopen
from urllib.error import URLError
import uvicorn

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

SERVICE_NAME   = "rapiro-lsa-ec2-backend"
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "rapiro-lsa-sessions-sae1")
S3_BUCKET      = os.environ.get("S3_BUCKET", "rapiro-lsa-models-datasets-295552411532-sae1")
S3_EVENTS_PREFIX = os.environ.get("S3_EVENTS_PREFIX", "events/")
AWS_REGION     = os.environ.get("AWS_REGION", "sa-east-1")
API_TOKEN      = os.environ.get("API_TOKEN", "")
WEB_URL        = os.environ.get("WEB_URL", "https://rapiro.onrender.com")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
PESOS_FILE = os.path.join(MODELS_DIR, "modelo_pesos.npz")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler_params.npz")

TIEMPO_CONFIRMACION = 1.0
CONFIANZA_MINIMA    = 0.80
HISTORIAL_MAX       = 10

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="RAPIRO-LSA EC2 Backend", version="2.1.0")

modelo_data    = {}
hand_detector  = None
face_detector  = None

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION) if DYNAMODB_TABLE else None
table    = dynamodb.Table(DYNAMODB_TABLE) if dynamodb else None
s3       = boto3.client("s3", region_name=AWS_REGION) if S3_BUCKET else None
polly    = boto3.client("polly", region_name=AWS_REGION)

# Frame procesado para el stream de video en vivo
latest_frame_jpeg = None
live_estado = {"letra": "---", "confianza": 0.0, "texto": "", "manos": 0}


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_TOKEN and x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalido")


def relu(x): return np.maximum(0, x)
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def predecir(vector):
    if not modelo_data: return "---", 0.0
    p = modelo_data["pesos"]
    x = (np.array(vector, dtype=np.float32) - modelo_data["mean"]) / modelo_data["scale"]
    x = relu(x @ p["W0"] + p["b0"])
    x = relu(x @ p["W1"] + p["b1"])
    x = relu(x @ p["W2"] + p["b2"])
    x = softmax(x @ p["W3"] + p["b3"])
    idx = int(np.argmax(x))
    return modelo_data["clases"][idx], float(x[idx])


def normalizar_mano(landmarks):
    muneca = landmarks[0]; base_medio = landmarks[9]
    distancia = math.sqrt((base_medio.x-muneca.x)**2+(base_medio.y-muneca.y)**2+(base_medio.z-muneca.z)**2)
    if distancia == 0: distancia = 1.0
    valores = []
    for lm in landmarks:
        valores.extend([round((lm.x-muneca.x)/distancia,4), round((lm.y-muneca.y)/distancia,4), round((lm.z-muneca.z)/distancia,4)])
    return valores, distancia, muneca

def extraer_vector(hand_results, nose_x=None, nose_y=None):
    manos = hand_results.hand_landmarks
    mano1_vals, escala1, muneca1 = normalizar_mano(manos[0])
    if len(manos) >= 2:
        mano2_vals, _, muneca2 = normalizar_mano(manos[1])
        dist_manos = [round((muneca2.x-muneca1.x)/escala1,4), round((muneca2.y-muneca1.y)/escala1,4), round((muneca2.z-muneca1.z)/escala1,4)]
    else:
        mano2_vals = [0.0]*63; dist_manos = [0.0,0.0,0.0]
    pos_cara = [round((muneca1.x-nose_x)/escala1,4), round((muneca1.y-nose_y)/escala1,4)] if nose_x and nose_y else [0.0,0.0]
    return mano1_vals + mano2_vals + dist_manos + pos_cara

def obtener_nariz(face_result):
    if not face_result.detections: return None, None
    d = face_result.detections[0]
    if d.keypoints and len(d.keypoints) > 2: return d.keypoints[2].x, d.keypoints[2].y
    return None, None


# ── Dibujar landmarks y HUD sobre el frame ────────────────────────────────
CONEXIONES = None
if MEDIAPIPE_AVAILABLE:
    CONEXIONES = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def dibujar_frame(frame, hand_result, face_result, prediccion, confianza, texto, progreso):
    """Dibuja landmarks, letra y texto sobre el frame."""
    if not CV2_AVAILABLE: return frame
    h, w = frame.shape[:2]
    colores_mano = [(0, 220, 0), (220, 160, 0)]

    # Dibujar manos
    for idx, landmarks in enumerate(hand_result.hand_landmarks):
        color = colores_mano[min(idx, 1)]
        puntos = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
        if CONEXIONES:
            for conn in CONEXIONES:
                cv2.line(frame, puntos[conn.start], puntos[conn.end], (255,255,255), 2)
        for px, py in puntos:
            cv2.circle(frame, (px, py), 5, color, -1)

    # Dibujar nariz
    nose_x, nose_y = obtener_nariz(face_result)
    if nose_x is not None:
        cv2.circle(frame, (int(nose_x*w), int(nose_y*h)), 8, (220, 0, 220), -1)

    # HUD superior
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 110), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Letra grande
    color_pred = (0,220,0) if confianza >= 0.85 else (0,180,220) if confianza > 0 else (160,160,160)
    cv2.putText(frame, str(prediccion), (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color_pred, 5, cv2.LINE_AA)

    # Confianza
    if confianza > 0:
        cv2.putText(frame, f"{confianza*100:.0f}%", (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_pred, 2, cv2.LINE_AA)

    # Barra de progreso
    cv2.rectangle(frame, (140,80), (340,94), (60,60,60), -1)
    if progreso > 0:
        cv2.rectangle(frame, (140,80), (140+int(progreso*200),94), color_pred, -1)

    # CLOUD badge
    cv2.putText(frame, "CLOUD", (w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,0,220), 2, cv2.LINE_AA)

    # Texto acumulado abajo
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h-60), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Texto:", (10, h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)
    mostrar = texto if texto else "|"
    if len(mostrar) > 40: mostrar = "..." + mostrar[-37:]
    cv2.putText(frame, mostrar, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

    return frame


# ── Render y Polly async ──────────────────────────────────────────────────
def _enviar_web_sync(endpoint, data):
    try:
        body = json.dumps(data).encode("utf-8")
        req = Request(f"{WEB_URL}/api/{endpoint}", data=body,
                     headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=3)
    except Exception: pass

async def enviar_a_web(endpoint, data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _enviar_web_sync, endpoint, data)

def _generar_audio_sync(texto):
    try:
        response = polly.synthesize_speech(Text=f"La sena detectada fue: {texto}",
                                           OutputFormat="mp3", VoiceId="Lupe", LanguageCode="es-US")
        return response["AudioStream"].read()
    except Exception as e:
        logger.error(f"Polly error: {e}"); return None

async def generar_audio(texto):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _generar_audio_sync, texto)

def _guardar_evento_sync(evento):
    if not table: return
    try:
        def decimalize(v):
            if isinstance(v, float): return Decimal(str(v))
            if isinstance(v, dict): return {k: decimalize(val) for k, val in v.items()}
            if isinstance(v, list): return [decimalize(i) for i in v]
            return v
        table.put_item(Item=decimalize(evento))
    except Exception as e: logger.error(f"DynamoDB error: {e}")

async def guardar_evento(evento):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _guardar_evento_sync, evento)


# ── Startup ───────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global hand_detector, face_detector
    logger.info("Inicializando backend v2.1...")
    try:
        pesos = dict(np.load(PESOS_FILE))
        scaler = dict(np.load(SCALER_FILE, allow_pickle=True))
        modelo_data["pesos"] = pesos; modelo_data["mean"] = scaler["mean"]
        modelo_data["scale"] = scaler["scale"]; modelo_data["clases"] = list(scaler["clases"])
        logger.info(f"Modelo cargado. Clases: {modelo_data['clases']}")
    except Exception as e: logger.error(f"Modelo error: {e}")

    if MEDIAPIPE_AVAILABLE:
        try:
            hand_detector = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
                    running_mode=mp_vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.7))
            face_detector = mp_vision.FaceDetector.create_from_options(
                mp_vision.FaceDetectorOptions(base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
                    running_mode=mp_vision.RunningMode.IMAGE, min_detection_confidence=0.5))
            logger.info("MediaPipe inicializado.")
        except Exception as e: logger.error(f"MediaPipe error: {e}")
    logger.info("Backend v2.1 listo.")


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
    return {"status": "ok", "service": SERVICE_NAME, "version": "2.1",
            "cv2_available": CV2_AVAILABLE, "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "modelo_cargado": bool(modelo_data), "clases": modelo_data.get("clases", []),
            "errors": {"cv2": CV2_ERROR, "mediapipe": MEDIAPIPE_ERROR}}


# ── POST /event ───────────────────────────────────────────────────────────
@app.post("/event")
async def event(payload: EventRequest, _: None = Depends(require_api_key)):
    evento = payload.model_dump(); evento["Timestamp"] = int(time.time())
    await guardar_evento(evento)
    return {"message": "Evento registrado", "event": evento}


# ── GET /video — Stream MJPEG en vivo ─────────────────────────────────────
def generar_mjpeg():
    while True:
        if latest_frame_jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame_jpeg + b"\r\n")
        time.sleep(0.02)

@app.get("/video")
def video_feed():
    return StreamingResponse(generar_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── GET /live — Pagina con video en vivo + deteccion ──────────────────────
LIVE_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAPIRO LSA - Deteccion en vivo</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e4e4e7;
       display:flex;flex-direction:column;align-items:center;padding:20px;min-height:100vh}
  h1{font-size:1.2rem;color:#71717a;margin-bottom:20px;letter-spacing:2px;text-transform:uppercase}
  .video-container{position:relative;border-radius:16px;overflow:hidden;border:2px solid #27272a;
                   max-width:700px;width:100%}
  .video-container img{width:100%;display:block}
  .info-bar{display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;justify-content:center}
  .info-card{background:#18181b;border:1px solid #27272a;border-radius:12px;padding:16px 24px;text-align:center}
  .info-label{font-size:0.7rem;color:#52525b;text-transform:uppercase;letter-spacing:1px}
  .info-value{font-size:2rem;font-weight:700;margin-top:4px}
  .info-value.letra{color:#22c55e;font-size:3.5rem}
  .info-value.texto{color:#e4e4e7;font-size:1.4rem;max-width:400px;word-wrap:break-word}
  .info-value.conf{color:#a855f7}
  .cloud-badge{background:#1e1b4b;color:#a855f7;padding:4px 12px;border-radius:20px;
               font-size:0.7rem;font-weight:600;letter-spacing:1px;margin-top:16px}
  .offline{color:#ef4444;font-size:1.2rem;margin-top:20px}
</style>
</head>
<body>
  <h1>RAPIRO LSA - Deteccion en vivo (Cloud)</h1>

  <div class="video-container">
    <img id="videoFeed" src="/video" alt="Video en vivo"
         onerror="this.style.display='none';document.getElementById('offlineMsg').style.display='block'">
    <div id="offlineMsg" class="offline" style="display:none;padding:40px;text-align:center">
      Esperando conexion del Rapiro...
    </div>
  </div>

  <div class="info-bar">
    <div class="info-card">
      <div class="info-label">Letra detectada</div>
      <div class="info-value letra" id="letraVivo">---</div>
    </div>
    <div class="info-card">
      <div class="info-label">Confianza</div>
      <div class="info-value conf" id="confVivo">0%</div>
    </div>
    <div class="info-card">
      <div class="info-label">Manos</div>
      <div class="info-value" id="manosVivo">0</div>
    </div>
    <div class="info-card">
      <div class="info-label">Texto acumulado</div>
      <div class="info-value texto" id="textoVivo">|</div>
    </div>
  </div>

  <div class="cloud-badge">PROCESAMIENTO EN AWS CLOUD - SAO PAULO</div>

<script>
async function actualizarEstado() {
  try {
    const res = await fetch('/api/live_estado');
    const data = await res.json();
    document.getElementById('letraVivo').textContent = data.letra || '---';
    document.getElementById('confVivo').textContent = (data.confianza*100).toFixed(0) + '%';
    document.getElementById('manosVivo').textContent = data.manos || '0';
    document.getElementById('textoVivo').textContent = data.texto || '|';
  } catch(e) {}
}
setInterval(actualizarEstado, 300);
</script>
</body>
</html>"""

@app.get("/live", response_class=HTMLResponse)
def live_page():
    return LIVE_HTML

@app.get("/api/live_estado")
def get_live_estado():
    return live_estado


# ── WebSocket /ws/frame ───────────────────────────────────────────────────
@app.websocket("/ws/frame")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame_jpeg

    await websocket.accept()
    session_id = f"ws-{int(time.time())}"
    logger.info(f"Rapiro conectado | Session: {session_id}")

    texto_formado = ""; historial = []; letra_candidata = None
    tiempo_inicio = 0.0; letra_confirmada = False; cooldown_hasta = 0.0
    ultimo_web = 0.0; ultimo_dynamo = 0.0

    try:
        while True:
            message = await websocket.receive_bytes()
            if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE or not hand_detector:
                await websocket.send_json({"error": "dependencias no disponibles"}); continue

            frame_np = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            if frame is None: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            hand_result = hand_detector.detect(mp_img)
            face_result = face_detector.detect(mp_img)

            ahora = time.time()
            num_manos = len(hand_result.hand_landmarks)
            prediccion = "---"; confianza = 0.0; confirmada = False; progreso = 0.0; audio_bytes = None

            if num_manos > 0:
                nose_x, nose_y = obtener_nariz(face_result)
                vector = extraer_vector(hand_result, nose_x, nose_y)
                prediccion, confianza = predecir(vector)

                historial.append(prediccion)
                if len(historial) > HISTORIAL_MAX: historial.pop(0)
                prediccion = max(set(historial), key=historial.count)

                if prediccion == "NADA":
                    letra_candidata = None; tiempo_inicio = 0.0; letra_confirmada = False
                elif confianza >= CONFIANZA_MINIMA and ahora > cooldown_hasta:
                    if prediccion == letra_candidata:
                        progreso = min((ahora - tiempo_inicio) / TIEMPO_CONFIRMACION, 1.0)
                        if progreso >= 1.0 and not letra_confirmada:
                            confirmada = True
                            if prediccion == "FINALIZAR" and texto_formado:
                                logger.info(f"FINALIZAR: \"{texto_formado}\"")
                                asyncio.create_task(enviar_a_web("finalizar", {"texto": texto_formado}))
                                asyncio.create_task(guardar_evento({"SessionId": session_id, "Timestamp": int(time.time()),
                                    "DetectedSign": "FINALIZAR", "TextoCompleto": texto_formado,
                                    "Confidence": round(confianza,4), "Source": "EC2 WebSocket"}))
                                audio_bytes = await generar_audio(texto_formado)
                                texto_formado = ""; letra_candidata = None; tiempo_inicio = 0.0
                                letra_confirmada = False; cooldown_hasta = ahora + 2.0; historial.clear()
                            elif prediccion == "ESPACIO":
                                texto_formado += " "; letra_confirmada = True; cooldown_hasta = ahora + 0.8
                                logger.info(f"+ ESPACIO -> \"{texto_formado}\"")
                            elif prediccion != "FINALIZAR":
                                texto_formado += prediccion; letra_confirmada = True; cooldown_hasta = ahora + 0.8
                                logger.info(f"+ {prediccion} -> \"{texto_formado}\"")
                        elif letra_confirmada: progreso = 1.0
                    else:
                        letra_candidata = prediccion; tiempo_inicio = ahora; letra_confirmada = False
                else:
                    letra_candidata = None; tiempo_inicio = 0.0; letra_confirmada = False
            else:
                historial.clear(); letra_candidata = None; tiempo_inicio = 0.0; letra_confirmada = False

            # Dibujar landmarks y HUD sobre el frame
            frame_dibujado = dibujar_frame(frame, hand_result, face_result, prediccion, confianza, texto_formado, progreso)
            _, jpeg_buf = cv2.imencode(".jpg", frame_dibujado, [cv2.IMWRITE_JPEG_QUALITY, 70])
            latest_frame_jpeg = jpeg_buf.tobytes()

            # Actualizar estado para /live
            live_estado["letra"] = prediccion
            live_estado["confianza"] = round(confianza, 3)
            live_estado["texto"] = texto_formado
            live_estado["manos"] = num_manos

            # Render cada 0.5s
            if ahora - ultimo_web > 0.5:
                asyncio.create_task(enviar_a_web("letra", {"letra": prediccion, "confianza": confianza, "texto": texto_formado}))
                ultimo_web = ahora

            if confirmada and ahora - ultimo_dynamo > 3.0:
                asyncio.create_task(guardar_evento({"SessionId": session_id, "Timestamp": int(time.time()),
                    "DetectedSign": prediccion, "TextoAcumulado": texto_formado,
                    "Confidence": round(confianza,4), "Source": "EC2 WebSocket"}))
                ultimo_dynamo = ahora

            respuesta = {"letra": prediccion, "confianza": round(confianza,3), "texto": texto_formado,
                         "confirmada": confirmada, "progreso": round(progreso,2), "manos": num_manos}
            await websocket.send_json(respuesta)
            if audio_bytes: await websocket.send_bytes(audio_bytes)

    except WebSocketDisconnect: logger.info(f"Rapiro desconectado | Session: {session_id}")
    except Exception as e: logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)