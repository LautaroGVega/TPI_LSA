"""
RAPIRO-LSA EC2 Backend v2.3 — Metricas + Aprendizaje Adaptativo (corregido)
------------------------------------------------------------------------------
Fixes v2.3:
  - Umbral adaptativo: baja cuando el usuario tiene dificultad (ayuda)
                        sube cuando el usuario es experto (evita falsos positivos)
  - Cooldown de 2s para escrituras a S3 (evita spam de incertidumbre)
  - Variable temporal texto_a_hablar para Polly

Endpoints:
  GET  /health      GET  /metrics     GET  /live
  GET  /video       POST /event       WS   /ws/frame
"""

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
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
    cv2 = None; CV2_AVAILABLE = False; CV2_ERROR = str(exc)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_ERROR = None
except Exception as exc:
    mp = None; MEDIAPIPE_AVAILABLE = False; MEDIAPIPE_ERROR = str(exc)

SERVICE_NAME   = "rapiro-lsa-ec2-backend"
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "rapiro-lsa-sessions-sae1")
S3_BUCKET      = os.environ.get("S3_BUCKET", "rapiro-lsa-models-datasets-295552411532-sae1")
S3_EVENTS_PREFIX = os.environ.get("S3_EVENTS_PREFIX", "events/")
AWS_REGION     = os.environ.get("AWS_REGION", "sa-east-1")
API_TOKEN      = os.environ.get("API_TOKEN", "")
WEB_URL        = os.environ.get("WEB_URL", "https://rapiro.onrender.com")

MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAND_MODEL  = os.path.join(MODELS_DIR, "hand_landmarker.task")
FACE_MODEL  = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
PESOS_FILE  = os.path.join(MODELS_DIR, "modelo_pesos.npz")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler_params.npz")

TIEMPO_CONFIRMACION = 1.0
CONFIANZA_MINIMA_DEFAULT = 0.80
HISTORIAL_MAX = 10
UMBRAL_PISO = 0.70
UMBRAL_TECHO = 0.90
S3_INCERTIDUMBRE_COOLDOWN = 2.0

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(SERVICE_NAME)

app = FastAPI(title="RAPIRO-LSA EC2 Backend", version="2.3.0")

modelo_data = {}
hand_detector = None
face_detector = None

dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION) if DYNAMODB_TABLE else None
table = dynamodb.Table(DYNAMODB_TABLE) if dynamodb else None
s3 = boto3.client("s3", region_name=AWS_REGION) if S3_BUCKET else None
polly = boto3.client("polly", region_name=AWS_REGION)

latest_frame_jpeg = None
live_estado = {"letra": "---", "confianza": 0.0, "texto": "", "manos": 0}
inicio_servidor = time.time()

metricas = {
    "frames_procesados": 0,
    "letras_confirmadas": 0,
    "sesiones_total": 0,
    "tiempo_mediapipe_acum": 0.0,
    "tiempo_inferencia_acum": 0.0,
    "tiempo_total_acum": 0.0,
    "confianza_acum": 0.0,
    "confianza_muestras": 0,
    "muestras_inciertas_s3": 0,
    "umbral_actual": CONFIANZA_MINIMA_DEFAULT,
    "adaptaciones_umbral": 0,
    "fps_actual": 0.0,
    "ultimo_tiempo_mediapipe_ms": 0.0,
    "ultimo_tiempo_inferencia_ms": 0.0,
    "ultimo_tiempo_total_ms": 0.0,
}

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

CONEXIONES = None
if MEDIAPIPE_AVAILABLE:
    CONEXIONES = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def dibujar_frame(frame, hand_result, face_result, prediccion, confianza, texto, progreso, umbral):
    if not CV2_AVAILABLE: return frame
    h, w = frame.shape[:2]
    colores_mano = [(0, 220, 0), (220, 160, 0)]
    for idx, landmarks in enumerate(hand_result.hand_landmarks):
        color = colores_mano[min(idx, 1)]
        puntos = [(int(lm.x*w), int(lm.y*h)) for lm in landmarks]
        if CONEXIONES:
            for conn in CONEXIONES:
                cv2.line(frame, puntos[conn.start], puntos[conn.end], (255,255,255), 2)
        for px, py in puntos:
            cv2.circle(frame, (px, py), 5, color, -1)
    nose_x, nose_y = obtener_nariz(face_result)
    if nose_x is not None:
        cv2.circle(frame, (int(nose_x*w), int(nose_y*h)), 8, (220, 0, 220), -1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, 130), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    color_pred = (0,220,0) if confianza >= 0.85 else (0,180,220) if confianza > 0 else (160,160,160)
    cv2.putText(frame, str(prediccion), (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.2, color_pred, 5, cv2.LINE_AA)
    if confianza > 0:
        cv2.putText(frame, f"{confianza*100:.0f}%", (140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_pred, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (140,75), (340,87), (60,60,60), -1)
    if progreso > 0:
        cv2.rectangle(frame, (140,75), (140+int(progreso*200),87), color_pred, -1)
    cv2.putText(frame, f"FPS:{metricas['fps_actual']:.1f}", (140, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"MP:{metricas['ultimo_tiempo_mediapipe_ms']:.0f}ms", (230, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"IA:{metricas['ultimo_tiempo_inferencia_ms']:.1f}ms", (330, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Umbral:{umbral:.2f}", (430, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,180,0), 1, cv2.LINE_AA)
    cv2.putText(frame, "CLOUD", (w-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,0,220), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Frames:{metricas['frames_procesados']} Letras:{metricas['letras_confirmadas']} S3:{metricas['muestras_inciertas_s3']}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,120,120), 1, cv2.LINE_AA)
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h-60), (w, h), (0,0,0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Texto:", (10, h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)
    mostrar = texto if texto else "|"
    if len(mostrar) > 40: mostrar = "..." + mostrar[-37:]
    cv2.putText(frame, mostrar, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return frame

def _guardar_incertidumbre_s3_sync(vector, prediccion, confianza, session_id):
    if not s3 or not S3_BUCKET: return
    try:
        key = f"incertidumbre/{session_id}/{int(time.time()*1000)}.json"
        data = {"vector": vector, "prediccion": prediccion, "confianza": confianza,
                "timestamp": time.time(), "session_id": session_id, "necesita_revision": True}
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=json.dumps(data).encode("utf-8"), ContentType="application/json")
        logger.info(f"Incertidumbre guardada: {prediccion} ({confianza:.2f})")
    except Exception as e:
        logger.error(f"S3 incertidumbre error: {e}")

async def guardar_incertidumbre_s3(vector, prediccion, confianza, session_id):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _guardar_incertidumbre_s3_sync, vector, prediccion, confianza, session_id)

def _enviar_web_sync(endpoint, data):
    try:
        body = json.dumps(data).encode("utf-8")
        req = Request(f"{WEB_URL}/api/{endpoint}", data=body, headers={"Content-Type": "application/json"}, method="POST")
        urlopen(req, timeout=3)
    except Exception: pass

async def enviar_a_web(endpoint, data):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _enviar_web_sync, endpoint, data)

def _generar_audio_sync(texto):
    try:
        response = polly.synthesize_speech(Text=f"La sena detectada fue: {texto}", OutputFormat="mp3", VoiceId="Lupe", LanguageCode="es-US")
        audio_data = response["AudioStream"].read()
        logger.info(f"Polly OK: {len(audio_data)} bytes para '{texto}'")
        return audio_data
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

def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_TOKEN and x_api_key != API_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalido")

@app.on_event("startup")
async def startup_event():
    global hand_detector, face_detector
    logger.info("Inicializando backend v2.3...")
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
    logger.info("Backend v2.3 listo.")

class EventRequest(BaseModel):
    SessionId: str = Field(default="manual-test-001")
    DetectedSign: str = Field(default="Hola")
    Confidence: float = Field(default=0.9, ge=0, le=1)
    DeviceId: str = Field(default="rapiro-lsa-ec2")
    Mode: str = Field(default="word")
    Source: str = Field(default="EC2 Manual Test")

@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_NAME, "version": "2.3",
            "cv2_available": CV2_AVAILABLE, "mediapipe_available": MEDIAPIPE_AVAILABLE,
            "modelo_cargado": bool(modelo_data), "clases": modelo_data.get("clases", []),
            "errors": {"cv2": CV2_ERROR, "mediapipe": MEDIAPIPE_ERROR}}

@app.get("/metrics")
def get_metrics():
    uptime = time.time() - inicio_servidor
    n = metricas["frames_procesados"] or 1
    return {
        "uptime_segundos": round(uptime, 1),
        "frames_procesados": metricas["frames_procesados"],
        "letras_confirmadas": metricas["letras_confirmadas"],
        "sesiones_total": metricas["sesiones_total"],
        "fps_actual": round(metricas["fps_actual"], 1),
        "rendimiento": {
            "mediapipe_promedio_ms": round((metricas["tiempo_mediapipe_acum"] / n) * 1000, 1),
            "inferencia_promedio_ms": round((metricas["tiempo_inferencia_acum"] / n) * 1000, 2),
            "total_promedio_ms": round((metricas["tiempo_total_acum"] / n) * 1000, 1),
            "ultimo_mediapipe_ms": round(metricas["ultimo_tiempo_mediapipe_ms"], 1),
            "ultimo_inferencia_ms": round(metricas["ultimo_tiempo_inferencia_ms"], 2),
            "ultimo_total_ms": round(metricas["ultimo_tiempo_total_ms"], 1),
        },
        "confianza_promedio": round(metricas["confianza_acum"] / max(metricas["confianza_muestras"], 1), 3),
        "aprendizaje": {
            "umbral_actual": round(metricas["umbral_actual"], 3),
            "umbral_default": CONFIANZA_MINIMA_DEFAULT,
            "adaptaciones_realizadas": metricas["adaptaciones_umbral"],
            "muestras_inciertas_recolectadas": metricas["muestras_inciertas_s3"],
        },
    }

@app.post("/event")
async def event(payload: EventRequest, _: None = Depends(require_api_key)):
    evento = payload.model_dump(); evento["Timestamp"] = int(time.time())
    await guardar_evento(evento)
    return {"message": "Evento registrado", "event": evento}

def generar_mjpeg():
    while True:
        if latest_frame_jpeg is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame_jpeg + b"\r\n")
        time.sleep(0.02)

@app.get("/video")
def video_feed():
    return StreamingResponse(generar_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

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
  h1{font-size:1.1rem;color:#71717a;margin-bottom:16px;letter-spacing:2px;text-transform:uppercase}
  .video-container{position:relative;border-radius:14px;overflow:hidden;border:2px solid #27272a;max-width:680px;width:100%}
  .video-container img{width:100%;display:block}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:14px;width:100%;max-width:680px}
  .card{background:#18181b;border:1px solid #27272a;border-radius:10px;padding:12px;text-align:center}
  .card-label{font-size:0.65rem;color:#52525b;text-transform:uppercase;letter-spacing:1px}
  .card-value{font-size:1.6rem;font-weight:700;margin-top:2px}
  .green{color:#22c55e} .purple{color:#a855f7} .yellow{color:#eab308} .white{color:#e4e4e7} .cyan{color:#06b6d4}
  .badge{background:#1e1b4b;color:#a855f7;padding:4px 12px;border-radius:20px;font-size:0.65rem;font-weight:600;letter-spacing:1px;margin-top:12px}
  .offline{color:#ef4444;font-size:1rem;padding:30px;text-align:center}
</style>
</head>
<body>
  <h1>RAPIRO LSA - Deteccion en vivo + Metricas</h1>
  <div class="video-container">
    <img id="videoFeed" src="/video" alt="Video en vivo"
         onerror="this.style.display='none';document.getElementById('off').style.display='block'">
    <div id="off" class="offline" style="display:none">Esperando conexion del Rapiro...</div>
  </div>
  <div class="grid">
    <div class="card"><div class="card-label">Letra</div><div class="card-value green" id="mLetra">---</div></div>
    <div class="card"><div class="card-label">Confianza</div><div class="card-value purple" id="mConf">0%</div></div>
    <div class="card"><div class="card-label">Manos</div><div class="card-value white" id="mManos">0</div></div>
    <div class="card"><div class="card-label">FPS</div><div class="card-value cyan" id="mFps">0</div></div>
    <div class="card"><div class="card-label">MediaPipe</div><div class="card-value yellow" id="mMP">0ms</div></div>
    <div class="card"><div class="card-label">Inferencia IA</div><div class="card-value yellow" id="mIA">0ms</div></div>
    <div class="card"><div class="card-label">Umbral adaptativo</div><div class="card-value yellow" id="mUmbral">0.80</div></div>
    <div class="card"><div class="card-label">Incertidumbres S3</div><div class="card-value purple" id="mS3">0</div></div>
    <div class="card"><div class="card-label">Frames totales</div><div class="card-value white" id="mFrames">0</div></div>
    <div class="card"><div class="card-label">Letras confirmadas</div><div class="card-value green" id="mLetras">0</div></div>
  </div>
  <div class="card" style="margin-top:14px;width:100%;max-width:680px">
    <div class="card-label">Texto acumulado</div>
    <div class="card-value white" style="font-size:1.8rem" id="mTexto">|</div>
  </div>
  <div class="badge">PROCESAMIENTO EN AWS CLOUD + METRICAS + APRENDIZAJE ADAPTATIVO</div>
<script>
async function poll() {
  try {
    const [e, m] = await Promise.all([
      fetch('/api/live_estado').then(r=>r.json()),
      fetch('/metrics').then(r=>r.json())
    ]);
    document.getElementById('mLetra').textContent = e.letra||'---';
    document.getElementById('mConf').textContent = (e.confianza*100).toFixed(0)+'%';
    document.getElementById('mManos').textContent = e.manos||'0';
    document.getElementById('mTexto').textContent = e.texto||'|';
    document.getElementById('mFps').textContent = m.fps_actual;
    document.getElementById('mMP').textContent = m.rendimiento.ultimo_mediapipe_ms+'ms';
    document.getElementById('mIA').textContent = m.rendimiento.ultimo_inferencia_ms+'ms';
    document.getElementById('mUmbral').textContent = m.aprendizaje.umbral_actual;
    document.getElementById('mS3').textContent = m.aprendizaje.muestras_inciertas_recolectadas;
    document.getElementById('mFrames').textContent = m.frames_procesados;
    document.getElementById('mLetras').textContent = m.letras_confirmadas;
  } catch(e){}
}
setInterval(poll, 400);
</script>
</body>
</html>"""

@app.get("/live", response_class=HTMLResponse)
def live_page():
    return LIVE_HTML

@app.get("/api/live_estado")
def get_live_estado():
    return live_estado

@app.websocket("/ws/frame")
async def websocket_endpoint(websocket: WebSocket):
    global latest_frame_jpeg

    await websocket.accept()
    session_id = f"ws-{int(time.time())}"
    metricas["sesiones_total"] += 1
    logger.info(f"Rapiro conectado | Session: {session_id}")

    texto_formado = ""
    historial = []
    letra_candidata = None
    tiempo_inicio = 0.0
    letra_confirmada = False
    cooldown_hasta = 0.0
    ultimo_web = 0.0
    ultimo_dynamo = 0.0
    ultimo_s3 = 0.0  # FIX: cooldown para S3

    historial_confianza = deque(maxlen=30)
    umbral_sesion = CONFIANZA_MINIMA_DEFAULT

    try:
        while True:
            message = await websocket.receive_bytes()
            if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE or not hand_detector:
                await websocket.send_json({"error": "dependencias no disponibles"})
                continue

            t0 = time.time()

            frame_np = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            if frame is None: continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            t_mp = time.time()
            hand_result = hand_detector.detect(mp_img)
            face_result = face_detector.detect(mp_img)
            tiempo_mediapipe = time.time() - t_mp

            ahora = time.time()
            num_manos = len(hand_result.hand_landmarks)
            prediccion = "---"; confianza = 0.0; confirmada = False
            progreso = 0.0; audio_bytes = None; texto_a_hablar = ""
            tiempo_inferencia = 0.0

            if num_manos > 0:
                nose_x, nose_y = obtener_nariz(face_result)
                vector = extraer_vector(hand_result, nose_x, nose_y)

                t_inf = time.time()
                prediccion, confianza = predecir(vector)
                tiempo_inferencia = time.time() - t_inf

                metricas["confianza_acum"] += confianza
                metricas["confianza_muestras"] += 1

                # APRENDIZAJE: umbral adaptativo (logica corregida)
                historial_confianza.append(confianza)
                if len(historial_confianza) >= 10:
                    promedio_conf = sum(historial_confianza) / len(historial_confianza)
                    if promedio_conf > 0.90:
                        nuevo_umbral = UMBRAL_TECHO  # 0.90 - experto, ser estricto
                    elif promedio_conf > 0.80:
                        nuevo_umbral = 0.80  # bueno, mantener standard
                    else:
                        nuevo_umbral = UMBRAL_PISO  # 0.70 - dificultad, ayudar
                    if abs(nuevo_umbral - umbral_sesion) > 0.01:
                        logger.info(f"Umbral: {umbral_sesion:.2f} -> {nuevo_umbral:.2f} (avg: {promedio_conf:.2f})")
                        metricas["adaptaciones_umbral"] += 1
                    umbral_sesion = nuevo_umbral
                    metricas["umbral_actual"] = umbral_sesion

                # APRENDIZAJE: recolectar incertidumbre (con cooldown)
                if 0.0 < confianza < 0.75 and ahora - ultimo_s3 > S3_INCERTIDUMBRE_COOLDOWN:
                    metricas["muestras_inciertas_s3"] += 1
                    ultimo_s3 = ahora
                    asyncio.create_task(guardar_incertidumbre_s3(vector, prediccion, confianza, session_id))

                historial.append(prediccion)
                if len(historial) > HISTORIAL_MAX: historial.pop(0)
                prediccion = max(set(historial), key=historial.count)

                if prediccion == "NADA":
                    letra_candidata = None; tiempo_inicio = 0.0; letra_confirmada = False
                elif confianza >= umbral_sesion and ahora > cooldown_hasta:
                    if prediccion == letra_candidata:
                        progreso = min((ahora - tiempo_inicio) / TIEMPO_CONFIRMACION, 1.0)
                        if progreso >= 1.0 and not letra_confirmada:
                            confirmada = True
                            metricas["letras_confirmadas"] += 1
                            if prediccion == "FINALIZAR" and texto_formado:
                                texto_a_hablar = texto_formado
                                logger.info(f"FINALIZAR: \"{texto_a_hablar}\"")
                                asyncio.create_task(enviar_a_web("finalizar", {"texto": texto_a_hablar}))
                                asyncio.create_task(guardar_evento({"SessionId": session_id, "Timestamp": int(time.time()),
                                    "DetectedSign": "FINALIZAR", "TextoCompleto": texto_a_hablar,
                                    "Confidence": round(confianza,4), "Source": "EC2 WebSocket"}))
                                audio_bytes = await generar_audio(texto_a_hablar)
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

            tiempo_total = time.time() - t0
            metricas["frames_procesados"] += 1
            metricas["tiempo_mediapipe_acum"] += tiempo_mediapipe
            metricas["tiempo_inferencia_acum"] += tiempo_inferencia
            metricas["tiempo_total_acum"] += tiempo_total
            metricas["ultimo_tiempo_mediapipe_ms"] = tiempo_mediapipe * 1000
            metricas["ultimo_tiempo_inferencia_ms"] = tiempo_inferencia * 1000
            metricas["ultimo_tiempo_total_ms"] = tiempo_total * 1000
            metricas["fps_actual"] = 1.0 / tiempo_total if tiempo_total > 0 else 0.0

            texto_mostrar = texto_a_hablar if texto_a_hablar else texto_formado
            frame_dibujado = dibujar_frame(frame, hand_result, face_result,
                                          prediccion, confianza, texto_mostrar, progreso, umbral_sesion)
            _, jpeg_buf = cv2.imencode(".jpg", frame_dibujado, [cv2.IMWRITE_JPEG_QUALITY, 70])
            latest_frame_jpeg = jpeg_buf.tobytes()

            live_estado["letra"] = prediccion
            live_estado["confianza"] = round(confianza, 3)
            live_estado["texto"] = texto_mostrar
            live_estado["manos"] = num_manos

            if ahora - ultimo_web > 0.5:
                asyncio.create_task(enviar_a_web("letra", {"letra": prediccion, "confianza": confianza, "texto": texto_formado}))
                ultimo_web = ahora

            if confirmada and prediccion != "FINALIZAR" and ahora - ultimo_dynamo > 3.0:
                asyncio.create_task(guardar_evento({"SessionId": session_id, "Timestamp": int(time.time()),
                    "DetectedSign": prediccion, "TextoAcumulado": texto_formado,
                    "Confidence": round(confianza,4), "Source": "EC2 WebSocket"}))
                ultimo_dynamo = ahora

            respuesta = {"letra": prediccion, "confianza": round(confianza,3), "texto": texto_formado,
                         "confirmada": confirmada, "progreso": round(progreso,2), "manos": num_manos,
                         "texto_hablado": texto_a_hablar, "umbral": round(umbral_sesion,3)}
            await websocket.send_json(respuesta)
            if audio_bytes:
                await websocket.send_bytes(audio_bytes)
                logger.info(f"Audio enviado: {len(audio_bytes)} bytes")

    except WebSocketDisconnect: logger.info(f"Rapiro desconectado | Session: {session_id}")
    except Exception as e: logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)