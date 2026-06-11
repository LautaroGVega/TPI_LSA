"""
AWS Lambda — Inferencia real + envío a página web
----------------------------------------------------
Recibe vector de 131 valores, predice la letra,
guarda en DynamoDB, genera audio con Polly si FINALIZAR,
y MANDA CADA PREDICCIÓN a la página web en Render.

Verificar en: https://rapiro.onrender.com
"""

import json
import os
import time
import boto3
import numpy as np
from io import BytesIO
from urllib.request import Request, urlopen
from urllib.error import URLError

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
polly = boto3.client("polly")

BUCKET  = os.environ.get("BUCKET_NAME", "rapiro-lsa-models-datasets-295552411532")
TABLE   = os.environ.get("TABLE_NAME", "rapiro-lsa-sessions")
REGION  = os.environ.get("AWS_REGION", "us-east-2")
WEB_URL = os.environ.get("WEB_URL", "https://rapiro.onrender.com")

modelo_cache = {}


def cargar_modelo():
    if modelo_cache:
        return
    print("[INFO] Cargando modelo desde S3...")

    obj = s3.get_object(Bucket=BUCKET, Key="models/modelo_pesos.npz")
    with BytesIO(obj["Body"].read()) as f:
        pesos = dict(np.load(f))

    obj = s3.get_object(Bucket=BUCKET, Key="models/scaler_params.npz")
    with BytesIO(obj["Body"].read()) as f:
        scaler_data = dict(np.load(f, allow_pickle=True))

    modelo_cache["pesos"] = pesos
    modelo_cache["mean"] = scaler_data["mean"]
    modelo_cache["scale"] = scaler_data["scale"]
    modelo_cache["clases"] = list(scaler_data["clases"])
    print(f"[INFO] Clases: {modelo_cache['clases']}")


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predecir(vector):
    p = modelo_cache["pesos"]
    x = (np.array(vector, dtype=np.float32) - modelo_cache["mean"]) / modelo_cache["scale"]
    x = relu(x @ p["W0"] + p["b0"])
    x = relu(x @ p["W1"] + p["b1"])
    x = relu(x @ p["W2"] + p["b2"])
    x = softmax(x @ p["W3"] + p["b3"])
    idx = int(np.argmax(x))
    return modelo_cache["clases"][idx], float(x[idx])


def guardar_dynamodb(session_id, letra, confianza, source):
    table = dynamodb.Table(TABLE)
    table.put_item(Item={
        "SessionId": session_id,
        "Timestamp": int(time.time() * 1000),
        "DetectedSign": letra,
        "Confidence": str(round(confianza, 4)),
        "Source": source,
    })


def generar_audio(texto, session_id):
    response = polly.synthesize_speech(
        Text=f"La seña detectada fue: {texto}",
        OutputFormat="mp3",
        VoiceId="Lupe",
        LanguageCode="es-US",
    )
    audio_key = f"audio/{session_id}_{int(time.time())}.mp3"
    s3.put_object(
        Bucket=BUCKET, Key=audio_key,
        Body=response["AudioStream"].read(),
        ContentType="audio/mpeg",
    )
    return audio_key


def enviar_a_web(endpoint, data):
    """Envía resultado a la página web en Render."""
    try:
        body = json.dumps(data).encode("utf-8")
        req = Request(
            f"{WEB_URL}/api/{endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urlopen(req, timeout=3)
        print(f"[WEB] Enviado a {WEB_URL}/api/{endpoint}")
    except (URLError, Exception) as e:
        print(f"[WEB] Error: {e}")


def lambda_handler(event, context):
    cargar_modelo()

    if isinstance(event, str):
        event = json.loads(event)

    vector = event.get("vector", [])
    session_id = event.get("SessionId", f"session-{int(time.time())}")
    source = event.get("Source", "unknown")
    texto_acumulado = event.get("texto_acumulado", "")
    es_finalizar = event.get("finalizar", False)

    if not vector or len(vector) != 131:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": f"Vector debe tener 131 valores, recibido: {len(vector)}"})
        }

    # Predecir
    letra, confianza = predecir(vector)

    # Guardar en DynamoDB
    guardar_dynamodb(session_id, letra, confianza, source)

    # Enviar predicción a la web (Render)
    enviar_a_web("letra", {
        "letra": letra,
        "confianza": confianza,
        "texto": texto_acumulado,
    })

    resultado = {
        "letra": letra,
        "confianza": round(confianza, 4),
        "session_id": session_id,
    }

    # Si FINALIZAR: generar audio + mandar a web
    if es_finalizar and texto_acumulado:
        audio_key = generar_audio(texto_acumulado, session_id)
        resultado["audio_key"] = audio_key
        resultado["audio_url"] = f"https://{BUCKET}.s3.{REGION}.amazonaws.com/{audio_key}"

        enviar_a_web("finalizar", {"texto": texto_acumulado})
        print(f"[FINALIZAR] Texto: {texto_acumulado} | Audio: {audio_key}")

    return {
        "statusCode": 200,
        "body": json.dumps(resultado)
    }