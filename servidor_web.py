"""
Servidor web — Muestra la traducción LSA en tiempo real
---------------------------------------------------------
Abre una página web donde se ve el texto que el sistema va formando.
Cuando recibe FINALIZAR, muestra el mensaje completo.

Uso:
  py -3.12 servidor_web.py

Después abrí en el navegador:
  http://localhost:5000
"""

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# Estado global
estado = {
    "texto_actual": "",
    "letra_actual": "",
    "confianza": 0.0,
    "finalizado": False,
    "historial": [],
}

HTML_PAGE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAPIRO LSA — Traductor de Lengua de Señas</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f1117;
    color: #e4e4e7;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
  }
  h1 {
    font-size: 1.4rem;
    font-weight: 500;
    color: #71717a;
    margin-bottom: 40px;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .letra-actual {
    font-size: 8rem;
    font-weight: 700;
    color: #22c55e;
    line-height: 1;
    min-height: 130px;
    transition: all 0.2s;
  }
  .letra-actual.nada { color: #3f3f46; }
  .letra-actual.finalizar { color: #a855f7; }
  .confianza {
    font-size: 1.2rem;
    color: #71717a;
    margin-top: 8px;
    min-height: 24px;
  }
  .texto-container {
    margin-top: 50px;
    width: 100%;
    max-width: 800px;
  }
  .texto-label {
    font-size: 0.85rem;
    color: #52525b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }
  .texto-formado {
    font-size: 2.8rem;
    font-weight: 600;
    color: #e4e4e7;
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 16px;
    padding: 30px 36px;
    min-height: 100px;
    word-wrap: break-word;
    transition: all 0.3s;
  }
  .texto-formado.finalizado {
    border-color: #22c55e;
    box-shadow: 0 0 30px rgba(34, 197, 94, 0.15);
    color: #22c55e;
  }
  .cursor { animation: blink 1s infinite; }
  @keyframes blink { 0%,50% { opacity: 1; } 51%,100% { opacity: 0; } }
  .historial {
    margin-top: 40px;
    width: 100%;
    max-width: 800px;
  }
  .historial-item {
    background: #18181b;
    border: 1px solid #27272a;
    border-radius: 10px;
    padding: 16px 20px;
    margin-top: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .historial-texto { font-size: 1.1rem; color: #a1a1aa; }
  .historial-hora { font-size: 0.8rem; color: #52525b; }
  .estado-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 20px;
  }
  .estado-badge.activo { background: #052e16; color: #22c55e; }
  .estado-badge.esperando { background: #1c1917; color: #78716c; }
  .estado-badge.completado { background: #1e1b4b; color: #a855f7; }
</style>
</head>
<body>
  <h1>Rapiro LSA — Traductor en tiempo real</h1>

  <div class="letra-actual" id="letra">—</div>
  <div class="confianza" id="confianza"></div>
  <div class="estado-badge esperando" id="estado">Esperando conexion</div>

  <div class="texto-container">
    <div class="texto-label">Mensaje</div>
    <div class="texto-formado" id="texto"><span class="cursor">|</span></div>
  </div>

  <div class="historial" id="historial"></div>

<script>
async function actualizar() {
  try {
    const res = await fetch('/api/estado');
    const data = await res.json();

    const letraEl = document.getElementById('letra');
    const confEl = document.getElementById('confianza');
    const textoEl = document.getElementById('texto');
    const estadoEl = document.getElementById('estado');

    // Letra actual
    letraEl.textContent = data.letra_actual || '—';
    letraEl.className = 'letra-actual';
    if (data.letra_actual === 'NADA') letraEl.classList.add('nada');
    if (data.letra_actual === 'FINALIZAR') letraEl.classList.add('finalizar');

    // Confianza
    if (data.confianza > 0) {
      confEl.textContent = (data.confianza * 100).toFixed(0) + '% confianza';
    } else {
      confEl.textContent = '';
    }

    // Texto formado
    if (data.finalizado) {
      textoEl.textContent = data.texto_actual;
      textoEl.className = 'texto-formado finalizado';
      estadoEl.textContent = 'Mensaje completo';
      estadoEl.className = 'estado-badge completado';
    } else if (data.texto_actual) {
      textoEl.innerHTML = data.texto_actual + '<span class="cursor">|</span>';
      textoEl.className = 'texto-formado';
      estadoEl.textContent = 'Detectando';
      estadoEl.className = 'estado-badge activo';
    } else {
      textoEl.innerHTML = '<span class="cursor">|</span>';
      textoEl.className = 'texto-formado';
      estadoEl.textContent = 'Esperando senas';
      estadoEl.className = 'estado-badge esperando';
    }

    // Historial
    const histEl = document.getElementById('historial');
    if (data.historial && data.historial.length > 0) {
      histEl.innerHTML = '<div class="texto-label">Historial</div>' +
        data.historial.map(h =>
          '<div class="historial-item">' +
            '<span class="historial-texto">' + h.texto + '</span>' +
            '<span class="historial-hora">' + h.hora + '</span>' +
          '</div>'
        ).reverse().join('');
    }

  } catch (e) {}
}

setInterval(actualizar, 300);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/estado")
def get_estado():
    return jsonify(estado)


@app.route("/api/letra", methods=["POST"])
def recibir_letra():
    data = request.json
    estado["letra_actual"] = data.get("letra", "")
    estado["confianza"] = data.get("confianza", 0.0)
    estado["texto_actual"] = data.get("texto", "")
    estado["finalizado"] = False
    return jsonify({"ok": True})


@app.route("/api/finalizar", methods=["POST"])
def finalizar():
    data = request.json
    texto = data.get("texto", "")
    from datetime import datetime
    estado["texto_actual"] = texto
    estado["finalizado"] = True
    estado["letra_actual"] = "FINALIZAR"
    estado["historial"].append({
        "texto": texto,
        "hora": datetime.now().strftime("%H:%M:%S"),
    })
    print(f"\n  ✓ MENSAJE FINALIZADO: \"{texto}\"")
    return jsonify({"ok": True})


@app.route("/api/reset", methods=["POST"])
def reset():
    estado["texto_actual"] = ""
    estado["letra_actual"] = ""
    estado["confianza"] = 0.0
    estado["finalizado"] = False
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("=" * 50)
    print("  Servidor web LSA")
    print("  Abrir en navegador: http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)