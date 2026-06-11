"""
Convierte dataset.csv de 63 valores a 131 valores
----------------------------------------------------
Agrega 68 ceros al final de cada fila:
  - 63 ceros → mano secundaria (no había)
  - 3 ceros  → distancia entre manos (no había)
  - 2 ceros  → posición facial (no se capturó)

Uso:
  py -3.12 convertir_dataset.py

Genera: dataset_131.csv (el original no se modifica)
"""

import csv
import os

ENTRADA = "dataset.csv"
SALIDA  = "dataset_131.csv"
CEROS_AGREGAR = 68   # 63 (mano 2) + 3 (distancia) + 2 (cara)


def main():
    if not os.path.exists(ENTRADA):
        print(f"[ERROR] No se encontró {ENTRADA}")
        return

    filas_convertidas = 0
    conteos = {}

    with open(ENTRADA, "r") as f_in, open(SALIDA, "w", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            if not row:
                continue

            letra = row[0]
            valores = row[1:]

            if len(valores) == 131:
                # Ya tiene 131 valores, copiar tal cual
                writer.writerow(row)
            elif len(valores) == 63:
                # Agregar 68 ceros
                nuevos_valores = valores + ["0.0"] * CEROS_AGREGAR
                writer.writerow([letra] + nuevos_valores)
                filas_convertidas += 1
            else:
                print(f"  [ADVERTENCIA] Fila con {len(valores)} valores (esperado 63 o 131), saltada")
                continue

            conteos[letra] = conteos.get(letra, 0) + 1

    print(f"\n── Conversión completada ──")
    print(f"  Filas convertidas: {filas_convertidas}")
    print(f"\n  Distribución:")
    for letra, n in sorted(conteos.items()):
        print(f"    {letra}: {n} muestras")
    print(f"\n  Total: {sum(conteos.values())} muestras")
    print(f"\n[INFO] Archivo generado: {SALIDA}")
    print(f"[INFO] El archivo original {ENTRADA} no fue modificado.")
    print(f"\n[SIGUIENTE PASO]")
    print(f"  1. Renombrá {ENTRADA} a dataset_viejo.csv (backup)")
    print(f"  2. Renombrá {SALIDA} a {ENTRADA}")
    print(f"  3. Las nuevas capturas con capturar.py ya van a generar 131 valores")
    print(f"  4. Ambos formatos conviven en el mismo archivo")


if __name__ == "__main__":
    main()