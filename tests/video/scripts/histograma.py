"""Genera histogramas de tiempos para las pruebas indoor y outdoor.

Por defecto busca ``processing_metrics.json`` dentro de los directorios cuyo
nombre contiene ``indoorTest`` u ``outdoorTest`` bajo ``tests/video/videos``.
Los dos primeros registros de cada archivo se descartan antes de graficar.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NOMBRES_PRUEBA = ("indoortest", "outdoortest")


def es_prueba_objetivo(archivo: Path, raiz: Path) -> bool:
    """Indica si el JSON pertenece a un directorio indoorTest/outdoorTest."""
    try:
        partes = archivo.relative_to(raiz).parts[:-1]
    except ValueError:
        partes = archivo.parts[:-1]
    return any(
        nombre in parte.lower()
        for parte in partes
        for nombre in NOMBRES_PRUEBA
    )


def buscar_metricas(raiz: Path) -> Iterable[Path]:
    return sorted(
        archivo
        for archivo in raiz.rglob("processing_metrics.json")
        if es_prueba_objetivo(archivo, raiz)
    )


def cargar_tiempos(archivo: Path) -> List[float]:
    with archivo.open("r", encoding="utf-8") as entrada:
        datos = json.load(entrada)

    frames = datos.get("frames")
    if not isinstance(frames, list):
        raise ValueError("el campo 'frames' no es una lista")

    tiempos: List[float] = []
    for indice, frame in enumerate(frames[2:], start=3):
        if not isinstance(frame, dict) or "processing_seconds" not in frame:
            raise ValueError(
                f"el resultado {indice} no contiene 'processing_seconds'"
            )
        tiempos.append(float(frame["processing_seconds"]))

    if not tiempos:
        raise ValueError("no quedan resultados después de eliminar los dos primeros")
    return tiempos


def numero_clases_sturges(cantidad_datos: int) -> int:
    """Calcula k = ceil(1 + log2(n)), según la regla de Sturges."""
    return math.ceil(1 + math.log2(cantidad_datos))


def crear_histograma(archivo: Path, tiempos: List[float], bins: int) -> Path:
    salida = archivo.with_name("histograma_tiempos_de_procesamiento.png")
    promedio = sum(tiempos) / len(tiempos)

    nombre_directorio = archivo.parent.name
    nombre_prueba = (
        "Prueba en interiores"
        if "indoortest" in nombre_directorio.lower()
        else "Prueba en exteriores"
    )

    figura, eje = plt.subplots(figsize=(10, 6))
    eje.hist(tiempos, bins=bins, color="#2878B5", edgecolor="white")
    eje.axvline(
        promedio,
        color="#D95319",
        linestyle="--",
        linewidth=2,
        label=f"Promedio: {promedio:.4f} s",
    )
    eje.set_title(f"Distribución de tiempos de procesamiento — {nombre_prueba}")
    eje.set_xlabel("Tiempo de procesamiento por frame (segundos)")
    eje.set_ylabel("Frecuencia")
    eje.grid(axis="y", alpha=0.25)
    eje.legend()
    figura.tight_layout()
    figura.savefig(salida, dpi=150)
    plt.close(figura)
    return salida


def main() -> int:
    raiz_predeterminada = Path(__file__).resolve().parents[1] / "videos"
    parser = argparse.ArgumentParser(
        description=(
            "Crea un histograma por processing_metrics.json de indoorTest y "
            "outdoorTest, omitiendo los dos primeros resultados."
        )
    )
    parser.add_argument(
        "raiz",
        nargs="?",
        type=Path,
        default=raiz_predeterminada,
        help="Directorio desde el cual buscar (predeterminado: tests/video/videos)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help=(
            "Cantidad de intervalos; si se omite, se calcula con la regla "
            "de Sturges"
        ),
    )
    args = parser.parse_args()

    raiz = args.raiz.resolve()
    if not raiz.is_dir():
        parser.error(f"el directorio no existe: {raiz}")
    if args.bins is not None and args.bins <= 0:
        parser.error("--bins debe ser mayor que cero")

    archivos = list(buscar_metricas(raiz))
    if not archivos:
        print(f"No se encontraron archivos de métricas en {raiz}")
        return 1

    hubo_errores = False
    for archivo in archivos:
        try:
            tiempos = cargar_tiempos(archivo)
            bins = args.bins or numero_clases_sturges(len(tiempos))
            salida = crear_histograma(archivo, tiempos, bins)
            print(
                f"Creado: {salida} "
                f"({len(tiempos)} resultados, {bins} intervalos)"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            hubo_errores = True
            print(f"Error en {archivo}: {error}")

    return 1 if hubo_errores else 0


if __name__ == "__main__":
    raise SystemExit(main())
