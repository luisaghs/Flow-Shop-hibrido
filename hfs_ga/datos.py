"""
Lectura y validación de los datos de entrada del problema

    FH4, ((P4), (Q20), (P9), (P8)) | ls, trans, am | ΣTj

Los datos pueden entregarse de dos formas:

1. Un único libro de Excel (.xlsx) con las hojas
   ``Trabajos``, ``Tiempos``, ``Maquinas`` y ``Transporte``.
2. Una carpeta con los archivos CSV
   ``trabajos.csv``, ``tiempos.csv``, ``maquinas.csv`` y ``transporte.csv``.

Ver README.md para la descripción detallada de cada hoja/columna.
Todas las magnitudes de tiempo están en HORAS.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NOMBRES_ETAPAS = ["Troquelado", "Ensamble", "Costura", "Volteado"]

# Configuración del campo α del problema: (tipo de máquinas, número de máquinas)
# P = paralelas idénticas, Q = paralelas uniformes
CONFIG_ESPERADA = [("P", 4), ("Q", 20), ("P", 9), ("P", 8)]

HOJAS = ["Trabajos", "Tiempos", "Maquinas", "Transporte"]


@dataclass
class Instancia:
    """Representación interna de una instancia del problema."""

    trabajos: list[str]                 # nombres de los trabajos (lotes)
    cantidad: np.ndarray                # (n,) unidades por lote
    fecha_entrega: np.ndarray           # (n,) fecha de entrega d_j [h]
    max_sublotes: np.ndarray            # (n,) máximo número de sublotes por lote
    tiempo_unitario: np.ndarray         # (n, K) horas por unidad en cada etapa (velocidad 1)
    transporte: np.ndarray              # (n, K-1) tiempo de transporte etapa k -> k+1 [h]
    velocidades: list[np.ndarray]       # por etapa: (m_k,) velocidad relativa de cada máquina
    disponibilidad: list[np.ndarray]    # por etapa: (m_k,) instante de disponibilidad inicial [h]
    nombres_maquinas: list[list[str]]   # por etapa: nombres de las máquinas
    nombres_etapas: list[str] = field(default_factory=lambda: list(NOMBRES_ETAPAS))

    @property
    def n(self) -> int:
        return len(self.trabajos)

    @property
    def K(self) -> int:
        return self.tiempo_unitario.shape[1]

    def resumen(self) -> str:
        lineas = [
            f"Trabajos: {self.n}  |  Unidades totales: {int(self.cantidad.sum())}",
            "Etapas:",
        ]
        for k in range(self.K):
            v = self.velocidades[k]
            tipo = "idénticas" if np.allclose(v, v[0]) else "uniformes"
            lineas.append(
                f"  {k + 1}. {self.nombres_etapas[k]:<11} {len(v):>2} máquinas {tipo}"
                f"  (disp. inicial {self.disponibilidad[k].min():.2f}-{self.disponibilidad[k].max():.2f} h)"
            )
        lineas.append(
            f"Fechas de entrega: {self.fecha_entrega.min():.1f} - {self.fecha_entrega.max():.1f} h"
        )
        return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower()
        .replace(" ", "_")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        for c in df.columns
    ]
    # eliminar filas completamente vacías
    return df.dropna(how="all")


def _leer_tablas(ruta: str) -> dict[str, pd.DataFrame]:
    if os.path.isdir(ruta):
        tablas = {}
        for hoja in HOJAS:
            archivo = os.path.join(ruta, hoja.lower() + ".csv")
            if not os.path.exists(archivo):
                raise FileNotFoundError(f"No se encontró el archivo '{archivo}'.")
            tablas[hoja] = pd.read_csv(archivo)
    else:
        ext = os.path.splitext(ruta)[1].lower()
        if ext not in (".xlsx", ".xlsm", ".xls"):
            raise ValueError("El archivo de datos debe ser Excel (.xlsx) o una carpeta con CSV.")
        libro = pd.read_excel(ruta, sheet_name=None)
        nombres = {k.strip().lower(): k for k in libro}
        tablas = {}
        for hoja in HOJAS:
            if hoja.lower() not in nombres:
                raise ValueError(
                    f"Falta la hoja '{hoja}' en el archivo Excel. "
                    f"Hojas encontradas: {list(libro)}"
                )
            tablas[hoja] = libro[nombres[hoja.lower()]]
    return {k: _normalizar_columnas(v) for k, v in tablas.items()}


def _exigir_columnas(df: pd.DataFrame, columnas: list[str], hoja: str) -> None:
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(
            f"En la hoja '{hoja}' faltan las columnas {faltan}. "
            f"Columnas encontradas: {list(df.columns)}"
        )


def cargar_datos(ruta: str, max_sublotes_defecto: int = 3) -> Instancia:
    """Lee los datos desde Excel o carpeta CSV y construye una :class:`Instancia`."""
    t = _leer_tablas(ruta)

    # ---------------- Trabajos ----------------
    tr = t["Trabajos"]
    _exigir_columnas(tr, ["trabajo", "cantidad", "fecha_entrega_h"], "Trabajos")
    tr["trabajo"] = tr["trabajo"].astype(str).str.strip()
    if tr["trabajo"].duplicated().any():
        raise ValueError("Hay trabajos repetidos en la hoja 'Trabajos'.")
    trabajos = tr["trabajo"].tolist()
    cantidad = tr["cantidad"].astype(float).to_numpy()
    if (cantidad <= 0).any() or not np.allclose(cantidad, np.round(cantidad)):
        raise ValueError("La 'cantidad' de cada trabajo debe ser un entero positivo.")
    cantidad = np.round(cantidad).astype(int)
    fecha = tr["fecha_entrega_h"].astype(float).to_numpy()
    if "max_sublotes" in tr.columns:
        ms = tr["max_sublotes"].fillna(max_sublotes_defecto).astype(int).to_numpy()
    else:
        ms = np.full(len(trabajos), max_sublotes_defecto, dtype=int)
    ms = np.clip(ms, 1, None)
    ms = np.minimum(ms, cantidad)  # no puede haber más sublotes que unidades
    idx = {j: i for i, j in enumerate(trabajos)}

    # ---------------- Tiempos de proceso ----------------
    ti = t["Tiempos"]
    cols_etapa = [f"etapa_{k + 1}" for k in range(len(NOMBRES_ETAPAS))]
    _exigir_columnas(ti, ["trabajo"] + cols_etapa, "Tiempos")
    ti["trabajo"] = ti["trabajo"].astype(str).str.strip()
    p = np.full((len(trabajos), len(cols_etapa)), np.nan)
    for _, fila in ti.iterrows():
        if fila["trabajo"] not in idx:
            raise ValueError(f"El trabajo '{fila['trabajo']}' de 'Tiempos' no existe en 'Trabajos'.")
        p[idx[fila["trabajo"]]] = [float(fila[c]) for c in cols_etapa]
    if np.isnan(p).any():
        faltan = [trabajos[i] for i in np.where(np.isnan(p).any(axis=1))[0]]
        raise ValueError(f"Faltan tiempos de proceso para: {faltan}")
    if (p < 0).any():
        raise ValueError("Los tiempos de proceso no pueden ser negativos.")

    # ---------------- Máquinas ----------------
    ma = t["Maquinas"]
    _exigir_columnas(ma, ["etapa", "maquina"], "Maquinas")
    if "velocidad" not in ma.columns:
        ma["velocidad"] = 1.0
    if "disponibilidad_h" not in ma.columns:
        ma["disponibilidad_h"] = 0.0
    ma["velocidad"] = ma["velocidad"].fillna(1.0).astype(float)
    ma["disponibilidad_h"] = ma["disponibilidad_h"].fillna(0.0).astype(float)
    ma["etapa"] = ma["etapa"].astype(int)
    velocidades, disponibilidad, nombres_maq = [], [], []
    for k in range(len(NOMBRES_ETAPAS)):
        sub = ma[ma["etapa"] == k + 1]
        if sub.empty:
            raise ValueError(f"No hay máquinas definidas para la etapa {k + 1}.")
        if (sub["velocidad"] <= 0).any():
            raise ValueError(f"Velocidades no positivas en la etapa {k + 1}.")
        velocidades.append(sub["velocidad"].to_numpy())
        disponibilidad.append(sub["disponibilidad_h"].to_numpy())
        nombres_maq.append(sub["maquina"].astype(str).tolist())

        tipo, m_esp = CONFIG_ESPERADA[k]
        if len(sub) != m_esp:
            warnings.warn(
                f"La etapa {k + 1} ({NOMBRES_ETAPAS[k]}) tiene {len(sub)} máquinas; "
                f"la configuración de referencia indica {m_esp}."
            )
        if tipo == "P" and not np.allclose(sub["velocidad"], sub["velocidad"].iloc[0]):
            warnings.warn(
                f"La etapa {k + 1} debería tener máquinas idénticas (P) pero las "
                "velocidades difieren; se tratarán como uniformes."
            )

    # ---------------- Transporte ----------------
    tp = t["Transporte"]
    _exigir_columnas(tp, ["de_etapa", "a_etapa", "tiempo_h"], "Transporte")
    K = len(NOMBRES_ETAPAS)
    transporte = np.zeros((len(trabajos), K - 1))
    tiene_trabajo = "trabajo" in tp.columns
    # primero filas generales (sin trabajo), luego filas específicas que las sobrescriben
    filas = list(tp.iterrows())
    for especifico in (False, True):
        for _, fila in filas:
            trab = fila["trabajo"] if tiene_trabajo else np.nan
            es_esp = tiene_trabajo and pd.notna(trab) and str(trab).strip() != ""
            if es_esp != especifico:
                continue
            a, b = int(fila["de_etapa"]), int(fila["a_etapa"])
            if b != a + 1 or not (1 <= a < K):
                raise ValueError(
                    f"Transporte inválido de la etapa {a} a la {b}: solo se admiten etapas consecutivas."
                )
            if especifico:
                trab = str(trab).strip()
                if trab not in idx:
                    raise ValueError(f"El trabajo '{trab}' de 'Transporte' no existe.")
                transporte[idx[trab], a - 1] = float(fila["tiempo_h"])
            else:
                transporte[:, a - 1] = float(fila["tiempo_h"])

    return Instancia(
        trabajos=trabajos,
        cantidad=cantidad,
        fecha_entrega=fecha,
        max_sublotes=ms,
        tiempo_unitario=p,
        transporte=transporte,
        velocidades=velocidades,
        disponibilidad=disponibilidad,
        nombres_maquinas=nombres_maq,
    )
