"""
Representación del cromosoma y decodificación en un programa de producción.

Cromosoma (dos partes):

* ``secuencia``: vector de longitud Σ S_j donde cada trabajo j aparece S_j veces
  (S_j = máximo de sublotes del trabajo). La r-ésima aparición de j representa
  el sublote r de j. El orden define la prioridad de los sublotes en la etapa 1.
* ``claves``: matriz (n, S_max) de llaves aleatorias en [0, 1]. Determinan el
  tamaño de los sublotes de cada trabajo (lot streaming con sublotes
  CONSISTENTES: el mismo tamaño en todas las etapas). Una llave por debajo de
  ``umbral_cero`` anula el sublote, por lo que el número de sublotes también
  se optimiza (entre 1 y S_j).

Decodificación (esquema de lista clásico en flow shop híbrido):

* Etapa 1: los sublotes se programan en el orden de ``secuencia``.
* Etapas 2..K: los sublotes se ordenan por su instante de llegada
  (fin en la etapa anterior + tiempo de transporte) -> regla FIFO.
* En cada etapa el sublote se asigna a la máquina que lo termina primero
  (ECT, Earliest Completion Time), considerando la disponibilidad inicial de
  las máquinas (am) y su velocidad (máquinas uniformes: p / v).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datos import Instancia


@dataclass
class Cromosoma:
    secuencia: np.ndarray   # (L,) enteros con índices de trabajos
    claves: np.ndarray      # (n, S_max) floats en [0, 1]

    def copia(self) -> "Cromosoma":
        return Cromosoma(self.secuencia.copy(), self.claves.copy())


@dataclass
class Programa:
    tardanza_total: float
    makespan: float
    completacion: np.ndarray        # (n,) C_j
    tardanza: np.ndarray            # (n,) T_j
    tamanos: list[np.ndarray]       # por trabajo: tamaños de sus sublotes (>0)
    operaciones: list[tuple]        # (trabajo, sublote, tamaño, etapa, maquina, inicio, fin)


def tamanos_sublotes(claves_j: np.ndarray, cantidad: int, s_max: int,
                     umbral_cero: float, tam_min: int) -> np.ndarray:
    """Convierte las llaves de un trabajo en tamaños enteros de sublotes.

    Devuelve un vector de longitud ``s_max`` con los tamaños (pueden ser 0 =
    sublote inexistente). La suma siempre es igual a ``cantidad``.
    """
    w = claves_j[:s_max].astype(float).copy()
    w[w < umbral_cero] = 0.0
    if w.sum() <= 0:
        w[np.argmax(claves_j[:s_max])] = 1.0

    while True:
        prop = w / w.sum() * cantidad
        base = np.floor(prop).astype(int)
        resto = cantidad - base.sum()
        if resto > 0:  # método del mayor residuo
            orden = np.argsort(-(prop - base) + (w == 0) * 10)
            base[orden[:resto]] += 1
        activos = base > 0
        pequenos = activos & (base < tam_min)
        if not pequenos.any() or activos.sum() <= 1:
            return base
        # se elimina el sublote activo más pequeño y se reparte de nuevo
        cand = np.where(pequenos)[0]
        w[cand[np.argmin(base[cand])]] = 0.0


def decodificar(cr: Cromosoma, inst: Instancia, umbral_cero: float = 0.15,
                tam_min: int = 1, detalle: bool = False) -> Programa:
    n, K = inst.n, inst.K
    p = inst.tiempo_unitario
    tr = inst.transporte

    # ---- tamaños de sublotes (consistentes en todas las etapas)
    tam = np.zeros((n, cr.claves.shape[1]), dtype=int)
    for j in range(n):
        tam[j, : inst.max_sublotes[j]] = tamanos_sublotes(
            cr.claves[j], int(inst.cantidad[j]), int(inst.max_sublotes[j]), umbral_cero, tam_min
        )

    # ---- lista de sublotes en el orden del cromosoma
    contador = np.zeros(n, dtype=int)
    sublotes = []  # (trabajo, índice sublote, tamaño)
    for j in cr.secuencia:
        r = contador[j]
        contador[j] += 1
        if tam[j, r] > 0:
            sublotes.append((j, r, tam[j, r]))
    S = len(sublotes)
    listo = np.zeros(S)          # instante en que el sublote puede empezar la etapa actual
    prioridad = np.arange(S)     # posición en el cromosoma (desempate)
    fin = np.zeros(S)
    operaciones = []

    for k in range(K):
        disp = inst.disponibilidad[k].astype(float).copy()
        vel = inst.velocidades[k]
        if k == 0:
            orden = prioridad
        else:
            orden = np.lexsort((prioridad, listo))   # FIFO por llegada
        for s in orden:
            j, r, q = sublotes[s]
            inicio = np.maximum(disp, listo[s])
            termino = inicio + q * p[j, k] / vel
            m = int(np.argmin(termino))
            disp[m] = termino[m]
            fin[s] = termino[m]
            if detalle:
                operaciones.append((j, r, q, k, m, float(inicio[m]), float(termino[m])))
        if k < K - 1:
            listo = fin + np.array([tr[j, k] for j, _, _ in sublotes])

    completacion = np.zeros(n)
    for s, (j, _, _) in enumerate(sublotes):
        completacion[j] = max(completacion[j], fin[s])
    tardanza = np.maximum(0.0, completacion - inst.fecha_entrega)

    return Programa(
        tardanza_total=float(tardanza.sum()),
        makespan=float(completacion.max()),
        completacion=completacion,
        tardanza=tardanza,
        tamanos=[tam[j][tam[j] > 0] for j in range(n)],
        operaciones=operaciones,
    )


def programa_a_tablas(prog: Programa, inst: Instancia) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convierte un programa decodificado (con detalle) en tablas legibles."""
    ops = pd.DataFrame(
        [
            {
                "trabajo": inst.trabajos[j],
                "sublote": r + 1,
                "tamano_sublote": q,
                "etapa": k + 1,
                "nombre_etapa": inst.nombres_etapas[k],
                "maquina": inst.nombres_maquinas[k][m],
                "inicio_h": round(ini, 4),
                "fin_h": round(fin, 4),
                "duracion_h": round(fin - ini, 4),
            }
            for j, r, q, k, m, ini, fin in prog.operaciones
        ]
    ).sort_values(["etapa", "maquina", "inicio_h"], ignore_index=True)

    trabajos = pd.DataFrame(
        {
            "trabajo": inst.trabajos,
            "cantidad": inst.cantidad,
            "num_sublotes": [len(t) for t in prog.tamanos],
            "tamanos_sublotes": [", ".join(map(str, t)) for t in prog.tamanos],
            "fecha_entrega_h": inst.fecha_entrega,
            "completacion_h": np.round(prog.completacion, 4),
            "tardanza_h": np.round(prog.tardanza, 4),
        }
    )
    return ops, trabajos
