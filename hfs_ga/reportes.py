"""Exportación de resultados: Excel, diagrama de Gantt y curva de convergencia."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .datos import Instancia
from .decodificador import programa_a_tablas
from .genetico import ParametrosAG, ResultadoAG


def exportar_excel(res: ResultadoAG, inst: Instancia, params: ParametrosAG, ruta: str) -> None:
    ops, trab = programa_a_tablas(res.programa, inst)
    resumen = pd.DataFrame(
        {
            "indicador": [
                "Tardanza total ΣTj (h)", "Makespan Cmax (h)", "Trabajos tardíos",
                "Tardanza máxima (h)", "Generaciones ejecutadas", "Tiempo de cómputo (s)",
            ],
            "valor": [
                round(res.programa.tardanza_total, 4), round(res.programa.makespan, 4),
                int((res.programa.tardanza > 1e-9).sum()), round(res.programa.tardanza.max(), 4),
                res.generaciones, round(res.tiempo_s, 2),
            ],
        }
    )
    param = pd.DataFrame({"parametro": list(vars(params)), "valor": [str(v) for v in vars(params).values()]})
    conv = pd.DataFrame(
        {
            "generacion": np.arange(1, len(res.historial_mejor) + 1),
            "mejor_tardanza_total_h": res.historial_mejor,
            "promedio_tardanza_total_h": res.historial_promedio,
        }
    )
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        resumen.to_excel(w, sheet_name="Resumen", index=False)
        trab.to_excel(w, sheet_name="Trabajos", index=False)
        ops.to_excel(w, sheet_name="Programa", index=False)
        conv.to_excel(w, sheet_name="Convergencia", index=False)
        param.to_excel(w, sheet_name="Parametros_AG", index=False)


def grafico_convergencia(res: ResultadoAG, ruta: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    g = np.arange(1, len(res.historial_mejor) + 1)
    ax.plot(g, res.historial_mejor, label="Mejor individuo", color="#1f77b4", lw=2)
    ax.plot(g, res.historial_promedio, label="Promedio población", color="#ff7f0e", lw=1, alpha=0.7)
    ax.set_xlabel("Generación")
    ax.set_ylabel("Tardanza total ΣTj (h)")
    ax.set_title("Convergencia del algoritmo genético")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def diagrama_gantt(res: ResultadoAG, inst: Instancia, ruta: str) -> None:
    ops, _ = programa_a_tablas(res.programa, inst)
    cmap = plt.get_cmap("tab20")
    colores = {t: cmap(i % 20) for i, t in enumerate(inst.trabajos)}

    filas = []  # (etiqueta, etapa, maquina)
    for k in range(inst.K):
        for nombre in inst.nombres_maquinas[k]:
            filas.append((f"E{k + 1}-{nombre}", k + 1, nombre))
    y = {(e, m): i for i, (_, e, m) in enumerate(filas)}

    fig, ax = plt.subplots(figsize=(16, max(6, 0.28 * len(filas))))
    for _, o in ops.iterrows():
        yi = y[(o["etapa"], o["maquina"])]
        ax.barh(yi, o["duracion_h"], left=o["inicio_h"], height=0.8,
                color=colores[o["trabajo"]], edgecolor="black", linewidth=0.3)
        if o["duracion_h"] > res.programa.makespan * 0.025:
            ax.text(o["inicio_h"] + o["duracion_h"] / 2, yi, f"{o['trabajo']}.{o['sublote']}",
                    ha="center", va="center", fontsize=5.5)
    # disponibilidad inicial de las máquinas (am)
    for k in range(inst.K):
        for m, nombre in enumerate(inst.nombres_maquinas[k]):
            d = inst.disponibilidad[k][m]
            if d > 0:
                ax.barh(y[(k + 1, nombre)], d, left=0, height=0.8, color="lightgrey", hatch="///",
                        edgecolor="grey", linewidth=0.3)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([f[0] for f in filas], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Tiempo (h)")
    ax.set_title(f"Diagrama de Gantt — ΣTj = {res.programa.tardanza_total:.2f} h, "
                 f"Cmax = {res.programa.makespan:.2f} h  (gris rayado = no disponible)")
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colores[t]) for t in inst.trabajos]
    ax.legend(handles, inst.trabajos, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7,
              title="Trabajo")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def guardar_todo(res: ResultadoAG, inst: Instancia, params: ParametrosAG, carpeta: str) -> dict[str, str]:
    os.makedirs(carpeta, exist_ok=True)
    rutas = {
        "excel": os.path.join(carpeta, "resultados.xlsx"),
        "gantt": os.path.join(carpeta, "gantt.png"),
        "convergencia": os.path.join(carpeta, "convergencia.png"),
    }
    exportar_excel(res, inst, params, rutas["excel"])
    diagrama_gantt(res, inst, rutas["gantt"])
    grafico_convergencia(res, rutas["convergencia"])
    return rutas
