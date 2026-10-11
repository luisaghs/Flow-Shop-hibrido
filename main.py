"""
Programación de producción en un Flow Shop híbrido de 4 etapas con algoritmo genético.

    FH4, ((P4), (Q20), (P9), (P8)) | ls, trans, am | ΣTj

Uso:
    python main.py --datos datos/plantilla_datos.xlsx
    python main.py --datos datos/mis_datos.xlsx --poblacion 100 --generaciones 500
    python main.py --datos datos/ejemplo_csv            # carpeta con CSV

Los resultados (Excel, Gantt y convergencia) se guardan en la carpeta --salida.
"""

import argparse
import os

from hfs_ga import AlgoritmoGenetico, ParametrosAG, cargar_datos, guardar_todo, programa_a_tablas


def main() -> None:
    d = ParametrosAG()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default="datos/plantilla_datos.xlsx",
                    help="Archivo Excel o carpeta CSV con los datos (por defecto la plantilla).")
    ap.add_argument("--salida", default="resultados", help="Carpeta de resultados.")
    ap.add_argument("--poblacion", type=int, default=d.tam_poblacion)
    ap.add_argument("--generaciones", type=int, default=d.generaciones)
    ap.add_argument("--pc", type=float, default=d.prob_cruce, help="Probabilidad de cruce.")
    ap.add_argument("--pm", type=float, default=d.prob_mutacion, help="Probabilidad de mutación.")
    ap.add_argument("--elitismo", type=int, default=d.elitismo)
    ap.add_argument("--torneo", type=int, default=d.tam_torneo)
    ap.add_argument("--sin-mejora", type=int, default=d.max_sin_mejora,
                    help="Generaciones sin mejora antes de detenerse.")
    ap.add_argument("--max-sublotes", type=int, default=3,
                    help="Máximo de sublotes por trabajo si no se indica en la hoja Trabajos.")
    ap.add_argument("--tam-min-sublote", type=int, default=d.tam_min_sublote,
                    help="Tamaño mínimo (unidades) de un sublote.")
    ap.add_argument("--semilla", type=int, default=d.semilla)
    a = ap.parse_args()

    if not os.path.exists(a.datos):
        raise SystemExit(
            f"No existe '{a.datos}'. Genere la plantilla con 'python generar_plantilla.py' "
            "o indique la ruta de su archivo con --datos."
        )

    inst = cargar_datos(a.datos, max_sublotes_defecto=a.max_sublotes)
    print("=" * 78)
    print("FH4, ((P4), (Q20), (P9), (P8)) | ls, trans, am | ΣTj")
    print("=" * 78)
    print(inst.resumen())
    print("-" * 78)

    params = ParametrosAG(
        tam_poblacion=a.poblacion, generaciones=a.generaciones, prob_cruce=a.pc,
        prob_mutacion=a.pm, elitismo=a.elitismo, tam_torneo=a.torneo,
        max_sin_mejora=a.sin_mejora, tam_min_sublote=a.tam_min_sublote, semilla=a.semilla,
    )
    res = AlgoritmoGenetico(inst, params).ejecutar()

    _, trab = programa_a_tablas(res.programa, inst)
    print("-" * 78)
    print(trab.to_string(index=False))
    print("-" * 78)
    print(f"Tardanza total ΣTj = {res.programa.tardanza_total:.3f} h")
    print(f"Makespan Cmax      = {res.programa.makespan:.3f} h")
    print(f"Generaciones: {res.generaciones}   Tiempo: {res.tiempo_s:.1f} s")

    rutas = guardar_todo(res, inst, params, a.salida)
    print("Resultados guardados en:")
    for nombre, ruta in rutas.items():
        print(f"  - {nombre}: {ruta}")


if __name__ == "__main__":
    main()
