# Flow-Shop-hibrido
Programación de la producción en un flow shop híbrido de 4 etapas con **algoritmo genético** y **reglas de despacho** (Python).

**FH4, ((P4), (Q20), (P9), (P8)) | ls | ΣTj** — Troquelado (4 P) → Ensamble (20 Q) → Costura (9 P) → Volteado (8 P).
Lot streaming con sublotes consistentes; se minimiza la tardanza total. Solo se consideran los tiempos de proceso
(transporte y demás tiempos se desprecian).

## Archivos

| Archivo | Contenido |
|---|---|
| `flow_shop_ga.py` | Código completo en un solo archivo (Colab o local) |
| `flow_shop_ga_colab.ipynb` | Cuaderno de Google Colab: subir base de datos → ejecutar → descargar |
| `datos/base_datos_flow_shop.xlsx` | Base de datos sintética: 20 trabajos, 4 etapas |
| `datos/instancia_pequena_5x3.xlsx` | Instancia de 5 trabajos y 3 etapas para validar en LEKIN |
| `docs/Guia_modelo_flow_shop.docx` | Guía de uso, significado de cada hoja/columna y código completo |

## Uso rápido

**Google Colab:** suba `flow_shop_ga_colab.ipynb` (Archivo → Subir cuaderno) y ejecute las celdas en orden.
O pegue `flow_shop_ga.py` en una celda y luego:

```python
RUTA = subir_base_datos()      # botón para elegir el Excel
resultados = ejecutar(RUTA)    # reglas de despacho + algoritmo genético
descargar_resultados()         # descarga resultados.zip
```

**Local:**

```bash
pip install numpy pandas openpyxl matplotlib
python flow_shop_ga.py --datos datos/base_datos_flow_shop.xlsx      # reglas + AG
python flow_shop_ga.py --datos datos/instancia_pequena_5x3.xlsx --solo-reglas
python flow_shop_ga.py --crear-ejemplos                             # regenerar las bases sintéticas
```

## Base de datos (Excel)

El modelo reconoce cada dato por el nombre de la hoja y del encabezado de columna; los valores se pueden cambiar libremente.

| Hoja | Una fila por | Columnas |
|---|---|---|
| `Etapas` | etapa | `etapa`, `nombre`, `tipo` (P/Q), `num_maquinas` |
| `Maquinas` | máquina | `etapa`, `maquina`, `velocidad` (1 en etapas P) |
| `Trabajos` | trabajo | `trabajo`, `demanda_pares`, `fecha_entrega_h`, `max_sublotes` (opcional) |
| `Tiempos` | trabajo | `trabajo`, `etapa_1` … `etapa_K` en min/docena |
| `Parametros` | parámetro | `parametro`, `valor` (unidades, lot streaming y parámetros del AG) |

La hoja `LEEME` del Excel y la guía en `docs/` describen cada columna en detalle.

## Reglas de despacho

FIFO, LIFO, SPT, LPT, EDD, MS (mínima holgura), CR (razón crítica), LWKR y MWKR, con programación no retrasada
etapa por etapa. Con `sublotes_reglas = 1` los trabajos se procesan como lotes completos, comparable con LEKIN.
Se reportan ΣTj, Cmax, trabajos tardíos, tardanza máxima y media, flujo medio y ΣCj. El Excel de resultados
incluye la hoja `Tiempos_lote_LEKIN` con los tiempos por lote en minutos para cargar la instancia en LEKIN.
