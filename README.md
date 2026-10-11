# Flow-Shop-hibrido
Construcción de código para programación de la producción mediante **algoritmo genético** (Python).

## Problema

**FH4, ((P4), (Q20), (P9), (P8)) | ls, trans, am | ΣTj**

| Campo | Significado |
|---|---|
| **α** = FH4, ((P4), (Q20), (P9), (P8)) | Flow shop híbrido de 4 etapas: 1. Troquelado (4 máquinas paralelas idénticas), 2. Ensamble (20 máquinas paralelas **uniformes**), 3. Costura (9 idénticas), 4. Volteado (8 idénticas). |
| **β** = ls | *Lot streaming* con **sublotes consistentes**: cada lote se divide en sublotes cuyo tamaño es el mismo en todas las etapas, y cada sublote pasa a la siguiente etapa sin esperar al resto del lote. |
| **β** = trans | Tiempos de transporte entre etapas consecutivas. |
| **β** = am | Cada máquina tiene un instante de disponibilidad inicial. |
| **γ** = ΣTj | Minimizar la tardanza total, Tj = max(0, Cj − dj), con fechas de entrega dj en horas. |

## Estructura del proyecto

```
├── main.py                 # Ejecución por línea de comandos
├── flow_shop_ga.ipynb      # Cuaderno (Google Colab / Jupyter) con botón para subir datos
├── generar_plantilla.py    # Crea la plantilla Excel de datos con un ejemplo
├── requirements.txt
├── datos/
│   ├── plantilla_datos.xlsx   # ← PLANTILLA: aquí van sus datos
│   └── ejemplo_csv/           # El mismo ejemplo en formato CSV
├── hfs_ga/
│   ├── datos.py            # Lectura y validación de los datos
│   ├── decodificador.py    # Cromosoma -> programa de producción
│   ├── genetico.py         # Algoritmo genético
│   └── reportes.py         # Excel de resultados, Gantt y convergencia
└── resultados/             # Se crea al ejecutar (resultados.xlsx, gantt.png, convergencia.png)
```

## Instalación

```bash
pip install -r requirements.txt
```

## Cómo subir sus datos al modelo

### 1. Use la plantilla

Abra `datos/plantilla_datos.xlsx` (si no existe, créela con `python generar_plantilla.py`).
Contiene una hoja **Instrucciones** y cuatro hojas de datos. Reemplace los valores de ejemplo
por los suyos **sin cambiar los nombres de las hojas ni de las columnas**. Todos los tiempos van en **horas**.

**Hoja `Trabajos`** – un renglón por lote

| trabajo | cantidad | fecha_entrega_h | max_sublotes |
|---|---|---|---|
| J01 | 580 | 85.8 | 5 |
| J02 | 410 | 62.8 | 5 |

* `cantidad`: unidades (p. ej. pares) del lote, entero.
* `fecha_entrega_h`: fecha de entrega en horas medidas desde el instante 0 (inicio del horizonte).
* `max_sublotes` (opcional): máximo número de sublotes; el algoritmo decide cuántos usar (entre 1 y ese valor) y de qué tamaño. Si se omite se usa 3 (o el valor de `--max-sublotes`).

**Hoja `Tiempos`** – tiempo de proceso **por unidad** en cada etapa

| trabajo | etapa_1 | etapa_2 | etapa_3 | etapa_4 |
|---|---|---|---|---|
| J01 | 0.0150 | 0.2200 | 0.0900 | 0.0750 |

* Si sus tiempos están en minutos por unidad, divida entre 60.
* El tiempo de un sublote de *q* unidades en una máquina es `q × tiempo_unitario / velocidad`.
* En la etapa 2 (máquinas uniformes) el valor es el tiempo estándar a velocidad 1.

**Hoja `Maquinas`** – un renglón por máquina (4 + 20 + 9 + 8 = 41 renglones)

| etapa | maquina | velocidad | disponibilidad_h |
|---|---|---|---|
| 1 | T01 | 1.0 | 0 |
| 2 | E01 | 1.15 | 6 |

* `velocidad`: 1 en las etapas de máquinas idénticas (1, 3 y 4); en la etapa 2 es el factor de velocidad de cada máquina (1.2 = 20 % más rápida que la estándar).
* `disponibilidad_h`: instante a partir del cual la máquina puede empezar a trabajar (am).

**Hoja `Transporte`**

| de_etapa | a_etapa | tiempo_h | trabajo |
|---|---|---|---|
| 1 | 2 | 0.5 | |
| 2 | 3 | 0.3 | |
| 3 | 4 | 0.4 | |

* Si `trabajo` está vacío, el tiempo aplica a todos los trabajos. Para un tiempo distinto en un trabajo concreto agregue otro renglón con su nombre en `trabajo`.

Guarde el archivo, por ejemplo como `datos/mis_datos.xlsx`.
También puede usar una carpeta con los CSV `trabajos.csv`, `tiempos.csv`, `maquinas.csv` y `transporte.csv` (mismas columnas; vea `datos/ejemplo_csv/`).

### 2a. Ejecutar desde la terminal

```bash
python main.py --datos datos/mis_datos.xlsx
# con parámetros propios del AG
python main.py --datos datos/mis_datos.xlsx --poblacion 100 --generaciones 500 --pm 0.3 --pc 0.9
```

Ver todas las opciones con `python main.py --help`.

### 2b. Ejecutar en Google Colab (subiendo el archivo desde su computador)

1. Abra `flow_shop_ga.ipynb` en Colab (*Archivo → Abrir cuaderno → GitHub* y pegue la URL del repositorio).
2. Ejecute las celdas en orden. En el paso **2. Subir sus datos** aparecerá el botón **"Elegir archivos"**: seleccione su Excel.
3. Al final se muestran la tabla de resultados, la convergencia, el diagrama de Gantt y se descarga `resultados.xlsx`.

En Jupyter local, cambie la variable `RUTA_DATOS` por la ruta de su archivo.

## Resultados

En la carpeta `resultados/`:

* `resultados.xlsx`: hojas **Resumen** (ΣTj, Cmax, trabajos tardíos…), **Trabajos** (sublotes, Cj, Tj por trabajo), **Programa** (cada sublote: etapa, máquina, inicio y fin), **Convergencia** y **Parametros_AG**.
* `gantt.png`: diagrama de Gantt por máquina (zonas grises rayadas = máquina no disponible).
* `convergencia.png`: evolución de la tardanza total por generación.

## Algoritmo genético

**Cromosoma de dos partes**

1. *Secuencia*: cada trabajo *j* aparece `max_sublotes_j` veces; la *r*-ésima aparición representa su sublote *r*. Define la prioridad en la etapa 1.
2. *Llaves de tamaño*: un número en [0, 1] por sublote. Se normalizan para repartir la cantidad del lote (enteros, método del mayor residuo). Una llave menor a 0.15 anula el sublote, de modo que **el número y el tamaño de los sublotes también se optimizan**. El tamaño es el mismo en las 4 etapas (sublotes consistentes).

**Decodificación**

* Etapa 1: sublotes en el orden de la secuencia.
* Etapas 2–4: sublotes en orden de llegada (fin en la etapa anterior + transporte), regla FIFO.
* Cada sublote se asigna a la máquina que lo termina primero (ECT) considerando su disponibilidad inicial y su velocidad.
* Cj = fin del último sublote del trabajo *j* en la etapa 4; objetivo ΣTj (el makespan solo desempata).

**Operadores**: población inicial aleatoria + individuos EDD; selección por torneo; cruce POX para la secuencia y uniforme por trabajo para las llaves; mutación por intercambio/inserción y perturbación gaussiana de llaves (incluida la activación/desactivación de sublotes); elitismo; parada por número de generaciones o por estancamiento.

| Parámetro (`main.py`) | Por defecto | Descripción |
|---|---|---|
| `--poblacion` | 80 | Tamaño de la población |
| `--generaciones` | 300 | Máximo de generaciones |
| `--pc` | 0.9 | Probabilidad de cruce |
| `--pm` | 0.3 | Probabilidad de mutación |
| `--elitismo` | 2 | Individuos que pasan intactos |
| `--torneo` | 3 | Tamaño del torneo |
| `--sin-mejora` | 100 | Generaciones sin mejora para detenerse |
| `--max-sublotes` | 3 | Sublotes máximos si no vienen en la hoja Trabajos |
| `--tam-min-sublote` | 1 | Tamaño mínimo de un sublote (unidades) |
| `--semilla` | 42 | Semilla aleatoria (reproducibilidad) |
