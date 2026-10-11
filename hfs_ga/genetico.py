"""
Algoritmo genético para FH4, ((P4), (Q20), (P9), (P8)) | ls, trans, am | ΣTj.

Operadores:
* Selección por torneo.
* Cruce de la secuencia: POX (Precedence Operation Crossover), que preserva
  la cantidad de apariciones de cada trabajo.
* Cruce de las llaves de tamaño: uniforme por trabajo (se hereda el vector
  completo de llaves de cada trabajo de uno u otro padre).
* Mutación de la secuencia: intercambio (swap) o inserción.
* Mutación de las llaves: perturbación gaussiana y, con baja probabilidad,
  activar/desactivar un sublote.
* Elitismo y criterio de parada por generaciones o por estancamiento.
* La población inicial incluye individuos construidos con la regla EDD
  (fecha de entrega más temprana) además de individuos aleatorios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .datos import Instancia
from .decodificador import Cromosoma, Programa, decodificar


@dataclass
class ParametrosAG:
    tam_poblacion: int = 80
    generaciones: int = 300
    prob_cruce: float = 0.9
    prob_mutacion: float = 0.3
    elitismo: int = 2
    tam_torneo: int = 3
    max_sin_mejora: int = 100      # parada por estancamiento
    umbral_cero: float = 0.15      # llave < umbral => sublote anulado
    tam_min_sublote: int = 1       # tamaño mínimo de un sublote (unidades)
    semilla: int | None = 42
    verbose: bool = True


@dataclass
class ResultadoAG:
    mejor: Cromosoma
    programa: Programa
    historial_mejor: list[float] = field(default_factory=list)
    historial_promedio: list[float] = field(default_factory=list)
    generaciones: int = 0
    tiempo_s: float = 0.0


class AlgoritmoGenetico:
    def __init__(self, inst: Instancia, params: ParametrosAG | None = None):
        self.inst = inst
        self.p = params or ParametrosAG()
        self.rng = np.random.default_rng(self.p.semilla)
        self.s_max = int(inst.max_sublotes.max())
        self.base_secuencia = np.repeat(np.arange(inst.n), inst.max_sublotes)

    # ------------------------------------------------------------------ util
    def _evaluar(self, cr: Cromosoma) -> tuple[float, float]:
        prog = decodificar(cr, self.inst, self.p.umbral_cero, self.p.tam_min_sublote)
        # objetivo principal ΣTj; el makespan solo se usa como desempate
        return prog.tardanza_total, prog.makespan

    @staticmethod
    def _clave_orden(fit: tuple[float, float]) -> tuple[float, float]:
        return (round(fit[0], 6), fit[1])

    def _aleatorio(self) -> Cromosoma:
        sec = self.rng.permutation(self.base_secuencia)
        claves = self.rng.random((self.inst.n, self.s_max))
        return Cromosoma(sec, claves)

    def _edd(self, n_sublotes: int) -> Cromosoma:
        """Individuo heurístico: trabajos por fecha de entrega, sublotes iguales."""
        orden = np.argsort(self.inst.fecha_entrega, kind="stable")
        sec = np.concatenate([np.full(self.inst.max_sublotes[j], j) for j in orden])
        claves = np.zeros((self.inst.n, self.s_max))
        for j in range(self.inst.n):
            s = min(n_sublotes, self.inst.max_sublotes[j])
            claves[j, :s] = 1.0
        return Cromosoma(sec, claves)

    # ------------------------------------------------------------ operadores
    def _torneo(self, fits: list) -> int:
        cand = self.rng.choice(len(fits), size=self.p.tam_torneo, replace=False)
        return int(min(cand, key=lambda i: self._clave_orden(fits[i])))

    def _cruce(self, a: Cromosoma, b: Cromosoma) -> tuple[Cromosoma, Cromosoma]:
        n = self.inst.n
        conjunto = self.rng.random(n) < 0.5
        h1 = self._pox(a.secuencia, b.secuencia, conjunto)
        h2 = self._pox(b.secuencia, a.secuencia, conjunto)
        mascara = self.rng.random(n) < 0.5
        c1 = np.where(mascara[:, None], a.claves, b.claves)
        c2 = np.where(mascara[:, None], b.claves, a.claves)
        return Cromosoma(h1, c1), Cromosoma(h2, c2)

    @staticmethod
    def _pox(p1: np.ndarray, p2: np.ndarray, conjunto: np.ndarray) -> np.ndarray:
        hijo = p1.copy()
        fijo = conjunto[p1]
        hijo[~fijo] = p2[~conjunto[p2]]
        return hijo

    def _mutar(self, cr: Cromosoma) -> None:
        sec = cr.secuencia
        L = len(sec)
        if L > 1 and self.rng.random() < self.p.prob_mutacion:
            i, j = self.rng.choice(L, size=2, replace=False)
            if self.rng.random() < 0.5:
                sec[i], sec[j] = sec[j], sec[i]
            else:  # inserción
                g = sec[i]
                sec[:] = np.insert(np.delete(sec, i), j, g)
        if self.rng.random() < self.p.prob_mutacion:
            j = self.rng.integers(self.inst.n)
            s = self.inst.max_sublotes[j]
            cr.claves[j, :s] = np.clip(cr.claves[j, :s] + self.rng.normal(0, 0.2, s), 0, 1)
        if self.rng.random() < self.p.prob_mutacion * 0.3:
            j = self.rng.integers(self.inst.n)
            r = self.rng.integers(self.inst.max_sublotes[j])
            cr.claves[j, r] = 0.0 if cr.claves[j, r] >= self.p.umbral_cero else self.rng.uniform(0.5, 1)

    # ------------------------------------------------------------- ejecución
    def ejecutar(self) -> ResultadoAG:
        t0 = time.time()
        P = self.p.tam_poblacion
        pob = [self._edd(1), self._edd(self.s_max)]
        pob += [self._aleatorio() for _ in range(P - len(pob))]
        fits = [self._evaluar(c) for c in pob]

        hist_mejor, hist_prom = [], []
        mejor_val, sin_mejora, g = None, 0, 0
        for g in range(1, self.p.generaciones + 1):
            orden = sorted(range(P), key=lambda i: self._clave_orden(fits[i]))
            nueva = [pob[i].copia() for i in orden[: self.p.elitismo]]
            nueva_fit = [fits[i] for i in orden[: self.p.elitismo]]

            while len(nueva) < P:
                a = pob[self._torneo(fits)]
                b = pob[self._torneo(fits)]
                if self.rng.random() < self.p.prob_cruce:
                    h1, h2 = self._cruce(a, b)
                else:
                    h1, h2 = a.copia(), b.copia()
                for h in (h1, h2):
                    if len(nueva) < P:
                        self._mutar(h)
                        nueva.append(h)
                        nueva_fit.append(self._evaluar(h))
            pob, fits = nueva, nueva_fit

            mejor_i = min(range(P), key=lambda i: self._clave_orden(fits[i]))
            actual = fits[mejor_i]
            hist_mejor.append(actual[0])
            hist_prom.append(float(np.mean([f[0] for f in fits])))
            if mejor_val is None or self._clave_orden(actual) < self._clave_orden(mejor_val):
                mejor_val, sin_mejora = actual, 0
            else:
                sin_mejora += 1
            if self.p.verbose and (g == 1 or g % 10 == 0):
                print(f"Gen {g:4d} | ΣTj mejor = {actual[0]:10.3f} h | "
                      f"Cmax = {actual[1]:8.3f} h | ΣTj promedio = {hist_prom[-1]:10.3f} h")
            if actual[0] <= 1e-9 and sin_mejora >= 20:
                break  # tardanza cero y sin mejoras en makespan
            if sin_mejora >= self.p.max_sin_mejora:
                if self.p.verbose:
                    print(f"Parada por estancamiento ({self.p.max_sin_mejora} generaciones sin mejora).")
                break

        mejor_i = min(range(P), key=lambda i: self._clave_orden(fits[i]))
        mejor = pob[mejor_i]
        prog = decodificar(mejor, self.inst, self.p.umbral_cero, self.p.tam_min_sublote, detalle=True)
        return ResultadoAG(mejor, prog, hist_mejor, hist_prom, g, time.time() - t0)
