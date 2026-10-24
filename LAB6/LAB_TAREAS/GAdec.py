import random
import copy
from typing import Dict, List
from CromosomaDec import CromosomaDecimal
from interfaz import Aptitud, Seleccion, Reproduccion, Mutacion

TIEMPOS = [8, 5, 7, 3, 6, 4]


# Coordina las etapas del algoritmo, siguiendo la plantilla de las reinas.
class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict, rng=random):
        self.num_inv = num_inv
        self.hyperparams = hyperparams
        self.evaluar: Aptitud = hyperparams['Aptitud']
        self.seleccion: Seleccion = hyperparams['Seleccion']
        self.reproducion: Reproduccion = hyperparams['Reproduccion']
        self.mutacion: Mutacion = hyperparams['Mutacion']
        self.num_remplazo = int(hyperparams['por_remplazo'] * num_inv)
        if num_inv < 3 or not 1 <= self.num_remplazo < num_inv:
            raise ValueError('Poblacion o proporcion de conservacion invalida.')
        # Lista de asignaciones candidatas.
        self.poblacion: List[CromosomaDecimal] = [
            CromosomaDecimal(num_genes) for _ in range(num_inv)]
        for cromosoma in self.poblacion:
            cromosoma.llenar_random_data(rng)
        self.generacion = 0

    def buscar_solucion(self, max_generaciones=500):
        if max_generaciones < 0:
            raise ValueError('El limite de generaciones no puede ser negativo.')
        tiempos = self.evaluar.tiempos
        # Alcanzar este tiempo demuestra que la asignacion es optima.
        limite = max(max(tiempos), (sum(tiempos) + 2) // 3)
        for generacion in range(max_generaciones + 1):
            self.evaluar.evaluar_poblacion(self.poblacion)
            self.poblacion.sort(key=lambda c: c.score, reverse=True)
            self.generacion = generacion
            mejor = self.poblacion[0]
            if max(mejor.cargas) == limite or generacion == max_generaciones:
                return mejor
            padres = self.seleccion.seleccion_poblacion(self.poblacion)
            # Copia los mejores y completa la siguiente generacion con hijos.
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(
                generacion_nueva, padres, self.num_inv)
            hijos = generacion_nueva[self.num_remplazo:]
            self.mutacion.mutacion_poblacion(hijos)
            self.poblacion = generacion_nueva


# Configura las mismas etapas del taller con las reglas de este problema.
def crear_algoritmo(semilla=18):
    from interfaz import (AptitudProcesadores, SeleccionRuleta,
                          ReproduccionUnCruce, MutacionDecimal)
    rng = random.Random(semilla)
    gaparams = {}
    gaparams['Aptitud'] = AptitudProcesadores(TIEMPOS)
    gaparams['Seleccion'] = SeleccionRuleta(rng)
    gaparams['Reproduccion'] = ReproduccionUnCruce(6, rng)
    gaparams['Mutacion'] = MutacionDecimal(0.05, rng)
    gaparams['por_remplazo'] = 0.5
    return GA(100, 6, gaparams, rng)


# Muestra las tareas asignadas y la suma de tiempos de cada procesador.
def mostrar_resultado(mejor, generacion):
    print('\nTIEMPOS DE LAS TAREAS')
    for i, tiempo in enumerate(TIEMPOS, start=1):
        print(f'T{i}: {tiempo}')
    print('\nCromosoma encontrado:', ''.join(map(str, mejor.genes)))
    for procesador in (1, 2, 3):
        indices = [i for i, p in enumerate(mejor.genes) if p == procesador]
        tareas = ', '.join(f'T{i + 1}' for i in indices) or 'Sin tareas'
        tiempo = mejor.cargas[procesador - 1]
        print(f'P{procesador}: {tareas} - Tiempo: {tiempo}')


if __name__ == '__main__':
    algoritmoGA = crear_algoritmo()
    mejor = algoritmoGA.buscar_solucion(max_generaciones=500)
    mostrar_resultado(mejor, algoritmoGA.generacion)
