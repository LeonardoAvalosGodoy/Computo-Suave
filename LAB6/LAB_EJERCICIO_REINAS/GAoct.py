import random
import copy
from typing import Dict, List
from CromosomaOct import CromosomaOctal
from interfaz import Aptitud, Seleccion, Reproduccion, Mutacion


# Coordina la creacion, evaluacion y renovacion de los tableros.
class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.num_inv = num_inv
        self.hyperparams = hyperparams
        self.evaluar: Aptitud = hyperparams["Aptitud"]
        self.seleccion: Seleccion = hyperparams["Seleccion"]
        self.reproducion: Reproduccion = hyperparams["Reproduccion"]
        self.mutacion: Mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)
        if num_inv < 3 or not 1 <= self.num_remplazo < num_inv:
            raise ValueError("Población o proporción de conservación inválida.")
        # Crea la lista de tableros que se usaran para buscar la solucion.
        self.poblacion: List[CromosomaOctal] = [
            CromosomaOctal(num_genes) for _ in range(num_inv)]
        for cromosoma in self.poblacion:
            cromosoma.llenar_random_data()
        self.generacion = 0

    # Repite la busqueda hasta resolver el tablero o llegar al limite.
    def buscar_solucion(self, max_generaciones: int = 5000):
        if max_generaciones < 0:
            raise ValueError("El límite de generaciones no puede ser negativo.")
        mejor_encontrado = None
        mejor_score_anterior = -1
        sin_mejora = 0
        for generacion in range(max_generaciones + 1):
            # Califica los tableros y los ordena del mejor al peor.
            self.evaluar.evaluar_poblacion(self.poblacion)
            self.poblacion.sort(key=lambda c: c.score, reverse=True)
            self.generacion = generacion
            mejor = self.poblacion[0]
            if mejor_encontrado is None or mejor.score > mejor_encontrado.score:
                mejor_encontrado = copy.deepcopy(mejor)
            if mejor.score > mejor_score_anterior:
                mejor_score_anterior = mejor.score
                sin_mejora = 0
            else:
                sin_mejora += 1
            if generacion % 100 == 0:
                promedio = sum(c.score for c in self.poblacion) / self.num_inv
                print(f"[{generacion:04d}] mejor: {mejor.score}/28 promedio: {promedio:.2f}")
            # Con 28 puntos no hay parejas de reinas que se ataquen.
            if mejor.score == 28:
                return mejor
            if generacion == max_generaciones:
                self.poblacion[-1] = mejor_encontrado
                self.poblacion.sort(key=lambda c: c.score, reverse=True)
                return self.poblacion[0]

            # Renovar opciones si no hay progreso durante 100 generaciones.
            # El mejor queda archivado; la población nueva explora otras opciones.
            if sin_mejora >= 100:
                for cromosoma in self.poblacion:
                    cromosoma.llenar_random_data()
                mejor_score_anterior = -1
                sin_mejora = 0
                continue

            padres = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(
                generacion_nueva, padres, self.num_inv)
            hijos = generacion_nueva[self.num_remplazo:]
            self.mutacion.mutacion_poblacion(hijos)
            self.poblacion = generacion_nueva


# Configura el algoritmo, inicia la busqueda y muestra los resultados.
if __name__ == "__main__":
    from interfaz import AptitudReinas, SeleccionTorneo, ReproduccionUnCruce, MutacionOctal

    num_genes = 8
    cantind = 100
    gaparams = {}
    gaparams["Aptitud"] = AptitudReinas()
    gaparams["Seleccion"] = SeleccionTorneo(3)
    gaparams["Reproduccion"] = ReproduccionUnCruce(num_genes)
    gaparams["Mutacion"] = MutacionOctal(0.05)
    gaparams["por_remplazo"] = 0.5
    algoritmoGA = GA(cantind, num_genes, gaparams)
    mejor = algoritmoGA.buscar_solucion(max_generaciones=5000)
    if mejor.score == 28:
        print(f"Solucion encontrada en la generacion {algoritmoGA.generacion}.")
    else:
        print("Se alcanzo el limite sin encontrar una solucion de 28 puntos.")
    print(f"{'id':^6} {'Cromosoma':^10} Score")
    for i, cromosoma in enumerate(algoritmoGA.poblacion):
        print(f"[{i+1:^4d}] {cromosoma.bits:08o} : {cromosoma.score}")
