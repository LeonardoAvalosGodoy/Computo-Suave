import random
import copy
from typing import Dict, List
from CromosomaBin import CromosomaBinario
from interfaz import Aptitud, Seleccion, Reproduccion, Mutacion


class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaBinario] = [CromosomaBinario(num_genes) for i in range(num_inv)]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar: Aptitud = hyperparams["Aptitud"]
        self.seleccion: Seleccion = hyperparams["Seleccion"]
        self.reproducion: Reproduccion = hyperparams["Reproduccion"]
        self.mutacion: Mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)
        for i in range(num_inv):
            self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: Dict):
        avg_tol: float = 99999999.99
        sum_score_prev: float = 0
        sum_score: float = 1
        generacion: int = 1
        matingpool = None
        generacion_inicial: List = copy.deepcopy(self.poblacion)
        generacion_nueva: List = None
        fail: bool = False
        fail_count: int = 0
        fail_score: float = 0.0
        while not fail:
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            sum_scores: float = sum([cromosoma.score for cromosoma in self.poblacion])
            sum_scores /= self.num_inv
            avg_tol: float = abs(sum_scores - sum_score_prev) / sum_score
            sum_score_prev = sum_scores
            if (generacion + 1) % 100 == 0:
                print(f"[{generacion:04d}] avgerr: {avg_tol:.5f} avgscore: {sum_scores:.5f}")
            if fail_score == avg_tol:
                fail_count += 1
            if fail_count >= 10:
                fail = True
                break
            if avg_tol < tolerancia:
                break
            fail_score = avg_tol
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=True)
            matingpool = self.seleccion.seleccion_poblacion(self.poblacion)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, matingpool, self.num_inv)
            self.mutacion.mutacion_poblacion(generacion_nueva)
            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)


def sum_pesos_test(pesos, cromosoma):
    s = 0
    for i in range(cromosoma.n_genes):
        if cromosoma.get_bits(i + 1) == 1:
            s += pesos[i]
    return s


if __name__ == "__main__":
    from interfaz import AptitudMochila, SeleccionTorneo, ReproduccionUnCruce, MutacionFlip

    variables = {}
    # Datos fijos para comparar los cambios de capacidad y de pesos.
    variables["pesos"] = [5, 6, 9, 9, 2, 4, 10, 10, 9, 7, 10, 9, 8, 10, 8, 4, 1, 10, 2, 2, 5, 2, 8, 1, 8, 6, 4, 7, 5, 6, 6, 7]
    variables["valores"] = [75, 91, 19, 53, 21, 81, 78, 47, 46, 68, 28, 92, 100, 83, 49, 13, 100, 57, 56, 69, 64, 21, 61, 84, 80, 73, 24, 64, 74, 86, 72, 60]
    variables["capacidad"] = 60  
    num_genes = 32
    cantind = 100
    gaparams = {}
    gaparams["Aptitud"] = AptitudMochila()
    gaparams["Seleccion"] = SeleccionTorneo(3)
    gaparams["Reproduccion"] = ReproduccionUnCruce(num_genes)
    gaparams["Mutacion"] = MutacionFlip(0.05)
    gaparams["por_remplazo"] = 0.5
    random.seed(0)  # Repite el mismo inicio para comparar las pruebas.
    print(f"Capacidad utilizada: {variables['capacidad']} kg", flush=True)
    algoritmoGA = GA(cantind, num_genes, gaparams)
    algoritmoGA.buscar_solucion(1e-4, variables)
    print(f"{'id':^6} {'Cromosoma':^32} Score / Peso / Estado")
    solucion = sorted(algoritmoGA.poblacion, key=lambda c: c.score, reverse=True)
    for i, cromosoma in enumerate(solucion):
        peso = sum_pesos_test(variables["pesos"], cromosoma)
        estado = "CUMPLE" if peso <= variables["capacidad"] else "SOBREPESO"
        print(f"[{i+1:^4d}] {cromosoma.bits:032b} : {cromosoma.score}->{peso:4d}kg {estado}")

    factibles = [c for c in solucion
                 if sum_pesos_test(variables["pesos"], c) <= variables["capacidad"]]
    print("\nRESULTADO PARA LA CAPACIDAD INDICADA")
    print(f"Capacidad: {variables['capacidad']} kg")
    if factibles:
        elegido = factibles[0]
        peso = sum_pesos_test(variables["pesos"], elegido)
        valor = sum(v for i, v in enumerate(variables["valores"])
                    if elegido.get_bits(i + 1) == 1)
        print(f"Cromosoma: {elegido.bits:032b}")
        print(f"Peso total: {peso} kg")
        print(f"Valor total: {valor}")
        print("Artículos:", [i+1 for i in range(num_genes) if elegido.get_bits(i+1)])
    else:
        print("La población final no contiene una combinación que cumpla la capacidad.")
