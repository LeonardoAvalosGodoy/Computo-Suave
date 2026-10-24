import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaOct import CromosomaOctal


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass


# Cambia al azar algunas filas de las reinas de los hijos.
class MutacionOctal(Mutacion):
    def __init__(self, probabilidad: float = 0.05):
        if not 0 <= probabilidad <= 1:
            raise ValueError("La probabilidad debe estar entre 0 y 1.")
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion):
        for cromosoma in poblacion:
            for gen in range(1, cromosoma.n_genes + 1):
                # Decide si cambia este gen; la configuracion usa un 5 %.
                if random.random() < self.prob:
                    anterior = cromosoma.get_bits(gen)
                    # Desplazar entre 1 y 7 asegura otra fila válida.
                    nueva_fila = (anterior + random.randint(1, 7)) % 8
                    cromosoma.set_bits(nueva_fila, gen)
