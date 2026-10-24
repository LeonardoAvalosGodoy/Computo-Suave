import random
from abc import ABC, abstractmethod


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion):
        pass


# Cambia al azar el procesador de algunas tareas de los hijos.
class MutacionDecimal(Mutacion):
    def __init__(self, probabilidad=0.05, rng=random):
        if not 0 <= probabilidad <= 1:
            raise ValueError('La probabilidad debe estar entre 0 y 1.')
        self.prob = probabilidad
        self.rng = rng

    def mutacion_poblacion(self, poblacion):
        for cromosoma in poblacion:
            for gen in range(1, cromosoma.n_genes + 1):
                if self.rng.random() < self.prob:
                    actual = cromosoma.get_gen(gen)
                    nuevo = self.rng.choice([p for p in (1, 2, 3) if p != actual])
                    cromosoma.set_gen(nuevo, gen)
