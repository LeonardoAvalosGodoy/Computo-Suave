from abc import ABC, abstractmethod
from typing import List
from CromosomaBin import CromosomaBinario
import random


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaBinario]):
        pass


class MutacionFlip(Mutacion):
    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaBinario]):
        max_valor = 2 ** poblacion[0].gen_width - 1
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                prob_gen = random.random()
                if prob_gen < self.prob:
                    valor = crom.get_bits(i)
                    valor = valor ^ max_valor
                    crom.set_bits(valor, i)
