from abc import ABC, abstractmethod
from typing import List
from CromosomaBin import CromosomaBinario
import random


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaBinario], potencial: List[CromosomaBinario], maxpoblacion: int) -> None:
        pass


class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion: List[CromosomaBinario], potencial: List[CromosomaBinario], maxpoblacion: int) -> None:
        currpo = len(poblacion)
        ordpadres = sorted(potencial, key=lambda c: c.score, reverse=True)
        idpadres: int = 0
        while currpo < maxpoblacion:
            hijoA = CromosomaBinario(self.num_genes)
            hijoB = CromosomaBinario(self.num_genes)
            puntocruce = random.randint(1, self.num_genes - 1)
            padreA = ordpadres[idpadres % maxpoblacion]
            padreB = ordpadres[(idpadres + 1) % maxpoblacion]
            for i in range(1, self.num_genes):
                if i <= puntocruce:
                    hijoA.set_bits(padreA.get_bits(i), i)
                    hijoB.set_bits(padreB.get_bits(i), i)
                else:
                    hijoA.set_bits(padreB.get_bits(i), i)
                    hijoB.set_bits(padreA.get_bits(i), i)
            if currpo < maxpoblacion:
                poblacion.append(hijoA)
                currpo += 1
            if currpo < maxpoblacion:
                poblacion.append(hijoB)
                currpo += 1
            idpadres += 1
