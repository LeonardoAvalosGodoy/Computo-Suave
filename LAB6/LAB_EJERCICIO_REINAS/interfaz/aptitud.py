from abc import ABC, abstractmethod
from typing import List
from CromosomaOct import CromosomaOctal


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal]):
        pass


# Califica cada tablero segun las parejas de reinas que se atacan.
class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal]):
        for cromosoma in poblacion:
            ataques = 0
            # Cada gen fija una columna; no hay dos reinas en una columna.
            for i in range(1, cromosoma.n_genes + 1):
                fila_i = cromosoma.get_bits(i)
                for j in range(i + 1, cromosoma.n_genes + 1):
                    fila_j = cromosoma.get_bits(j)
                    # Comprueba si las dos reinas comparten fila o diagonal.
                    misma_fila = fila_i == fila_j
                    misma_diagonal = abs(fila_i - fila_j) == abs(i - j)
                    if misma_fila or misma_diagonal:
                        ataques += 1
            cromosoma.score = 28 - ataques
