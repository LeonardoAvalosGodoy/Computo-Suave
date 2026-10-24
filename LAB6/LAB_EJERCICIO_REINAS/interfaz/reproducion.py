import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaOct import CromosomaOctal


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion: List[CromosomaOctal],
                              potencial: List[CromosomaOctal], maxpoblacion: int):
        pass


# Crea hijos mezclando los genes de dos padres en un punto de corte.
class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int = 8):
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion, potencial, maxpoblacion):
        # Ordena los padres del mejor al peor.
        padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        idpadres = 0
        while len(poblacion) < maxpoblacion:
            padre_a = padres[idpadres % len(padres)]
            padre_b = padres[(idpadres + 1) % len(padres)]
            # Elige donde se separan las partes que aporta cada padre.
            corte = random.randint(1, self.num_genes - 1)
            hijo_a = CromosomaOctal(self.num_genes)
            hijo_b = CromosomaOctal(self.num_genes)
            # Se copian los ocho genes, incluido el último.
            for gen in range(1, self.num_genes + 1):
                if gen <= corte:
                    hijo_a.set_bits(padre_a.get_bits(gen), gen)
                    hijo_b.set_bits(padre_b.get_bits(gen), gen)
                else:
                    hijo_a.set_bits(padre_b.get_bits(gen), gen)
                    hijo_b.set_bits(padre_a.get_bits(gen), gen)
            # Agrega los hijos a la lista de la nueva poblacion.
            poblacion.append(hijo_a)
            if len(poblacion) < maxpoblacion:
                poblacion.append(hijo_b)
            idpadres += 1
