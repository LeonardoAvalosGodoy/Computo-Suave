import random
from abc import ABC, abstractmethod
from CromosomaDec import CromosomaDecimal


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(self, poblacion, potencial, maxpoblacion):
        pass


# Mezcla dos padres mediante un punto de corte para crear dos hijos.
class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes=6, rng=random):
        self.num_genes = num_genes
        self.rng = rng

    def reproduccion_poblacion(self, poblacion, potencial, maxpoblacion):
        idpadres = 0
        while len(poblacion) < maxpoblacion:
            padre_a = potencial[idpadres % len(potencial)]
            padre_b = potencial[(idpadres + 1) % len(potencial)]
            corte = self.rng.randint(1, self.num_genes - 1)
            hijo_a = CromosomaDecimal(self.num_genes)
            hijo_b = CromosomaDecimal(self.num_genes)
            for gen in range(1, self.num_genes + 1):
                if gen <= corte:
                    hijo_a.set_gen(padre_a.get_gen(gen), gen)
                    hijo_b.set_gen(padre_b.get_gen(gen), gen)
                else:
                    hijo_a.set_gen(padre_b.get_gen(gen), gen)
                    hijo_b.set_gen(padre_a.get_gen(gen), gen)
            poblacion.append(hijo_a)
            if len(poblacion) < maxpoblacion:
                poblacion.append(hijo_b)
            idpadres += 2
