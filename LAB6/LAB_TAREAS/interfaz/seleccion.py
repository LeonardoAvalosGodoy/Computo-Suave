import random
from abc import ABC, abstractmethod


class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion):
        pass


# Elige padres con probabilidades proporcionales a su fitness.
class SeleccionRuleta(Seleccion):
    def __init__(self, rng=random):
        self.rng = rng

    def seleccion_poblacion(self, poblacion):
        return self.rng.choices(poblacion, weights=[c.score for c in poblacion],
                                k=len(poblacion))
