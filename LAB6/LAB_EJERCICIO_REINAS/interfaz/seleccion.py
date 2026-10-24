import random
from abc import ABC, abstractmethod
from typing import List
from CromosomaOct import CromosomaOctal


class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass


# Elige posibles padres mediante torneos; se configura con tres participantes.
class SeleccionTorneo(Seleccion):
    def __init__(self, participantes: int = 3):
        self.participantes = participantes

    def seleccion_poblacion(self, poblacion: List[CromosomaOctal]):
        # Lista donde se guardan los ganadores de los torneos.
        ganadores = []
        for _ in range(len(poblacion)):
            # Toma candidatos al azar y elige al de mayor puntuacion.
            competidores = random.sample(poblacion, self.participantes)
            ganador = max(competidores, key=lambda c: c.score)
            ganadores.append(ganador)
        return ganadores
