from abc import ABC, abstractmethod


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion):
        pass


# Suma los tiempos por procesador y califica cada asignacion.
class AptitudProcesadores(Aptitud):
    def __init__(self, tiempos):
        if len(tiempos) != 6 or any(type(t) is not int or t <= 0 for t in tiempos):
            raise ValueError('Se requieren seis tiempos enteros positivos.')
        self.tiempos = list(tiempos)

    def evaluar_poblacion(self, poblacion):
        for cromosoma in poblacion:
            cromosoma.cargas = [0, 0, 0]
            for gen, tiempo in enumerate(self.tiempos, start=1):
                procesador = cromosoma.get_gen(gen)
                cromosoma.cargas[procesador - 1] += tiempo
            cromosoma.score = 1 / (1 + max(cromosoma.cargas))
