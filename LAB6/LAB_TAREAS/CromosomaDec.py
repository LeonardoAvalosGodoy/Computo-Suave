import random


# Guarda los seis procesadores asignados a las seis tareas.
class CromosomaDecimal(object):
    def __init__(self, genes=6):
        if genes != 6:
            raise ValueError('Se requieren seis genes.')
        self.n_genes = genes
        self.genes = [1] * genes
        self.cargas = [0, 0, 0]
        self.score = 0.0

    # Asigna un procesador al azar a cada tarea.
    def llenar_random_data(self, rng=random):
        for gen in range(1, self.n_genes + 1):
            self.set_gen(rng.randint(1, 3), gen)

    # Consulta el procesador de una tarea, numerada del 1 al 6.
    def get_gen(self, gen):
        if not 1 <= gen <= self.n_genes:
            raise IndexError('El gen debe estar entre 1 y 6.')
        return self.genes[gen - 1]

    # Cambia el procesador de la tarea indicada.
    def set_gen(self, valor, gen):
        if valor not in (1, 2, 3):
            raise ValueError('El procesador debe ser 1, 2 o 3.')
        self.get_gen(gen)
        self.genes[gen - 1] = valor
