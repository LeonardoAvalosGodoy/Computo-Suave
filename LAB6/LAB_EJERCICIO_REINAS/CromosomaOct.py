import random
from typing import Dict


# Representa un tablero y guarda las filas de sus ocho reinas.
class CromosomaOctal(object):
    """Ocho genes octales: una columna por gen y una fila de 0 a 7."""

    def __init__(self, genes: int = 8):
        if genes != 8:
            raise ValueError("El problema utiliza exactamente ocho genes.")
        self.n_genes: int = genes
        self.gen_width: int = 3  
        self.score: int = 0
        self.bits: int = 0
        # Guarda los datos necesarios para localizar cada gen en el entero.
        self.mascaras: Dict = {}
        for gen in range(1, self.n_genes + 1):
            posicion = (self.n_genes - gen) * self.gen_width
            self.mascaras["p_" + str(gen)] = posicion
            self.mascaras["m_" + str(gen)] = 7 << posicion

    # Asigna una fila al azar a cada reina.
    def llenar_random_data(self):
        for gen in range(1, self.n_genes + 1):
            self.set_bits(random.randint(0, 7), gen)

    # Consulta la fila de la reina de la columna indicada.
    def get_bits(self, gen: int) -> int:
        if not 1 <= gen <= self.n_genes:
            raise IndexError("El gen debe estar entre 1 y 8.")
        return ((self.bits & self.mascaras["m_" + str(gen)])
                >> self.mascaras["p_" + str(gen)])

    # Cambia la fila de la reina de la columna indicada.
    def set_bits(self, valor: int, gen: int) -> None:
        if not 0 <= valor <= 7:
            raise ValueError("Un gen octal solo admite valores de 0 a 7.")
        self.get_bits(gen)
        self.bits = ((self.bits & ~self.mascaras["m_" + str(gen)])
                     | (valor << self.mascaras["p_" + str(gen)]))
