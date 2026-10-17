import random
from typing import Dict


class CromosomaBinario(object):
    def __init__(self, genes: int):
        self.n_genes: int = genes
        self.score: int = -1
        self.gen_width: int = 32 // genes
        self.mascaras: Dict = {}
        self.bits: int = 0x00000000
        max_value: int = 2 ** self.gen_width - 1
        for g, i in enumerate(reversed(range(0, 32, self.gen_width))):
            self.mascaras["p_" + str(g + 1)] = i
            self.mascaras["m_" + str(g + 1)] = max_value << i

    def llenar_random_data(self):
        max_value = 2 ** self.gen_width - 1
        for i in range(self.n_genes):
            valor = random.randint(0, max_value)
            self.set_bits(valor, i + 1)

    def registrar_mascaras(self, mascara: int, gen: int) -> None:
        self.mascaras["m_" + str(gen)] = mascara

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m_" + str(gen)]) >> self.mascaras["p_" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = ((self.bits & ~self.mascaras["m_" + str(gen)])
                     | (valor << self.mascaras["p_" + str(gen)]))
