import os
import numpy
import matplotlib.pyplot as plt
from typing import override
from src.annealing import SimulatedAnnealing, Domain, FastCooler, History
from src.cnf import CNF

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results'

files = os.listdir(FORMULAS_DIR)
instances = [CNF(os.path.join(FORMULAS_DIR, f)) for f in files]

class SATDomain(Domain):
    def __init__(self, instance: CNF, flip_prob):
        self.flip_prob = flip_prob
        self.instance = instance
    
    @override
    def get_neighbour(self, current):
        flip_mask = numpy.random.random(current.size) < self.flip_prob
        return numpy.bitwise_xor(current, flip_mask)
    
    @override
    def random_value(self):
        return numpy.random.rand(self.instance.num_vars) > 0.5
    
    @override    
    def cost(self, value):
        _, num_unsatisfied = self.instance.get_valoration_metrics(value)
        return num_unsatisfied
    
# Instance 1
domain_1 = SATDomain(instances[1], flip_prob=0.05)
cooler_1 = FastCooler(t0=250, t_final=1, i_max=7000)

# for max_i in :
sa = SimulatedAnnealing(
    t0=250,
    max_i=7000,
    domain=domain_1,
    cooler=cooler_1,
    mode='minimize',
    goal=0,
    epsilon=0
)

history_1 = sa.run()

# Plotting
ax = history_1.graph('Cláusulas Insatisfeitas')
cooler_1.graph(ax=ax.twinx(), num_samples=history_1.get_num_iterations())
plt.legend()
plt.show()

# Create a figure
# plt.figure(figsize=(10, 6))