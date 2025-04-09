import os
import matplotlib.pyplot as plt
from src.annealing import SimulatedAnnealing, get_cooling_schedule_label
from src.sat import SATDomain

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results/example_experiment'

# Configurações de experimento
COOLING_SCHEDULE = 1

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(FORMULAS_DIR)
paths = [os.path.join(FORMULAS_DIR, f) for f in files]
instances = [SATDomain(p, flip_prob=0.01) for p in paths]

sa = SimulatedAnnealing(
    cooling_schedule_i=COOLING_SCHEDULE,
    domain=instances[0],
)

run1 = sa.run(
    t0=150,
    t_final=1,
    sa_max=5,
    eval_max=15000,
)
energies = run1['energies']
temperatures = run1['temperatures']

# Plotando gráfico de energia
_, ax_energy = plt.subplots()
ax_energy.plot(range(len(energies)), energies, 'b')
ax_energy.set_ylabel('Cláusulas Insatisfeitas', color='blue')
ax_energy.set_xlabel('Iteração')
ax_energy.tick_params(axis='y', labelcolor='blue')

# Plotando gŕafico de temperatura
ax_temp = ax_energy.twinx()
ax_temp.plot(range(len(temperatures)), temperatures, 'r--', label=get_cooling_schedule_label(COOLING_SCHEDULE))
ax_temp.set_ylabel('Temperatura', color='red')
ax_temp.tick_params(axis='y', labelcolor='red')

plt.title(instances[0].get_domain_label())
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'exemplo.png'))