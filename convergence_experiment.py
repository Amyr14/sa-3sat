import os
import matplotlib.pyplot as plt
from src.annealing import SimulatedAnnealing, get_cooling_schedule_label
from src.sat import SATDomain

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results/convergence_experiment'

# Configurações de experimento
COOLING_SCHEDULE = 1

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(FORMULAS_DIR)
paths = [os.path.join(FORMULAS_DIR, f) for f in files]
instances = [SATDomain(p, flip_prob=0.01) for p in paths]

fig, plot_axes = plt.subplots(1, 3, figsize=(20, 6))

for instance, ax in zip(instances, plot_axes):
    algorithm = SimulatedAnnealing(
        cooling_schedule_i=COOLING_SCHEDULE,
        domain=instance
    )
    result = algorithm.run(
        t0=200,
        t_final=1,
        sa_max=3,
        eval_max=100000,
    )
    energies = result['energies']
    temperatures = result['temperatures']
    
    # Plotando gráfico de energia
    ax.set_title(instance.get_label())
    ax.plot(range(len(energies)), energies, 'b')
    ax.set_ylabel('Cláusulas Insatisfeitas', color='blue')
    ax.set_xlabel('Iteração')
    ax.tick_params(axis='y', labelcolor='blue')

    # Plotando gŕafico de temperatura
    ax_temp = ax.twinx()
    ax_temp.plot(range(len(temperatures)), temperatures, 'r--', label=get_cooling_schedule_label(COOLING_SCHEDULE))
    ax_temp.set_ylabel('Temperatura', color='red')
    ax_temp.tick_params(axis='y', labelcolor='red')

# plt.title('Gráficos de Convergência')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'convergencia.png'))