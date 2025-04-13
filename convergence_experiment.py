import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.annealing import SimulatedAnnealing, get_cooling_schedule_label
from src.sat import SATDomain

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results/convergence'

# Configurações de experimento
COOLING_SCHEDULE = 2
FLIP_FACTOR = 0.05
SA_MAX = 5
T0 = 200
T_FINAL = 1
NUM_EVAL = 100000

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(FORMULAS_DIR)
paths = [os.path.join(FORMULAS_DIR, f) for f in files]
instances = [SATDomain(p, flip_factor=FLIP_FACTOR) for p in paths]

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 4, figure=fig)

ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[:2, 2:])
ax3 = fig.add_subplot(gs[2:4, 1:3])  
plot_axes = [ax1, ax2, ax3]

for instance, ax in zip(instances, plot_axes):
    algorithm = SimulatedAnnealing(
        cooling_schedule_i=COOLING_SCHEDULE,
        domain=instance
    )
    result = algorithm.run(
        t0=T0,
        t_final=T_FINAL,
        sa_max=SA_MAX,
        eval_max=NUM_EVAL,
    )
    energies = result['energies']
    temperatures = result['temperatures']
    
    # Plot energy graph
    ax.set_title(instance.get_label())
    ax.plot(range(len(energies)), energies, 'b')
    ax.set_ylabel('Cláusulas Insatisfeitas', color='blue')
    ax.set_xlabel('Iteração')
    ax.tick_params(axis='y', labelcolor='blue')

    # Plot temperature graph
    ax_temp = ax.twinx()
    ax_temp.plot(range(len(temperatures)), temperatures, 'r--', label=get_cooling_schedule_label(COOLING_SCHEDULE))
    ax_temp.set_ylabel('Temperatura', color='red')
    ax_temp.tick_params(axis='y', labelcolor='red')

# Add legend to the last subplot (optional)
ax_temp.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'convergencia.png'))