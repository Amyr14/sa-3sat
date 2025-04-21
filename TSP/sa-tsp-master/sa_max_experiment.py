import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import json
from src.annealing import SimulatedAnnealing
from src.tsp import TSPDomain

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results/sa_max'

# Configurações de experimento
COOLING_SCHEDULE = 1
SA_MAX_VALUES = (1, 5, 10)
EVAL_NUM = 100000
T0 = 100
T_FINAL = 0.001
FLIP_FACTOR = 0.01

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(FORMULAS_DIR)
paths = [os.path.join(FORMULAS_DIR, f) for f in files]
instances = [TSPDomain(p, flip_factor=FLIP_FACTOR) for p in paths]
mean_std_dict = {}

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 4, figure=fig)

ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[:2, 2:])
ax3 = fig.add_subplot(gs[2:4, 1:3])  
plot_axes = [ax1, ax2, ax3]

for instance, ax in zip(instances, plot_axes):
    print(instance.get_label())
    algorithm = SimulatedAnnealing(
            cooling_schedule_i=COOLING_SCHEDULE,
            domain=instance
        )
    
    # Resultados referentes a cada SaMax
    results_dict = {sa_max: [] for sa_max in SA_MAX_VALUES}
    
    for sa_max in SA_MAX_VALUES:
        print(f'SaMax: {sa_max}')
        for i in range(30):
            print(f'Iteração {i}')
            result = algorithm.run(
                t0=T0,
                t_final=T_FINAL,
                sa_max=sa_max,
                eval_max=EVAL_NUM,
            )
            best_ever = result['best_ever_energy']
            results_dict.get(sa_max).append(best_ever)
    
    results_df = pd.DataFrame(results_dict)
    mean_std_dict.update({instance.get_label(): [(sa_max, np.mean(results_dict[sa_max]).item(), np.std(results_dict[sa_max]).item()) for sa_max in SA_MAX_VALUES]})
    ax.set_title(instance.get_label())
    ax.set_ylabel('Cláusulas Insatisfeitas')
    ax.set_xlabel('SaMax')
    results_df.plot(kind='box', ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'sa_max_box_plots.png'))

with open(os.path.join(RESULTS_DIR, 'mean_std.json'), mode='w') as fp:
    mean_std_json = json.dumps(mean_std_dict, indent=4, ensure_ascii=False)
    fp.write(mean_std_json)