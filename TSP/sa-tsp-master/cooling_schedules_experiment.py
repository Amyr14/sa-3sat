import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import numpy as np
import json
from src.annealing import SimulatedAnnealing
from src.tsp import TSPDomain

# Configurações de diretório
DISTANCES_DIR = './instances/distances'
RESULTS_DIR = './results/results_tsp/cooling_schedules'

# Configurações de experimento
COOLING_SCHEDULES = [1, 2, 3]
EVAL_NUM = 100000
SA_MAX = 1 # 1, 5 ou 10
T0 = 8000
T_FINAL = 0.001
SWAP_FACTOR = 1 # 1, 3 ou 5

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(DISTANCES_DIR)
paths = [os.path.join(DISTANCES_DIR, f) for f in files]

#Usado para rodar apenas um de cada vez, (por conta da necessidade de ser usar diferenes parâmetros)
#paths.remove("./instances/distances\\kroA100-tsp_matrix.txt")
paths.remove("./instances/distances\\eil51-tsp_matrix.txt")
for path in paths:
    print(path)

instances = [TSPDomain(p, SWAP_factor=SWAP_FACTOR) for p in paths]
mean_std_dict = {}

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 4, figure=fig)

ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[:2, 2:])
ax3 = fig.add_subplot(gs[2:4, 1:3])  
plot_axes = [ax1, ax2, ax3]


for instance, ax in zip(instances, plot_axes):
    print(instance.get_label())
    results_dict = {cooling: [] for cooling in COOLING_SCHEDULES}
    
    for cooling in COOLING_SCHEDULES:
        algorithm = SimulatedAnnealing(
                cooling_schedule_i=cooling,
                domain=instance
            )
    
        for i in range(30):
            print(f'Iteração {i}')
            result = algorithm.run(
                t0=T0,
                t_final=T_FINAL,
                sa_max=SA_MAX,
                eval_max=EVAL_NUM,
            )
            best_ever = result['best_ever_energy']
            results_dict.get(cooling).append(best_ever)
    
    results_df = pd.DataFrame(results_dict)
    mean_std_dict.update({instance.get_label(): [(cooling, np.mean(results_dict[cooling]).item(), np.std(results_dict[cooling]).item()) for cooling in COOLING_SCHEDULES]})
    ax.set_title(instance.get_label())
    ax.set_ylabel('Distância Total')
    ax.set_xlabel('Rotina de Resfriamento')
    ax.tick_params(axis='y')
    results_df.plot(kind='box', ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cooling_box_plots.png'))

with open(os.path.join(RESULTS_DIR, 'mean_std.json'), mode='w') as fp:
    mean_std_json = json.dumps(mean_std_dict, indent=4, ensure_ascii=False)
    fp.write(mean_std_json)