import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from src.annealing import SimulatedAnnealing
from src.sat import SATDomain

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results/convergence_experiment'

# Configurações de experimento
COOLING_SCHEDULE = 1
SA_MAX_VALUES = (1, 5, 10)
EVAL_NUM = 100000

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(FORMULAS_DIR)
paths = [os.path.join(FORMULAS_DIR, f) for f in files]
instances = [SATDomain(p, flip_prob=0.01) for p in paths]
mean_std_dict = {}

fig, plot_axes = plt.subplots(1, 3, figsize=(20, 6))

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
                t0=200,
                t_final=1,
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
plt.savefig(os.path.join(RESULTS_DIR, 'box_plots.png'))
print(json.dumps(mean_std_dict, indent=4, ensure_ascii=False))