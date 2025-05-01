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
RESULTS_DIR = './results/results_tsp/swaps'

# Configurações de experimento
COOLING_SCHEDULE = 2
SA_MAX = 10  # Agora usando SA_MAX fixo como 10
EVAL_NUM = 100000
T0 = 100
T_FINAL = 0.001
SWAP_FACTORS = [1, 3, 5]  # Analisando diferentes valores de SWAP_FACTOR

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(DISTANCES_DIR)
paths = [os.path.join(DISTANCES_DIR, f) for f in files]

paths.remove("./instances/distances\\kroA100-tsp_matrix.txt")
#paths.remove("./instances/distances\\eil51-tsp_matrix.txt")
for path in paths:
    print(path)

instances = [TSPDomain(p, SWAP_factor=1) for p in paths]  # Atribuindo SWAP_factor=1 inicialmente para instâncias
mean_std_dict = {}

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(4, 4, figure=fig)

ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[:2, 2:])
ax3 = fig.add_subplot(gs[2:4, 1:3])  
plot_axes = [ax1, ax2, ax3]

# Agora, iteramos sobre os diferentes valores de SWAP_FACTOR
for instance, ax in zip(instances, plot_axes):
    print(instance.get_label())
    
    # Resultados referentes a cada SWAP_FACTOR
    results_dict = {swap_factor: [] for swap_factor in SWAP_FACTORS}
    
    for swap_factor in SWAP_FACTORS:
        print(f'SWAP_FACTOR: {swap_factor}')
        
        # Alterando o SWAP_factor para cada instância
        #instance.set_swap_factor(swap_factor)  # Atualizando o SWAP_factor da instância
        instance.SWAP_factor = swap_factor
        
        # Executando o algoritmo com SA_MAX fixo em 10
        algorithm = SimulatedAnnealing(
                cooling_schedule_i=COOLING_SCHEDULE,
                domain=instance
            )
        
        for i in range(10):
            print(f'Iteração {i}')
            result = algorithm.run(
                t0=T0,
                t_final=T_FINAL,
                sa_max=SA_MAX,  # Usando SA_MAX fixo
                eval_max=EVAL_NUM,
            )
            best_ever = result['best_ever_energy']
            results_dict.get(swap_factor).append(best_ever)
    
    # Organizando os resultados em um DataFrame
    results_df = pd.DataFrame(results_dict)
    mean_std_dict.update({instance.get_label(): [(swap_factor, np.mean(results_dict[swap_factor]).item(), np.std(results_dict[swap_factor]).item()) for swap_factor in SWAP_FACTORS]})
    
    # Plotando os resultados
    ax.set_title(instance.get_label())
    ax.set_ylabel('Distância Total')
    ax.set_xlabel('SWAP_FACTOR')
    results_df.plot(kind='box', ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'swap_factor_box_plots.png'))

# Salvando as métricas médias e desvio padrão em um arquivo JSON
with open(os.path.join(RESULTS_DIR, 'mean_std.json'), mode='w') as fp:
    mean_std_json = json.dumps(mean_std_dict, indent=4, ensure_ascii=False)
    fp.write(mean_std_json)
