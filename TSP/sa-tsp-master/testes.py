import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from src.annealing import SimulatedAnnealing, get_cooling_schedule_label
from src.tsp import TSPDomain
import numpy

def read_values(path):
    cities = []
    with open(path, 'r') as tsp_file:

        header = tsp_file.readline()

        for line in tsp_file:
                if line == 'EOF':
                    break
                term = list(map(int, line.split()[1:3]))  #[0:3] para incluir NODE
                cities.append(term)

    
    cities = numpy.array(cities)
    #print(cities)
    #print(len(cities))
    return cities

def plot_cities(cities, solution):
    cities = numpy.array(cities)
    x = cities[:, 0]
    y = cities[:, 1]

    #plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='black', label='Cidades')

    # Numerar os pontos
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 1, yi + 1, str(i), fontsize=8)

    if solution is not None:
        ordered = cities[solution]

        # Desenhar as setas entre as cidades
        for i in range(len(ordered)):
            start = ordered[i]
            end = ordered[(i + 1) % len(ordered)]  # fecha o ciclo

            dx = end[0] - start[0]
            dy = end[1] - start[1]

            # distance = numpy.sqrt(dx**2 + dy**2)
            # head_w = distance * 0.10   # ou 0.02 dependendo do visual
            # head_l = distance * 0.10

            # plt.arrow(
            #     start[0], start[1], dx * 0.95, dy * 0.95 , #para impedir que a seta ultrapasse o limite do ponto
            #     head_width=head_w,
            #     head_length=head_l,
            #     fc='blue', ec='blue',
            #     length_includes_head=True
            # )


            # A seta é desenhada com uma escala menor para não passar do ponto
            plt.arrow(start[0], start[1], dx * 0.90 , dy * 0.90,
                        head_width=2, head_length=3, fc='blue', ec='blue', length_includes_head=True)

        # Vértice inicial em verde
        start = cities[solution[0]]
        plt.scatter(start[0], start[1], color='green', s=100, zorder=5, label="início")

        # Vértice final em vermelho
        end = cities[solution[-1]]
        plt.scatter(end[0], end[1], color='red', s=100, zorder=5, label="fim")

    plt.title(f'Mapa para {len(cities)} cidades')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


# Configurações de diretório
DISTANCES_DIR = './instances/distances'
RESULTS_DIR = './results/convergence'

# Configurações de experimento
COOLING_SCHEDULE = 1
SWAP_FACTOR = 5 #pode ser 1, 3 ou 5.
SA_MAX = 10
T0 = 100
T_FINAL = 0.001
NUM_EVAL = 10000

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(DISTANCES_DIR)
paths = [os.path.join(DISTANCES_DIR, f) for f in files]
instances = [TSPDomain(p, SWAP_factor=SWAP_FACTOR) for p in paths]

fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 4, figure=fig)

ax1 = fig.add_subplot(gs[:2, :2])
ax2 = fig.add_subplot(gs[:2, 2:])
plot_axes = [ax1, ax2]

best_solutions = []

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

    best_solution = result['best_ever']
    best_solutions.append((instance.get_label(), best_solution))
    
    energies = result['energies']
    temperatures = result['temperatures']
    
    # Plot energy graph
    ax.set_title(instance.get_label())
    ax.plot(range(len(energies)), energies, 'b')
    ax.set_ylabel('Distância Total', color='blue')
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
plt.show()
plt.savefig(os.path.join(RESULTS_DIR, 'convergencia.png'))


#Adicionando a melhor solução:

for label, vector_best_solution in best_solutions:
    print(f"\nMelhor solução para {label}:")
    print("Ordem das cidades visitadas:", vector_best_solution)
    print(f"Quantidade de cidades: {len(vector_best_solution)}")

    if len(vector_best_solution) == 51:
        path_cities = "instances/points/eil51-tsp.txt"
    else:
        path_cities = "instances/points/kroA100-tsp.txt"

    print(f"Arquivo de cidades usado: {path_cities}")
    cities = read_values(path_cities)
    plot_cities(cities, vector_best_solution)

