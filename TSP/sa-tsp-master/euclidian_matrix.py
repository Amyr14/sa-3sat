import numpy
import matplotlib.pyplot as plt

def read_values(path):
    print("Criando a Matriz de Distância Euclidiana")
    
    cities = []
    with open(path, 'r') as tsp_file:

        header = tsp_file.readline()

        for line in tsp_file:
                if line == 'EOF':
                    break
                term = list(map(int, line.split()[1:3]))  #[0:3] para incluir NODE
                cities.append(term)

    
    cities = numpy.array(cities)
    print(cities)
    print(len(cities))
    return cities
    


def plot_cities(cities):
    cities = numpy.array(cities)
    x = cities[:, 0]
    y = cities[:, 1]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue')

    # Opcional: numerar os pontos
    for i, (xi, yi) in enumerate(cities):
        plt.text(xi + 1, yi + 1, str(i), fontsize=8)

    plt.title('Mapa de Cidades (Nodos)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')  # mantém escala 1:1 no plano
    plt.show()

def construct_euclidian_matrix(array):
    num_cities = len(array)
    dist_matrix = numpy.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist = numpy.linalg.norm(array[i] - array[j])
                dist_matrix[i][j] = dist
            else:
                dist_matrix[i][j] = 0.0  # distância para si mesmo

    print("Matriz de Distâncias:")
    print(dist_matrix)
    #output_path = "results/distance/distance_51.txt"
    output_path = "results/distance/distance_100.txt"
    with open(output_path, 'w') as f:
        for row in dist_matrix:
            f.write(' '.join(f'{val:.2f}' for val in row))  # arredondado com 2 casas decimais
            f.write('\n')
    return dist_matrix

if __name__ == "__main__":
    path_51 = "instances\\eil51-tsp.txt"
    path_100 = "instances\\kroA100-tsp.txt"
    array = read_values(path_51)
    #construct_euclidian_matrix(array)
    plot_cities(array)
    #array = read_values(path_100)
    #construct_euclidian_matrix(array)
