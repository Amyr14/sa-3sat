import random
from src.annealing import Domain
from typing import override
import numpy

# class TSP:
#     # def __init__(self, path):
#     #     with open(path, 'r') as tsp_file:
#     #         #self.num_vars, self.num_clauses = get_header_info(tsp_file)
        
#     #         points = []
#     #         for line in tsp_file:

#     #             if line == 'EOF':
#     #                 break
#     #             term = list(map(int, line.split()[1:3])) 
#     #             points.append(term)
            
#     #         self.cities_points = numpy.array(points)

#     #     self.num_cities = len(points)
    
#     #O custo seria a distância euclidiana do ponto na matriz? 
#     def get_valoration_metrics(self, valoration):
#         return self.cost(valoration)
    

class TSPDomain(Domain):
    def __init__(self, distance_matrix_path, SWAP_factor):
        self.SWAP_factor = SWAP_factor
        self.load_distance_matrix(distance_matrix_path)
        self.num_cities = self.distance_matrix.shape[0] 

    def load_distance_matrix(self, path):
        self.distance_matrix = numpy.loadtxt(path)
    

    """
    Realiza entre 1 e 5 perturbações (swaps) aleatórias na solução atual.
    Cada perturbação consiste em trocar dois índices distintos do vetor.
    """
    @override
    def get_neighbour(self, current):
        num_swaps = random.randint(1, int(self.SWAP_factor)) 
        new = numpy.copy(current)

        for _ in range(num_swaps):
            i, j = numpy.random.choice(current.size, size=2, replace=False)
            new[i], new[j] = new[j], new[i]  # faz o swap

        return new
        
    @override
    def random_value(self):
        #Gera um vetor de números aleatórios que não se repete, do tamanho do número de cidades
        return numpy.random.permutation(self.num_cities)
    
    @override    
    def cost(self, valoration):
        total_cost = sum(
            self.distance_matrix[valoration[i]][valoration[i + 1]]
            for i in range(len(valoration) - 1)
        )
        total_cost += self.distance_matrix[valoration[-1]][valoration[0]]
        return total_cost
        
    

    @override
    def get_label(self):
        return f'TSP {self.num_cities} cidades'

def get_header_info(tsp_file):
    header = tsp_file.readline()
    return map(int, header.split()[1:])