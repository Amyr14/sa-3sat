import numpy
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt

class Cooler(ABC):
    def __init__(self, t0, t_final=0):
        self.t0 = t0
        self.t_final = t_final
    
    @abstractmethod
    def cool(self, i):
        pass

    def graph(self, num_samples, ax=None, color='red'):
        x = numpy.arange(num_samples)
        y = [self.cool(i) for i in range(num_samples)]
        
        if ax is None:
            _, ax = plt.subplots()
            
        ax.plot(x, y, 'r--', label=self.get_function_label())
        ax.set_ylabel('Temperatura', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        return ax
    
    @abstractmethod
    def get_function_label(self) -> str:
        pass

# class SmoothCooler(Cooler):
#     def __init__(self, t0: float, cooling_factor: float):
#         super().__init__(t0, t_final=0)
#         self.cooling_factor = cooling_factor
    
#     def cool(self, i: int):
#         return self.t0 * numpy.power(self.cooling_factor, -(i/))
    
#     def get_function_label(self):
#         return r'$T = T_0 *{\alpha} ^{-i}$'
        
    
class FastCooler(Cooler):
    def __init__(self, t0: float, t_final: float, i_max: int):
        super().__init__(t0, t_final)
        self.i_max = i_max
    
    def cool(self, i: int):
        return self.t0 * numpy.power(self.t_final/self.t0, i/self.i_max)
    
    def get_function_label(self):
        return r'$T_i = T_0 \cdot (\frac{T_N}{T_0})^{\frac{i}{N}}$'

class Domain(ABC):
    @abstractmethod
    def get_neighbour(self, current):
        pass
    
    @abstractmethod
    def random_value(self):
        pass
    
    @abstractmethod
    def cost(self, value):
        pass

class History():
    def __init__(self, mode='minimize'):
        self.scores = []
        self.best_ever = None
        self.mode = mode
    
    def register_score(self, score):
        if self.best_ever is None or is_best_score(score, self.best_ever, self.mode):
            self.best_ever = score
        
        self.scores.append(score)
        
    def get_num_iterations(self):
        return len(self.scores)
        
    def graph(self, label, color='blue', ax=None):
        if ax is None:
            _, ax = plt.subplots()
        
        ax.plot(range(len(self.scores)), self.scores, 'b', label=label)
        ax.set_ylabel('Custo', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        return ax

def is_best_score(new_score, best, mode):
    return new_score < best if mode == 'minimize' else new_score > best

class Explorer():
    def __init__(self, domain: Domain, goal, epsilon, mode='minimize'):
        self.domain = domain
        self.mode = mode
        self.current = None
        self.current_energy = None
        self.goal = goal
        self.espilon = epsilon
        
    def step(self, temperature):
        if self.current is None:
            self.current = self.domain.random_value()
            self.current_energy = self.domain.cost(self.current)

        else:
            neighbour = self.domain.get_neighbour(self.current)
            neighbour_energy = self.domain.cost(neighbour)
            energy_delta = neighbour_energy - self.current_energy
            
            if (
                is_better_solution(energy_delta, self.mode) or
                accept_worse(energy_delta, temperature)
            ):
                self.current = neighbour
                self.current_energy = neighbour_energy
        
        return self.current_energy
    
    def reached_optimum(self):
        return abs(self.current_energy - self.goal) <= self.espilon if self.current_energy is not None else False
    
def is_better_solution(energy_delta, mode):
    return energy_delta < 0 if mode == 'minimize' else energy_delta > 0

def accept_worse(energy_delta, temperature):
    return numpy.random.rand() < numpy.exp(-energy_delta/temperature) if temperature != 0 else False # não funciona em caso de maximização

class SimulatedAnnealing:
    def __init__(self, t0: float, max_i: int, domain: Domain, cooler: Cooler, goal, epsilon, mode='minimize'):
        self.t0 = t0
        self.max_i = max_i
        self.explorer = Explorer(domain, goal, epsilon, mode)
        self.cooler = cooler
        self.mode = mode
        
    def run(self) -> History:
        history = History(self.mode)
        t = self.t0
        i = 0
        
        while i < self.max_i and not self.explorer.reached_optimum():
            score = self.explorer.step(temperature=t)
            t = self.cooler.cool(i)
            history.register_score(score)
            i += 1

        return history