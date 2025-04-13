import numpy
from abc import abstractmethod, ABC
    
def cooling_schedule1(i, t0, t_final, eval_max):
    return t0 * (t_final/t0) ** (i/eval_max)

def cooling_schedule2(i, t0, t_final, eval_max):
    A = ((t0 - t_final) * (eval_max + 1)) / eval_max
    B = t0 - A
    return A / (i+1) + B

def cooling_schedule3(i, t0, t_final, eval_max):
    return 0.5 * (t0 - t_final) * (1 - numpy.tanh(10*i/eval_max - 5)) + t_final

COOLING_SCHEDULES = [
    cooling_schedule1,
    cooling_schedule2,
    cooling_schedule3,
]

# Para facilitar a criação de gráficos
COOLING_SCHEDULES_LABELS = [
    r'$T_i = T_0 \cdot (\frac{T_N}{T_0})^{\frac{i}{N}}$',
    r'$T_i = \frac{A}{i + 1} + B$',
    r'$\frac{1}{2}(T_0 - T_N)  (1 - \tanh(\frac{10i}{N} - 5) + T_N$'
]

def get_cooling_schedule_label(cooling_schedule_i):
    return COOLING_SCHEDULES_LABELS[cooling_schedule_i - 1]
        
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
    
    @abstractmethod
    def get_label(self) -> str:
        pass


class SimulatedAnnealing:
    def __init__(self, cooling_schedule_i: int, domain: Domain):
        self.domain = domain
        self.cooling_schedule = COOLING_SCHEDULES[cooling_schedule_i - 1]

    def run(self, t0, t_final, sa_max, eval_max) -> dict:
        iter_t = 0
        i_eval = 0
        t = t0
        current = self.domain.random_value()
        current_energy = self.domain.cost(current)
        best_ever = current
        best_ever_energy = current_energy
        
        # Para registro
        energies = [current_energy]
        temperatures = [t0]
        
        while i_eval < eval_max:
            i_eval += 1
            
            while iter_t < sa_max:
                iter_t += 1
                neighbour = self.domain.get_neighbour(current)
                neighbour_energy = self.domain.cost(neighbour)
                energy_delta = neighbour_energy - current_energy
                
                if energy_delta < 0:
                    current = neighbour
                    current_energy = neighbour_energy
                    if neighbour_energy < best_ever_energy:
                        best_ever = neighbour
                        best_ever_energy = neighbour_energy
                
                elif accept_worse(energy_delta, t):
                    current = neighbour
                    current_energy = neighbour_energy
            
            t = self.cooling_schedule(i_eval, t0, t_final, eval_max)
            iter_t = 0
            temperatures.append(t)
            energies.append(current_energy)
                
        return {
            'temperatures': temperatures,
            'energies': energies,
            'best_ever': best_ever,
            'best_ever_energy': best_ever_energy,
        }

def accept_worse(energy_delta, temperature):
    return numpy.random.rand() < numpy.exp(-energy_delta/temperature) if temperature != 0 else False