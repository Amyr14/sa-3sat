import os
import matplotlib.pyplot as plt
from src.annealing import SimulatedAnnealing, get_cooling_schedule_label
from src.sat import SATDomain

# Configurações de diretório
FORMULAS_DIR = './instances'
RESULTS_DIR = './results/sa_max_experiment'

# Configuração de experimento

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

files = os.listdir(FORMULAS_DIR)
paths = [os.path.join(FORMULAS_DIR, f) for f in files]
instances = [SATDomain(p, flip_prob=0.01) for p in paths]
