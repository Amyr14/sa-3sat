import numpy

class CNF:
    def __init__(self, path):
        with open(path, 'r') as cnf_file:
            self.num_vars, self.num_clauses = get_header_info(cnf_file)
        
            formula = []
            for line in cnf_file:
                if line[0] == '%':
                    break
                term = list(map(int, line.split()[0:3])) 
                formula.append(term)
            
            self.formula = numpy.array(formula)
    
    # Criar um tipo Valoration?
    def get_valoration_metrics(self, valoration):
        if valoration.size != self.num_vars:
            raise Exception('Número de variáveis inválido')
        
        variables = valoration[abs(self.formula) - 1]
        signs = self.formula > 0
        clause_values = (variables == signs).any(axis=1)
        num_satisfied = numpy.count_nonzero(clause_values) 
        num_unsatisfied = self.num_clauses - num_satisfied
        return num_satisfied, num_unsatisfied
    
# talvez o número de termos seja desnecessario
def get_header_info(cnf_file):
    header = cnf_file.readline()
    return map(int, header.split()[1:])