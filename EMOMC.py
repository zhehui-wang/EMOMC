import function as Fun
import numpy as np
import random
import sys   
import argparse
import config as C
from pymoo.util.misc import stack
from pymoo.model.problem import Problem
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_algorithm, get_sampling, get_crossover, get_mutation
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


print("unit of size : kBytes")
print("unit of energy : 0.01 mJ")

# number of layers in the model 
layer_dim = C.get_counter()


working_mode = Fun.args.working_mode
number_generations = Fun.args.number_generations
population_size = Fun.args.population_size
size_constraint = Fun.args.size_constraint
energy_constraint = Fun.args.energy_constraint
pre_prune_upper_bound = Fun.args.pre_prune_upper_bound
pre_prune_lower_bound = Fun.args.pre_prune_lower_bound
pre_prune_step = Fun.args.pre_prune_step


# there are layer_dim+1 variables.
# the first element is for pruning, and the rest elements are for quantzation bits in each layer
lower_bound = [2] * (layer_dim + 1)
upper_bound = [12] * (layer_dim + 1)
lower_bound[0] =  pre_prune_lower_bound
upper_bound[0] =  pre_prune_upper_bound


class MyProblem(Problem):

    def __init__(self):
        super().__init__(n_var=layer_dim+1,
                         n_obj=2,
                         n_constr=1,
                         xl=lower_bound,
                         xu=upper_bound)

    def _evaluate(self, x, out, *args, **kwargs):
       
        population_size = x.shape[0]
        target_quatnize = [0] * layer_dim
        target_prune = 0
       
        accuracy, energy, size = [0] * population_size, [0] * population_size, [0] * population_size         
       
        for i in range (population_size):
                 
            # the first layer_dim slot is reserved for pruning
            for j in range (layer_dim): target_quatnize [j] = x.item((i,j+1))
            target_prune = int(x.item((i,0)))  
            
            print("\nnode", i)
            accuracy[i], energy[i], size[i] = Fun.sample_point(target_prune, target_quatnize)              
            
            #setup constraints
            if (working_mode == 1) : size[i] =  size[i]-size_constraint
            if (working_mode == 2) : energy[i] =  energy[i] - energy_constraint

            # accruacy is the larger the better
            accuracy[i] *= -1

        # trade-off between energy and accuracy    
        if (working_mode == 1):
            f1, f2 =  np.array(accuracy), np.array(energy)
            g1 =  np.array(size)
        # trade-off between size and ass  
        if (working_mode == 2):     
            f1, f2 =  np.array(accuracy), np.array(size)
            g1 =  np.array(energy)

        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1]) 
  
# main optimization function
def optimize():

    problem = MyProblem()  

    algorithm = NSGA2(
        pop_size=population_size * 2,
        n_offsprings=population_size,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )   
       
    termination = get_termination("n_gen", number_generations)    
   
    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=1,
                   pf=problem.pareto_front(use_cache=False),
                   save_history=True,
                   verbose=True)

    # print & plot the results
    print("Best solution found: %s" % res.X)
    print("Function value: %s" % res.F)

    pf = problem.pareto_front(use_cache=False, flatten=False)
    plot = Scatter(title = "Objective Space")
    plot.add(res.F)
    plot.add(pf, plot_type="line", color="black", alpha=0.7)
    plot.show()
 
if __name__ == '__main__':

    if (working_mode > 0):
        #optimization
        optimize()
        
    if (working_mode == 0):
        # step1: pre train the model from scratch
        Fun.pre_train()
        # step2: pre prune the model 
        for i in range(pre_prune_lower_bound, pre_prune_upper_bound+1, pre_prune_step):
            Fun.pre_prune(i);
