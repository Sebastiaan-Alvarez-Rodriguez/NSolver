from nsolver.solver import Solver, evaluate

from copy import copy
import numpy as np


class SimulatedAnnealing(Solver):
    '''Simulated annealing algorithm to solve magic N-cubes.'''
    def __init__(self):
        self.pm = 2            # mutation rate
        self.alpha = 0.7       # temperature decaying parameter
        self.iter_length = 100 # number of evaluations per iteration

        self.T = 25000


    @staticmethod
    def generate_random_answer(n, dim):
        '''Produces a valid, completely random answer.
        Note: An answer to the problem is not necessarily a solution to the problem.
              In fact, a random answer is unlikely to be a solution for the N-cube problem.
        Args:
            n (int): Axiomial dimension vector length (e.g., for a magic square of 3x3 fields (a 3-2D cube), n=3 and dim=2.
            dim (int): The amount of correlated dimensions (e.g., for a magic cube of x*x*x*x fields (a x-4D cube, n=x and dim=4.'''
        return np.random.permutation(np.arange(1, (n ** dim)+1))


    @staticmethod
    def mutate_answer(s, n, dim, pm, fitness, fitness_optimal):
        '''Mutates our current answer in hopes of finding a closer solution. Returns the permutated solution.
        The given mutation is a simple swap for a number of times, depending on the difference between the known optimum and given `s`, and `pm`.
        Args:
            s (list(int)): current answer to change. Note: Returns a new solution without modifying this parameter.
            n (int): Axiomial dimension vector length (e.g., for a magic square of 3x3 fields (a 3-2D cube), n=3 and dim=2.
            dim (int): The amount of correlated dimensions (e.g., for a magic cube of x*x*x*x fields (a x-4D cube, n=x and dim=4.
            pm (float): Permutation.
            fitness (double): The fitness value of our current solution `s`.
            fitness_optimal (double): The fitness value of the currently known most optimal solution.
        Returns:
            list(int): The permutated solution.'''

        length = n ** dim
        s_cpy = copy(s)

        # Do more mutations if we are far away from the optimal solution we found
        diff = max(fitness - fitness_optimal, 1)
        t_evals = int(np.ceil(diff * pm))
        
        # Mutate n times by swapping
        for i in range(1, t_evals):
            mut_left = np.random.randint(0, high=length-1)
            mut_right = np.random.randint(mut_left, high=length)
            s_cpy[mut_left], s_cpy[mut_right] = s_cpy[mut_right], s_cpy[mut_left]
        return s_cpy


        def execute(self, n, dim, evaluations):
            # if do_plot:
            #     plt.ion()
            #     fig = plt.figure()

            #     ax1 = plt.subplot(131)
            #     line1 = ax1.plot(hist_best_f[:evalcount])[0]
            #     ax1.set_title('minimal global error')
            #     ax1.set_ylabel('error')
            #     ax1.set_xlabel('evaluations')
            #     ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

            #     ax2 = plt.subplot(132)
            #     line2 = ax2.plot(np.arange(itercount), hist_temperature[:itercount])[0]
            #     ax2.set_title('temperature')
            #     ax2.set_ylabel('T')
            #     ax2.set_xlabel('iteration')
            #     ax2.set_ylim([0, T])

            #     ax3 = plt.subplot(133)
            #     bars3 = ax3.bar(np.arange(len(solution_optimal)), solution_optimal)
            #     ax3.set_title('best representation')
            #     ax3.set_ylabel('value')
            #     ax3.set_xlabel('representation index')

            #     plt.show(block=False)
            
            evalcount = 0
            itercount = 0

            # Statistics data
            hist_best_f = np.array([np.nan] * evaluations)
            num_iterations = int(np.ceil(evaluations / self.iter_length))
            hist_iter_f = np.array([np.nan] * num_iterations)
            hist_temperature = np.array([np.nan] * num_iterations)


            # Generate initial solution and evaluate
            solution_optimal = generate_random_answer(n, dim)
            fitness_optimal = evaluate(x)  # evaluate the solution
            solution = copy(solution_optimal)
            fitness = fitness_optimal


            while evalcount < evaluations:
                hist_temperature[itercount] = self.T

                self.iter_length = min(self.iter_length, evaluations-evalcount)
                for _ in range(self.iter_length):

                    solution_new = mutate_answer(solution, n, dim, self.pm, fitness, fitness_optimal)   # Generate a new solution by permutating the current solution
                    fitness_new = fitness_func(solution_new)   # evaluate the new solution

                    if fitness_new < fitness or np.random.randn() < np.exp(-(fitness_new - fitness) / T):
                        # Our found mutation is closer to a solution than the current answer, or
                        # annealing formula mandates we pick this solution, even if it is a bit worse in terms of fitness.
                        solution = solution_new
                        fitness = fitness_new

                    if fitness > 2 * fitness_optimal: 
                        # Reset to the optimal solution if we are too far away from found optimum.
                        solution = copy(solution_optimal)
                        fitness = fitness_optimal
                    
                    if fitness < fitness_optimal:
                        # Update the best solution found so far if our current solution is better.
                        fitness_optimal = fitness
                        solution_optimal = copy(solution)

                    hist_best_f[evalcount] = fitness_optimal   # tracking the best fitness ever found

                    # Generation best statistics
                    hist_iter_f[itercount] = fitness
                    
                    # Plot statistics
                    # if do_plot:
                    #     line1.set_data(np.arange(evalcount), hist_best_f[:evalcount])
                    #     ax1.set_xlim([0, evalcount])
                    #     ax1.set_ylim([0, np.max(hist_best_f[:evalcount])])

                    #     line2.set_data(np.arange(itercount), hist_temperature[:itercount])
                    #     ax2.set_xlim([0, itercount])

                    #     for bar, h in zip(bars3, solution_optimal):
                    #         bar.set_height(h)

                    #     plt.pause(0.00001)
                    #     plt.draw()
                    evalcount += 1   # Increase evaluation counter
                T = alpha * T

                print(f'{evalcount}: current fitness: {fitness_optimal}')
                itercount += 1   # Increase iteration counter

            # if return_stats:
            #     return solution_optimal, fitness_optimal, hist_best_f
            # else:
            #     return solution_optimal, fitness_optimal
            return solution_optimal