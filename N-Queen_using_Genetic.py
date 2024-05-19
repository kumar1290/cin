## N-Queen using Genetic Algorithm

import random

class NQueensGeneticAlgorithm:
    def __init__(self, population_size, n, crossover_rate, mutation_rate):
        self.population_size = population_size
        self.n = n
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        return [random.sample(range(1, self.n + 1), self.n) for _ in range(self.population_size)]

    def fitness(self, chromosome):
        attacks = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(i - j) == abs(chromosome[i] - chromosome[j]):
                    attacks += 1
        return self.n * (self.n - 1) / 2 - attacks  # Maximum non-attacking pairs - number of attacking pairs

    def single_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.n - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, chromosome):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.n), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

    def select_parents(self, population):
        # Tournament selection
        tournament_size = 2
        selected_parents = []
        for _ in range(self.population_size):
            participants = random.sample(population, tournament_size)
            selected_parents.append(max(participants, key=self.fitness))
        return selected_parents

    def evolve(self, generations):
        population = self.initialize_population()
        for generation in range(generations):
            new_population = []
            for _ in range(self.population_size // 2):  # Create new population
                parent1, parent2 = random.sample(population, 2)
                if random.random() < self.crossover_rate:
                    child1, child2 = self.single_point_crossover(parent1, parent2)
                    new_population.extend([child1, child2])
                else:
                    new_population.extend([parent1, parent2])
            # Mutation
            for individual in new_population:
                self.mutate(individual)
            population = new_population
        best_solution = max(population, key=self.fitness)
        return best_solution

# Example usage
population_size = 100
n = 8  # Number of queens
crossover_rate = 0.8
mutation_rate = 0.1
generations = 1000

# Create NQueensGeneticAlgorithm object
ga = NQueensGeneticAlgorithm(population_size, n, crossover_rate, mutation_rate)

# Solve N-Queens problem
solution = ga.evolve(generations)
print("Best solution:", solution)
