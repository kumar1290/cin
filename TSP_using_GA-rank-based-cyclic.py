import random
import numpy as np

class TSPGeneticAlgorithm:
    def __init__(self, num_cities, population_size, max_generations, cities_distances):
        self.num_cities = num_cities
        self.population_size = population_size
        self.max_generations = max_generations
        self.cities_distances = cities_distances

    def initialize_population(self):
        return [random.sample(range(self.num_cities), self.num_cities) for _ in range(self.population_size)]

    def rank_selection(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)
        ranks = np.arange(1, len(fitness_scores) + 1)
        probabilities = ranks / sum(ranks)
        selected_indices = np.random.choice(sorted_indices, size=self.population_size, replace=True, p=probabilities)
        return [self.population[i] for i in selected_indices]

    def calculate_fitness(self, chromosome):
        total_distance = sum(
            self.cities_distances[chromosome[i]][chromosome[(i + 1) % self.num_cities]]
            for i in range(self.num_cities)
        )
        return 1 / total_distance  # Inverse of the total distance is used as fitness

    def crossover_cyclic_inversion(self, parent1, parent2):
        child = [-1] * self.num_cities
        start_index = random.randint(0, self.num_cities - 1)
        end_index = random.randint(0, self.num_cities - 1)
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        child[start_index:end_index + 1] = parent1[start_index:end_index + 1]
        for i in range(start_index, end_index + 1):
            if parent2[i] not in child:
                idx = i
                while True:
                    if parent2[idx] in child:
                        idx = parent2.index(parent2[idx])
                    else:
                        break
                child[idx] = parent2[i]
        for i in range(self.num_cities):
            if child[i] == -1:
                child[i] = parent2[i]
        return child

    def mutation_inversion_reversing(self, chromosome):
        start_index = random.randint(0, self.num_cities - 1)
        end_index = random.randint(0, self.num_cities - 1)
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        chromosome[start_index:end_index + 1] = chromosome[start_index:end_index + 1][::-1]
        return chromosome

    def evolve(self):
        self.population = self.initialize_population()
        for generation in range(self.max_generations):
            fitness_scores = [self.calculate_fitness(chromosome) for chromosome in self.population]
            # Selection
            selected_parents = self.rank_selection(fitness_scores)
            # Crossover
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child = self.crossover_cyclic_inversion(parent1, parent2)
                new_population.append(child)
            # Mutation
            for i in range(len(new_population)):
                if random.random() < mutation_rate:
                    new_population[i] = self.mutation_inversion_reversing(new_population[i])
            self.population = new_population
        best_solution = max(self.population, key=self.calculate_fitness)
        return best_solution

# Example usage
num_cities = 10  # Number of cities
population_size = 100  # Population size
max_generations = 1000  # Maximum number of generations
mutation_rate = 0.1  # Mutation rate

# Generate random distances between cities (example)
cities_distances = np.random.rand(num_cities, num_cities)

# Create TSP genetic algorithm object
tsp_genetic_algorithm = TSPGeneticAlgorithm(num_cities, population_size, max_generations, cities_distances)

# Solve TSP
best_route = tsp_genetic_algorithm.evolve()
print("Best Route:", best_route)
