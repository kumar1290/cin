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

    def calculate_fitness(self, chromosome):
        total_distance = sum(
            self.cities_distances[chromosome[i]][chromosome[(i + 1) % self.num_cities]]
            for i in range(self.num_cities)
        )
        return 1 / total_distance  # Inverse of the total distance is used as fitness

    def tournament_selection(self, population, k=3):
        participants = random.sample(population, k)
        return max(participants, key=self.calculate_fitness)

    def partially_mapped_crossover(self, parent1, parent2):
        # Randomly select two crossover points
        start_idx = random.randint(0, self.num_cities - 1)
        end_idx = random.randint(start_idx, self.num_cities - 1)
        # Create child
        child = [None] * self.num_cities
        for i in range(start_idx, end_idx + 1):
            child[i] = parent1[i]
        # Map elements from parent2 to child
        for i in range(start_idx, end_idx + 1):
            if parent2[i] not in child:
                idx = parent2.index(parent1[i])
                while child[idx] is not None:
                    idx = parent2.index(parent1[idx])
                child[idx] = parent2[i]
        # Copy the remaining elements from parent2
        for i in range(self.num_cities):
            if child[i] is None:
                child[i] = parent2[i]
        return child

    def scrambled_mutation(self, chromosome):
        # Randomly select two positions and scramble the values in between
        start_idx = random.randint(0, self.num_cities - 1)
        end_idx = random.randint(start_idx, self.num_cities - 1)
        scrambled_section = chromosome[start_idx:end_idx + 1]
        random.shuffle(scrambled_section)
        chromosome[start_idx:end_idx + 1] = scrambled_section
        return chromosome

    def evolve(self):
        self.population = self.initialize_population()
        for generation in range(self.max_generations):
            fitness_scores = [self.calculate_fitness(chromosome) for chromosome in self.population]
            # Selection
            selected_parents = [self.tournament_selection(self.population) for _ in range(self.population_size)]
            # Crossover
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1]
                child = self.partially_mapped_crossover(parent1, parent2)
                new_population.append(child)
            # Mutation
            for i in range(len(new_population)):
                if random.random() < mutation_rate:
                    new_population[i] = self.scrambled_mutation(new_population[i])
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
