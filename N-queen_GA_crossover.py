import random

def initialize_population(pop_size, n):
    return [[random.randint(0, n-1) for _ in range(n)] for _ in range(pop_size)]

def fitness(chromosome):
    n = len(chromosome)
    attacks = 0
    for i in range(n):
        for j in range(i + 1, n):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == abs(i - j):
                attacks += 1
    return n * (n - 1) // 2 - attacks

def crossover_uniform(parent1, parent2):
    n = len(parent1)
    child = [-1] * n
    for i in range(n):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def crossover_shuffle(parent1, parent2):
    n = len(parent1)
    crossover_point = random.randint(1, n - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

def crossover_multiparent(parents):
    n = len(parents[0])
    child = [-1] * n
    for i in range(n):
        child[i] = random.choice(parents)[i]
    return child

def crossover_three_parent(parent1, parent2, parent3):
    n = len(parent1)
    child = [-1] * n
    for i in range(n):
        choices = [parent1[i], parent2[i], parent3[i]]
        child[i] = random.choice(choices)
    return child

def genetic_algorithm(pop_size, n, max_generations):
    population = initialize_population(pop_size, n)
    for generation in range(max_generations):
        # Evaluate fitness
        fitness_scores = [fitness(chromosome) for chromosome in population]
        # Selection
        selected_parents = random.choices(population, weights=fitness_scores, k=pop_size)
        # Crossover
        new_population = []
        for i in range(0, pop_size, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            # Choose crossover technique
            child = crossover_uniform(parent1, parent2)
            #child = crossover_shuffle(parent1, parent2)
            #child = crossover_multiparent(selected_parents[i:i+3])
            #child = crossover_three_parent(parent1, parent2, selected_parents[i + 2])
            new_population.append(child)
        population = new_population
    # Return the best solution found
    best_solution = max(population, key=fitness)
    return best_solution

# Example usage
n = 8  # Size of the board
pop_size = 100  # Population size
max_generations = 1000  # Maximum number of generations
solution = genetic_algorithm(pop_size, n, max_generations)
print("Solution:", solution)
