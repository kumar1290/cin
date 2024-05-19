import numpy as np

class AntColony:
    def __init__(self, num_ants, num_iterations, alpha, beta, rho, q, cities_distances):
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # Pheromone influence
        self.beta = beta  # Heuristic influence
        self.rho = rho  # Evaporation rate
        self.q = q  # Pheromone deposit factor
        self.cities_distances = cities_distances
        self.num_cities = len(cities_distances)
        self.pheromone = np.ones((self.num_cities, self.num_cities)) / self.num_cities
        self.best_solution = None
        self.best_distance = float('inf')

    def construct_solutions(self):
        for _ in range(self.num_iterations):
            solutions = []
            for ant in range(self.num_ants):
                solution = self.construct_solution()
                solutions.append((solution, self.calculate_distance(solution)))
                if solutions[-1][1] < self.best_distance:
                    self.best_solution = solution
                    self.best_distance = solutions[-1][1]
            self.update_pheromone(solutions)

    def construct_solution(self):
        visited = [False] * self.num_cities
        solution = []
        current_city = 0  # Start from city 0
        visited[current_city] = True
        solution.append(current_city)
        for _ in range(self.num_cities - 1):
            probabilities = self.calculate_probabilities(current_city, visited)
            next_city = np.random.choice(range(self.num_cities), p=probabilities)
            solution.append(next_city)
            visited[next_city] = True
            current_city = next_city
        return solution

    def calculate_probabilities(self, current_city, visited):
        pheromone_values = self.pheromone[current_city] ** self.alpha
        heuristic_values = (1 / (self.cities_distances[current_city] + 1e-10)) ** self.beta
        probabilities = pheromone_values * heuristic_values
        probabilities[visited] = 0
        probabilities /= probabilities.sum()
        return probabilities

    def calculate_distance(self, solution):
        distance = 0
        for i in range(len(solution)):
            distance += self.cities_distances[solution[i-1]][solution[i]]
        return distance

    def update_pheromone(self, solutions):
        self.pheromone *= (1 - self.rho)  # Evaporation
        for solution, distance in solutions:
            for i in range(len(solution)):
                self.pheromone[solution[i-1]][solution[i]] += self.q / distance

# Example usage
num_ants = 10
num_iterations = 100
alpha = 1  # Pheromone influence
beta = 2  # Heuristic influence
rho = 0.1  # Evaporation rate
q = 1  # Pheromone deposit factor

# Example distances between cities (you need to provide your distances)
cities_distances = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

# Create AntColony object and solve TSP
aco = AntColony(num_ants, num_iterations, alpha, beta, rho, q, cities_distances)
aco.construct_solutions()
print("Optimal solution:", aco.best_solution)
print("Optimal distance:", aco.best_distance)
