import random

def initialize_population(pop_size, string_length):
    population = []
    for _ in range(pop_size):
        individual = ''.join(random.choice('01') for _ in range(string_length))
        population.append(individual)
    return population

def calculate_fitness(individual):
    return individual.count('1')

def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    probabilities = [fitness / total_fitness for fitness in fitness_scores]
    parents = random.choices(population, weights=probabilities, k=2)
    return parents

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring = parent1[:point] + parent2[point:]
    return offspring

def mutate(individual, mutation_rate):
    mutated = ''.join(
        bit if random.random() > mutation_rate else ('1' if bit == '0' else '0')
        for bit in individual
    )
    return mutated

def genetic_algorithm(string_length, pop_size, num_generations, mutation_rate):
    population = initialize_population(pop_size, string_length)
    
    for generation in range(num_generations):
        fitness_scores = [calculate_fitness(individual) for individual in population]
        if max(fitness_scores) == string_length:
            print(f"Optimal solution found in generation {generation}")
            break
        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            new_population.extend([offspring1, offspring2])
        
        population = new_population
        print(f"Generation {generation}: Max Fitness = {max(fitness_scores)}")
    fitness_scores = [calculate_fitness(individual) for individual in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]
    return best_individual, max(fitness_scores)

if __name__ == "__main__":
    string_length = 10
    pop_size = 20
    num_generations = 100
    mutation_rate = 0.05

    best_solution, best_fitness = genetic_algorithm(
        string_length, pop_size, num_generations, mutation_rate
    )
    print(f"Best Solution: {best_solution} with Fitness: {best_fitness}")
