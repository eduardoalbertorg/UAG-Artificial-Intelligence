from typing import List
import random
import math
import numpy as np
import matplotlib.pyplot as plt

Genome_distances = List[float]
CITIES_DICT = {
    1: [1, 3],
    2: [2, 5],
    3: [2, 7],
    4: [4, 2],
    5: [4, 4],
    6: [4, 7],
    7: [4, 8],
    8: [5, 3],
    9: [6, 1],
    10: [6, 6],
    11: [7, 8],
    12: [8, 2],
    13: [8, 7],
    14: [9, 3],
    15: [10, 7],
    16: [11, 1],
    17: [11, 4],
    18: [11, 6],
    19: [12, 7],
    20: [13, 5]
}
Population = List[int]


def generate_genome_distances():
    for i, genome in enumerate(Population):
        Genome_distances[i] = calculate_total_distance_in_genome(genome)
    return Genome_distances

def generate_cromosome(size: int) -> List[int]:
    aux_list = list()
    aux_list = [value for value in range(1, size+1)]
    random.shuffle(aux_list)
    return aux_list

def generate_population(size: int, genome_size: int) -> Population:
    return [generate_cromosome(genome_size) for _ in range(size)]

def calculate_total_distance_in_genome(genome: List[int]) -> float:
    total_distance = 0
    for i in range(1, len(genome)-1):
        total_distance += math.dist(CITIES_DICT[genome[i]], CITIES_DICT[genome[i+1]])
        #total_distance += math.dist(genome[i], genome[i+1])
    return total_distance

def selection(contestants: List[int]) -> List[int]:
    distances = [calculate_total_distance_in_genome(contestant) for contestant in contestants]
    winner_contestant_index = np.argmin(distances)
    return contestants[winner_contestant_index]

def crossover(genome: List[int]) -> List[int]:
    techniques_list = [technique_1, technique_2]
    return random.choice(techniques_list)(genome)
    
def technique_1(genome: List[int]) -> List[int]:
    print("###Applying technique 1")
    genome_size = len(genome) - 1
    half = genome_size // 2
    segment_length = random.randint(0, half)
    starting_point = random.randint(0, half - segment_length)
    second_starting_point = half + 1
    genome[starting_point: starting_point + segment_length], genome[second_starting_point: second_starting_point + segment_length] = genome[second_starting_point: second_starting_point + segment_length], genome[starting_point: starting_point + segment_length]
    return genome

def technique_2(genome: List[int]) -> List[int]:
    print("###Applying technique 2")
    last_index = len(genome) - 1
    left_pointer = random.randint(0, last_index - 1)
    right_pointer = random.randint(left_pointer + 1, last_index)
    while left_pointer < right_pointer:
        genome[left_pointer], genome[right_pointer] = genome[right_pointer], genome[left_pointer]
        left_pointer += 1
        right_pointer -= 1
    return genome

def get_winner_index(population: Population, contestants_size: int) -> int:
    population_size = len(population)
    contestants_indices = random.choices(range(0, population_size-1), k=contestants_size)
    shortest_distance = float('inf')
    winner_index = None
    for i in contestants_indices:
        distance = calculate_total_distance_in_genome(population[i])
        if distance < shortest_distance:
            shortest_distance = distance
            winner_index = i
    return winner_index

def plot_city_route(winner_index):
    winner_route_x = list()
    winner_route_y = list()
    cities_x = list()
    cities_y = list()
    for city in Population[winner_index]:
        x, y = CITIES_DICT.get(city)
        winner_route_x.append(x)
        winner_route_y.append(y)
    for x, y in CITIES_DICT.values():
        cities_x.append(x)
        cities_y.append(y)
    ax1.clear()
    ax1.set_title("Route")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.plot(winner_route_x, winner_route_y, linestyle='dashed')
    ax1.plot(cities_x, cities_y, linestyle='none', marker='o')

def plot_fitness_aptitude_function(generation_list, distance_evolution_list):
    ax2.clear()
    ax2.set_title("Fitness function")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Distance")
    ax2.plot(generation_list, distance_evolution_list)

if __name__ == "__main__":
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    generation_list = list()
    distance_evolution_list = list()
    #POPULATION_SIZE = 10 #100
    POPULATION_SIZE = 100
    #GENOME_LENGTH = 5#20
    GENOME_LENGTH = 20
    #GENERATION_LIMIT = 10#100
    GENERATION_LIMIT = 100
    #NUMBER_OF_TOURNAMENTS = 10#100
    NUMBER_OF_TOURNAMENTS = 100
    #NUMBER_OF_CONTESTANTS_PER_TOURNAMENT: int = 2#int(100 * 0.05)
    NUMBER_OF_CONTESTANTS_PER_TOURNAMENT: int = int(100 * 0.05)
    Genome_distances = [float('inf') for _ in range(POPULATION_SIZE)]
    Population = generate_population(POPULATION_SIZE, GENOME_LENGTH)
    for generation in range(GENERATION_LIMIT):
        generation_list.append(generation)
        children = list()
        for tournament in range(NUMBER_OF_TOURNAMENTS):
            # Calculate the distance for the population
            Genome_distances = generate_genome_distances()
            # Pick random contestants for the tournament
            contestants = random.choices(Population, k=NUMBER_OF_CONTESTANTS_PER_TOURNAMENT)
            # Select the winner from the tournament with the smallest traveling distance
            #winner_genome = selection(contestants)
            winner_index = get_winner_index(Population[:], NUMBER_OF_CONTESTANTS_PER_TOURNAMENT)
            winner_cromosome = Population[winner_index]
            # Generate offspring
            child = crossover(winner_cromosome[:])
            children.append(child[:])
        Population = children
        Genome_distances = generate_genome_distances()
        population_winner = min(Genome_distances)
        distance_evolution_list.append(population_winner)
        plot_city_route(winner_index)
        plot_fitness_aptitude_function(generation_list, distance_evolution_list)
        plt.show(block = False)
        plt.pause(0.5)

    