import logging    # first of all import the module
import math
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import List
from copy import copy
import pdb

logging.basicConfig(filename='std.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
#Let us Create an object 
logger = logging.getLogger()

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG)
population = list()
genome_fitness_array = list()

def get_reference_func():
    y = list()
    A = 8
    B = 25
    C = 4
    D = 45
    E = 10
    F = 17
    G = 35
    for i in range(1000):
        x = i / 10
        y.append((A * ((B * math.sin(x / C)) + (D * math.cos(x / E)))) + (F * x) - G)
    return y

def initialize_genome_fitness(size: int):
    return [float('inf') for _ in range(size)]

def generate_population(size: int, cromosome_size: int) -> population:
    return [generate_cromosomes(cromosome_size) for _ in range(size)]

def generate_cromosomes(size: int):
    return [random.randint(1, 255) for _ in range(size)]

def get_function_from_cromosome(cromosome: list):
    y = list()
    weight = max(cromosome)
    cromosome = [gene / weight for gene in cromosome]
    for i, value in enumerate(cromosome):
        if value == 0:
            cromosome[i] = 0.1
    m1 = cromosome[0]
    de1 = cromosome[1]
    m2 = cromosome[2]
    de2 = cromosome[3]
    m3 = cromosome[4]
    de3 = cromosome[5]
    p1 = cromosome[6]
    q1 = cromosome[7]
    p2 = cromosome[8]
    q2 = cromosome[9]
    p3 = cromosome[10]
    q3 = cromosome[11]

    for i in range(1000):
        x = i / 10
        mf1 = math.exp((-(x - m1)**2) / (2 * de1**2))
        mf2 = math.exp((-(x - m2)**2) / (2 * de2**2))
        mf3 = math.exp((-(x - m3)**2) / (2 * de3**2))
        b = mf1 + mf2 + mf3
        a1 = mf1 * (p1 * x + q1)
        a2 = mf2 * (p2 * x + q2)
        a3 = mf3 * (p3 * x + q3)
        a = a1 + a2 + a3
        if a == b and b == 0:
            print(cromosome)
            print(f"a: {a}, b: {b}")
            y.append(0.1)
        else:
            y.append(a/b)
    return y

def convert_to_binary(value: int) -> str:
    logger.debug(f"## Converting to Binary: {value} {type(value)}")
    return bin(value).replace("0b", "").zfill(8)

def split_binary_value(upper_length: int, value: str):
    upper_value = value[:upper_length]
    lower_value = value[upper_length:]
    return upper_value, lower_value

def select_random_bits_to_cut() -> int:
    return random.randint(1, 55)

def select_cromosome_cut_position(bits_to_cut: int) -> tuple[int, float]:
    '''
        Method to return the index for the cut to be made
    '''
    fractional = None
    index_to_cut = None
    if bits_to_cut % 8 != 0:
        fractional, index_to_cut = math.modf(bits_to_cut / 8)
    else:
        index_to_cut = bits_to_cut / 8 - 1
    return int(index_to_cut), fractional

def is_cut_clean(bits_to_cut: int):
    return bits_to_cut % 8 == 0

def split_gene(cromosome: list, index_to_cut: int, bits_to_cut_gene: int):
    '''
    Method to split the gene inside of a cromosome
    '''
    encoded_gene_to_cut = cromosome[index_to_cut]
    decoded_gene_to_cut = convert_to_binary(encoded_gene_to_cut)
    upper_value, lower_value = split_binary_value(bits_to_cut_gene, decoded_gene_to_cut)
    return upper_value, lower_value

def reproduce(father_cromosome: list, mother_cromosome: list, bits_to_cut_cromosome = None):
    logger.info(f"## Reproducing")
    logger.debug(f"## Parent1: {father_cromosome}")
    logger.debug(f"## Parent2: {mother_cromosome}")
    child_1 = list()
    child_2 = list()
    if bits_to_cut_cromosome == None:
        bits_to_cut_cromosome = select_random_bits_to_cut()

    logger.debug(f"Bits to cut: {bits_to_cut_cromosome}")
    index_to_cut, fractional = select_cromosome_cut_position(bits_to_cut_cromosome)
    logger.debug(f"Index to cut: {index_to_cut}, Fractional: {fractional}")
    if fractional != None:
        # Means that we must cut through the gene
        bits_to_cut_gene = int(8 * fractional)
        # Split the whole gene in 2 parts -> Upper and Lower
        gene_parent_1_upper, gene_parent_1_lower = split_gene(copy(father_cromosome), index_to_cut, bits_to_cut_gene)
        gene_parent_2_upper, gene_parent_2_lower = split_gene(copy(mother_cromosome), index_to_cut, bits_to_cut_gene)
        # Fuse the new child using bit exchange from both parents
        gene_child_1 = gene_parent_1_upper + gene_parent_2_lower
        gene_child_2 = gene_parent_2_upper + gene_parent_1_lower
        # Place the children where they belong
        father_cromosome[index_to_cut] = int(gene_child_2, 2)
        mother_cromosome[index_to_cut] = int(gene_child_1, 2)
    # Split the cromosome(row) into 2 parts -> Upper and Lower
    parent_1_upper, parent_1_lower = father_cromosome[:index_to_cut], father_cromosome[index_to_cut:]
    parent_2_upper, parent_2_lower = mother_cromosome[:index_to_cut], mother_cromosome[index_to_cut:]
    # Fuse parts from both parents to create new children
    child_1 = parent_1_upper + parent_2_lower
    child_2 = parent_2_upper + parent_1_lower
    logger.debug(f"## Resulting children:")
    logger.debug(f"## Children 1: {child_1}")
    logger.debug(f"## Children 2: {child_2}")
    return child_1, child_2


def get_best_fitted_parents(population: list, tournament_limit: int, number_of_contestants: int, exclude_indices_array: list = None):
    logger.info(f"## Getting best fitted parents")
    parents_array = list()
    contestants_indices = list()
    population_size = len(population)
    for _ in range(tournament_limit):
        if exclude_indices_array == None:
            contestants_indices = random.choices(range(0, population_size-1), k=number_of_contestants)
        else:
            contestants_indices = random.choices([i for i in range(0, population_size-1) if i not in exclude_indices_array], k=number_of_contestants)
            #contestants_indices = random.choices(range(0, population_size-1) if not in , k=number_of_contestants)
        shortest_error = float('inf')
        winner_index = None
        for i in contestants_indices:
            fitness = genome_fitness_array[i]
            if fitness < shortest_error:
                shortest_error = fitness
                winner_index = i
        parents_array.append(winner_index)
    logger.debug(f"## Best fitted parents: \n{parents_array}")
    return parents_array

def mutate(population: list, population_percentage: int):
    bits_to_mutate = int(population_percentage * len(population) / 100)
    logger.info(f"## Mutating total of {bits_to_mutate} bits")
    for i in range(bits_to_mutate):
        logger.debug(f"## Current mutation {i}")
        random_population_index = random.randint(0, len(population) - 1)
        bit = random.randint(1, 56)
        cut_index, fractional = select_cromosome_cut_position(bit)
        if fractional == None:
            fractional = 0
        bit_position = int(8 * fractional)
        population[random_population_index] = negate_bit(population[random_population_index], cut_index, bit_position)
    return population

def negate_bit(cromosome:list, cut_index:int, bit_position: int):
    logger.info("## Negating")
    logger.debug(f"## Cut index {cut_index}")
    logger.debug(f"## Cromosome: {len(cromosome)} {cromosome}")
    binary_gene_value_as_list = list(convert_to_binary(cromosome[cut_index]))
    binary_gene_value_as_list[bit_position] = '0' if (binary_gene_value_as_list[bit_position] == 1) else '1'
    binary_gene_value_as_string = "".join(binary_gene_value_as_list)
    decimal_gene_value = int(binary_gene_value_as_string, 2)
    cromosome[cut_index] = decimal_gene_value
    return cromosome

def elitism(elitist_array):
    elitist_array = generate_populations_fitness(reference_function, population)
    pass

def generate_populations_fitness(reference_function, population: list):
    logger.info(f"## Generating fitness for cromosome")
    for i, cromosome in enumerate(population):
        summation = 0.0
        cromosome_func = get_function_from_cromosome(cromosome)
        for index in range(len(reference_function)):
            summation += int(abs(reference_function[index] - cromosome_func[index]))
        #genome_fitness_array[i] = np.sum(np.absolute(reference_function - get_function_from_cromosome(cromosome)))
        genome_fitness_array[i] = summation
    return genome_fitness_array

if __name__ == "__main__":
    POPULATION_SIZE = 100
    CROMOSOME_SIZE = 12
    GENERATIONS_LIMIT = POPULATION_SIZE * 2
    NUMBER_OF_CONTESTANTS = int(POPULATION_SIZE * 0.05)
    TOURNAMENT_LIMIT = int(POPULATION_SIZE / 2)
    APPLY_ELITISM = False
    MUTATION_PERCENTAGE = 7
    reference_function = get_reference_func()
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    population = generate_population(POPULATION_SIZE, CROMOSOME_SIZE)
    # Prepares fitness array
    genome_fitness_array = initialize_genome_fitness(POPULATION_SIZE)
    print(f"size of ref func = {len(reference_function)}")
    print(f"size of population func = {len(population)}")
    #input()
    # Initializes fitness array with corresponding values
    genome_fitness_array = generate_populations_fitness(copy(reference_function), copy(population))
    generation_list = list()
    best_fit_list = list()
    for generation in range(GENERATIONS_LIMIT):
        logger.debug(f"Current generation: # {generation}")
        generation_list.append(generation)
        children = list()
        fathers_indices_array = get_best_fitted_parents(copy(population), TOURNAMENT_LIMIT, NUMBER_OF_CONTESTANTS)
        fathers_array = [population[index] for index in fathers_indices_array]
        mothers_indices_array = get_best_fitted_parents(copy(population), TOURNAMENT_LIMIT, NUMBER_OF_CONTESTANTS, fathers_indices_array)
        mothers_array = [population[index] for index in mothers_indices_array]

        for i in range(len(fathers_array)):
            logger.info(f"Current reproduction cycle: # {i}")
            child_1, child_2 = reproduce(copy(fathers_array[i]), copy(mothers_array[i]))
            children.append(child_1)
            children.append(child_2)

        if MUTATION_PERCENTAGE > 0:
            logger.info("Mutating")
            children = mutate(copy(children), MUTATION_PERCENTAGE)
        
        if APPLY_ELITISM:
            logger.info("Applying elitism")
            elitist_array = population + children
            children = elitism(elitist_array)

        logger.debug(f"$$ Children #{generation} size: {len(children)}, values: \n {children}")
        population = copy(children)
        genome_fitness_array = generate_populations_fitness(copy(reference_function), copy(population))
        logger.debug(f"@@@@ FA: {genome_fitness_array}")
        best_child_index = np.argmin(genome_fitness_array)
        best_fit_child_value = genome_fitness_array[best_child_index]
        best_fit_list.append(best_fit_child_value)
        ax1.clear()
        ax1.set_title("Functions")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.plot(reference_function, label="Target")
        ax1.plot(get_function_from_cromosome(copy(population[best_child_index])), linestyle='dashed', label="Current best fit")
        ax2.clear()
        ax2.set_title("Best fit per generation")
        ax2.plot(generation_list, best_fit_list)
        plt.show(block = False)
        plt.pause(0.2)
