import math
import numpy as np
from random import randint
from typing import List

'''
Genetic Algorithm
Elaborated by Eduardo Alberto Rodriguez Garcia

- Each city is represented by coordinates (X, Y) and is known as a gene
- Each row is a cromosome
- All the data together is known as a population

[(8, 9), (8, 6), (6, 5), (1, 5), (6, 8)]
[(7, 2), (6, 7), (5, 8), (5, 4), (9, 9)]
[(9, 2), (1, 1), (3, 3), (6, 8), (3, 9)]
[(2, 4), (9, 1), (2, 1), (4, 6), (8, 7)]
........
[(9, 2), (1, 1), (3, 3), (6, 8), (3, 9)]

'''


class City(object):
    '''
    Class to represent the each city and a way to randomly
    generate each X and Y point
    '''

    def __init__(self):
        self.point = [self.generate_random_point(), self.generate_random_point()]

    def generate_random_point(self) -> int:
        return randint(1, 9)

    def __repr__(self) -> str:
        return f"{self.point}"


def generate_cromosome(size: int):
    return [City() for _ in range(size)]


def generate_population(population_size: int, cromosome_size: int):
    return [generate_cromosome(cromosome_size) for _ in range(population_size)]


def calculate_total_distance_between_cities_from_same_cromosome(cromosome_list) -> float:
    cromosome_size = len(cromosome_list)
    total_distance = 0
    for i in range(cromosome_size - 1):
        print(f"Getting point at index: {i} with value: {cromosome_list[i].point}")
        total_distance += math.dist(cromosome_list[i].point, cromosome_list[i + 1].point)
    return total_distance


def get_contestants_list(population_list: List[object], contestants_index: List[int]):
    return [population_list[index] for index in contestants_index]


def main():
    cromosome_list = List[City]
    population_list = List[cromosome_list]
    population_size = 100
    cromosome_size = 5
    shortest_distance = 0
    arbitrary_contestants_number = 5
    population_list = generate_population(population_size, cromosome_size)
    print(*population_list, sep = "\n")
    print(f"#### DEBUG")

    # Competition to select the best route
    for _ in range(population_size):
        # Selects 5 cromosomes(rows) contestants from the whole population
        contestants_index = np.random.permutation(population_size)[:arbitrary_contestants_number].tolist()
        #contestants_index = [randint(0, population_size - 1) for _ in range(arbitrary_contestants_number)]
        print("### Contestants index ###")
        print(contestants_index)
        
        contestants_list = get_contestants_list(population_list, contestants_index)
        print("### Contestants List ###")
        print(contestants_list, sep = "\n")
        # Using zip(keys, values) to merge 2 lists into a dict. E.g.: { 9: [9, 1, 3], 8: [6, 5, 2]}
        # where the key is the index in the population and the value is a the list of the cromosome
        contestant_dict = dict(zip(contestants_index, contestants_list))
        winner = [float('inf'), 0]
        
        # The key is the index in the population list
        # The value = population_list[index]; the value of the cromosome at index specified
        for key, value in contestant_dict.items():
            total_cities_distance = calculate_total_distance_between_cities_from_same_cromosome(value)
            print(f"@@@ Evaluating distance betwen cities for contestang at index: {key}, distance value: {total_cities_distance}")
            if total_cities_distance < winner[0]:
                print(f"@@@@ New winner found at index: {key}, with value: {total_cities_distance}")
                winner[0] = total_cities_distance
                winner[1] = key
        
        print(f"### Mutating child and swapping place with parent at {winner[1]} ###")
        
    
    
    
    


if __name__ == "__main__": 
    main()