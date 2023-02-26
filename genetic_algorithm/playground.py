import itertools
import numpy as np
import matplotlib.pyplot as plt


test_dict = {
    
    2: 90,
    3: 60,
    4: 30,
    1: 70,
    5: 100
}

#new_dict = dict(itertools.islice(sorted(test_dict.items(), key=lambda item: item[1]), 5))

#print(new_dict)

A = 8
B = 25
C = 4
D = 45
E = 10
F = 17
G = 35

x = np.arange(0, 100, 0.1)  # crea un array de valores de x de 0 a 100 en incrementos de 0.1
y = A * (B * np.sin(x / C) + D * np.cos(x / E)) + F * x - G

print("y", y)
#input()

#plt.plot(x, y)
#plt.show()


array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

resulting_array = np.concatenate((array1, array2))
print(resulting_array)
