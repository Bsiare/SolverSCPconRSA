import numpy as np
import random

def iterarRSA(maxIter, t, dimension, poblacion, bestSolution):

    Alpha = 0.1
    Beta = 0.005

    for i in range(len(poblacion)):
        ES = 2 * random.randint(-1, 1) * (1 - (t / maxIter)) 
        for j in range(dimension):
            R = bestSolution[j] - poblacion[i][j] / (bestSolution[j] + np.finfo(float).eps)
            P = Alpha + (poblacion[i][j] - np.mean(poblacion[i])) / (bestSolution[j] * (1 - 0) + np.finfo(float).eps)
            Eta = bestSolution[j] * P

            if t < maxIter / 4:
                poblacion[i][j] = bestSolution[j] - Eta * Beta - R * random.random()
            elif maxIter / 4 <= t < 2 * maxIter / 4:
                poblacion[i][j] = bestSolution[j] * poblacion[random.randint(0, len(poblacion) - 1)][j] * ES * random.random()
            elif 2 * maxIter / 4 <= t < 3 * maxIter / 4:
                poblacion[i][j] = bestSolution[j] * P * random.random()
            else:
                poblacion[i][j] = bestSolution[j] - Eta * np.finfo(float).eps - R * random.random()


            poblacion[i][j] = max(0, min(1, poblacion[i][j]))

    return np.array(poblacion)

