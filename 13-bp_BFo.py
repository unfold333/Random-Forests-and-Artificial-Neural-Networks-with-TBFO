import numpy as np
from BFOIndividual import BFOIndividual
import random
import copy
import matplotlib.pyplot as plt
import math
import scipy.stats as ss


class BacterialForagingOptimization:

    '''
    The class for baterial foraging optimization algorithm
    '''

    def __init__(self, sizepop, vardim, bound, params):
        '''
        sizepop: population sizepop
        vardim: dimension of variables
        bound: boundaries of variables
        param: algorithm required parameters, it is a list which is consisting of [Ned, Nre, Nc, Ns, C, ped, d_attract, w_attract, d_repellant, w_repellant]
        '''
        self.sizepop = sizepop
        self.vardim = vardim
        self.bound = bound
        self.population = []
        self.bestPopulation = []
        self.accuFitness = np.zeros(self.sizepop)
        self.fitness = np.zeros(self.sizepop)
        self.params = params
        self.trace = np.zeros(
            (self.params[0] * self.params[1] * self.params[2], 2))

    def initialize(self):
        '''
        initialize the population
        '''
        for i in range(0, self.sizepop):
            ind = BFOIndividual(self.vardim, self.bound)
            ind.generate()
            self.population.append(ind)

    def evaluate(self):
        '''
        evaluation of the population fitnesses
        '''
        for i in range(0, self.sizepop):
            self.population[i].calculateFitness()
            self.fitness[i] = self.population[i].fitness

    def sortPopulation(self):
        '''
        sort population according descending order
        '''
        sortedIdx = np.argsort(self.accuFitness)
        newpop = []
        newFitness = np.zeros(self.sizepop)
        for i in range(0, self.sizepop):
            ind = self.population[sortedIdx[i]]
            newpop.append(ind)
            self.fitness[i] = ind.fitness
        self.population = newpop

    def solve(self):
        '''
        evolution process of baterial clony foraging algorithm
        '''
        self.t = 0
        self.initialize()
        self.evaluate()
        bestIndex = np.argmin(self.fitness)
        self.best = copy.deepcopy(self.population[bestIndex])

        for i in range(0, self.params[0]):
            for j in range(0, self.params[1]):
                for k in range(0, self.params[2]):
                    self.t += 1
                    self.chemotaxls()
                    self.evaluate()
                    best = np.min(self.fitness)
                    bestIndex = np.argmin(self.fitness)
                    if best < self.best.fitness:
                        self.best = copy.deepcopy(self.population[bestIndex])
                    self.avefitness = np.mean(self.fitness)
                    self.trace[self.t - 1, 0] = self.best.fitness
                    self.trace[self.t - 1, 1] = self.avefitness
                    print("Generation %d: optimal function value is: %f; average function value is %f" % (
                        self.t, self.trace[self.t - 1, 0], self.trace[self.t - 1, 1]))
                self.reproduction()
            self.eliminationAndDispersal()

        print("Optimal function value is: %f; " %
              self.trace[self.t - 1, 0])
        print("Optimal solution is:")
        print(self.best.chrom)
        self.printResult()

    def chemotaxls(self):
        '''
        chemotaxls behavior of baterials
        '''
        for i in range(0, self.sizepop):
            tmpInd = copy.deepcopy(self.population[i])
            self.fitness[i] += self.communication(tmpInd)
            Jlast = self.fitness[i]
            rnd = np.random.uniform(low=-1, high=1.0, size=self.vardim)
            phi = rnd / np.linalg.norm(rnd)
            # phi = phi.astype(int)

            if i>0:
                prng = np.random.RandomState(123456789)  # 定义局部种子
                t = prng.standard_t(i, size=1)
                t = float(t)# t 分布
                cc = self.params[4] + self.params[4] * t
                tmpInd.chrom = tmpInd.chrom + cc * phi
                self.params[4] = cc
            else:
                # phi = phi.astype(int)
                tmpInd.chrom =tmpInd.chrom + self.params[4] * phi
            for k in range(0, self.vardim):
                if tmpInd.chrom[k] < self.bound[0, k]:
                    tmpInd.chrom[k] = self.bound[0, k]
                if tmpInd.chrom[k] > self.bound[1, k]:
                    tmpInd.chrom[k] = self.bound[1, k]
            tmpInd.calculateFitness()
            m = 0
            while m < self.params[3]:
                if tmpInd.fitness < Jlast:
                    Jlast = tmpInd.fitness
                    self.population[i] = copy.deepcopy(tmpInd)
                    # print m, Jlast
                    tmpInd.fitness += self.communication(tmpInd)

                    # prng = np.random.RandomState(123456789)  # 定义局部种子
                    # t = prng.standard_t(i, size=1)
                    # t = int(t)  # t 分布
                    # cc = self.params[4] + self.params[4] * t
                    # tmpInd.chrom =tmpInd.chrom + cc * phi
                    # self.params[4] = cc

                    tmpInd.chrom =tmpInd.chrom + self.params[4] * phi

                    for k in range(0, self.vardim):
                        if tmpInd.chrom[k] < self.bound[0, k]:
                            tmpInd.chrom[k] = self.bound[0, k]
                        if tmpInd.chrom[k] > self.bound[1, k]:
                            tmpInd.chrom[k] = self.bound[1, k]
                    tmpInd.calculateFitness()
                    m += 1
                else:
                    m = self.params[3]
            self.fitness[i] = Jlast
            self.accuFitness[i] += Jlast

    def communication(self, ind):
        '''
        cell to cell communication
        '''
        Jcc = 0.0
        term1 = 0.0
        term2 = 0.0
        for j in range(0, self.sizepop):
            term = 0.0
            for k in range(0, self.vardim):
                term += (ind.chrom[k] -
                         self.population[j].chrom[k]) ** 2
            term1 -= self.params[6] * np.exp(-1 * self.params[7] * term)
            term2 += self.params[8] * np.exp(-1 * self.params[9] * term)
        Jcc = term1 + term2

        return Jcc

    def reproduction(self):
        '''
        reproduction of baterials
        '''
        self.sortPopulation()
        newpop = []
        for i in range(0, self.sizepop / 2):
            newpop.append(self.population[i])
        for i in range(self.sizepop / 2, self.sizepop):
            self.population[i] = newpop[i - self.sizepop / 2]

    def eliminationAndDispersal(self):
        '''
        elimination and dispersal of baterials
        '''
        for i in range(0, self.sizepop):
            rnd = random.random()
            if rnd < self.params[5]:
                self.population[i].generate()

    def printResult(self):
        '''
        plot the result of the baterial clony foraging algorithm
        '''
        x = np.arange(0, self.t)
        y1 = self.trace[:, 0]
        y2 = self.trace[:, 1]
        plt.plot(x, y1, 'r', label='optimal value')
        plt.plot(x, y2, 'g', label='average value')
        plt.xlabel("Iteration")
        plt.ylabel("function value")
        plt.title(
            "Baterial clony foraging algorithm for function optimization")
        plt.legend()
        plt.show()


bound = np.tile([[1], [100]], 2)
bfo = BacterialForagingOptimization(26, 2, bound,  [2, 4, 50, 4, 50, 0.25, 0.05, 0.05, 0.05, 0.05])
# [Ned, Nre, Nc, Ns, C, ped, d_attract, w_attract, d_repellant, w_repellant]
# Ned:消除-分散事件的次数（迁移代数）
# Nre:复制步骤的次数（繁衍代数）
# Nc:趋化步骤的步数
# Ns:游泳长度
# C:翻转过程中采取 指定的随机方向的步长的大小（运行长度单元）
# Ped:消除-分散概率
bfo.solve()



