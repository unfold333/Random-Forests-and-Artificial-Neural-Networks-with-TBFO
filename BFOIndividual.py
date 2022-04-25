import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
# import ObjFunction


class BFOIndividual:

    '''
    individual of baterial clony foraging algorithm
    '''

    def __init__(self,  vardim, bound):
        '''
        vardim: dimension of variables
        bound: boundaries of variables
        '''
        self.vardim = vardim
        self.bound = bound
        self.fitness = 0.
        self.trials = 0

    def generate(self):
        '''
        generate a random chromsome for baterial clony foraging algorithm
        '''
        len = self.vardim
        rnd = np.random.random(size=len)
        self.chrom = np.zeros(len,dtype = int)
        for i in range(0, len):
            self.chrom[i] = self.bound[0, i] + \
                (self.bound[1, i] - self.bound[0, i]) * rnd[i]

    def calculateFitness(self):
        '''
        calculate the fitness of the chromsome
        '''
        # self.fitness = ObjFunction.GrieFunc(
        #     self.vardim, self.chrom, self.bound)

        df = pd.read_csv('codon1.csv', low_memory=False)
        x = df.iloc[:, 5:]
        y = df.iloc[:,0]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10, shuffle=True)

        ### 神经网络
        # print('神经网络')
        clf=MLPClassifier(solver='adam', activation='relu',hidden_layer_sizes=(self.chrom.astype(int)),random_state=1,max_iter=5000)
        clf.fit(x_train, y_train)
        acc_test=clf.score(x_test,y_test)

        self.fitness = 1-acc_test