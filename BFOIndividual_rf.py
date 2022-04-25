import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# import ObjFunction


class BFOIndividual_rf:

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
        y = df.iloc[:, 1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10, shuffle=True)

        rfc = RandomForestClassifier(n_estimators=self.chrom[0].astype(int), criterion='gini', max_depth=None, bootstrap=True)
        rfc.fit(x_train, y_train)

        y_predict_test = rfc.predict(x_test)
        acc_test = metrics.accuracy_score(y_test, y_predict_test)


        self.fitness = 1-acc_test