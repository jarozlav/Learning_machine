from LearningMethod import LearningMethod
from Matriz.Matriz import Matriz
from Matriz.MatrizMath import MatrizMath
from SelfOrganizingMap import SelfOrganizingMap
from NormalizeInput import NormalizeInput
from numpy import *

class TrainSelfOrganizingMap:
    '''
    Esta clase permite entrenar la red con el algoritmo
    Kohonen
    '''
    
    
    #Atributos privados de la clase
    __REDUCTION = 0.99 #Reductor de aprendizaje
    __SOM = None #Red SOM
    __TRAIN = None #Matriz a entrenar (ndarray)
    __LEARNMETHOD = None #Metodo de aprendizaje (Additive o substractive)
    __LEARNRATE = 0.9 #Rango de aprendizaje 
    __OUTPUTNEURONCOUNT = 0 #Neuronas de salida (Capa de salida)
    __INPUTNEURONCOUNT = 0 #Neuronas de entrada (Capa de entrada)
    __TOTALERROR = 1.0 #Error total 
    __BESTNET = None #Mejor red
    __WON = [] #Neuronas activadas o ganadoras
    __WORK = None #Matriz de trabajo
    __BESTERROR = 1.7976931348623157e+308 #Max_value
    __CORRECT = None #Matriz
    __GLOBALERROR = 0.0 #

    
    #Constructor de la clase
    def __init__(self, som, train, learnMethod, learnRate):
        self.__SOM = som
        self.__TRAIN = train
        self.__LEARNMETHOD= LearningMethod(learnMethod)
        self.__LEARNRATE = learnRate
        self.__OUTPUTNEURONCOUNT = som.getOutputNeuronCount()
        self.__INPUTNEURONCOUNT = som.getInputNeuronCount()
        #self.__TOTALERROR = 1.0
        for tset in range(len(train)):
            dptr =Matriz.createColumnMatriz(train[tset])
            if(MatrizMath.vectorLength(dptr) < 1.E-30):
                raise Exception("El entrenamiento multiplicativo es un caso inoperable")
        self.__BESTNET = SelfOrganizingMap(self.__INPUTNEURONCOUNT, self.__OUTPUTNEURONCOUNT, self.__SOM.getNormalizationType())
        self.__WON = [0 for y in range (self.__OUTPUTNEURONCOUNT)]
        self.__CORRECT = Matriz(self.__OUTPUTNEURONCOUNT, self.__INPUTNEURONCOUNT + 1)
        if(self.__LEARNMETHOD.getLearningMethod == "ADDITIVE"):
            self.__WORK = Matriz(1, self.__INPUTNEURONCOUNT + 1)
        else:
            self.__WORK = None
        self.initialize()
        #self.__BESTERROR = 1.7976931348623157e+308 #Max_value

    def adjustWeights(self):
        for i in range(self.__OUTPUTNEURONCOUNT):
            if(self.__WON[i] == 0):
                continue
            f = 1.0 / self.__WON[i]
            if(self.__LEARNMETHOD.getLearnMethod() == "SUBSTRACTIVE" ):
                f *= self.__LEARNRATE
            length = 0.0
            for j in range(self.__INPUTNEURONCOUNT +1 ):
                corr = f * self.__CORRECT.get(i, j)
                self.__SOM.getOutputWeights().add(i, j, corr)
                length += corr * corr

    def copyWeights(self, source, target):
        MatrizMath.copy(source.getOutputWeights(), target.getOutputWeights())

    def evaluateErrors(self):
        self.__CORRECT.clear()
        for i in range(len(self.__WON)):
            self.__WON[i] = 0
        self.__GLOBALERROR = 0.0
        for tset in range(len(self.__TRAIN)):
            input = NormalizeInput(self.__TRAIN[tset], self.__SOM.getNormalizationType())
            best = self.__SOM.winner_(input)
            self.__WON[best] += 1
            wptr = self.__SOM.getOutputWeights().getRow(best)
            length = 0.0
            diff = 0
            for i in range(self.__INPUTNEURONCOUNT):
                diff = self.__TRAIN[tset][i] * input.getNormFact() - wptr.get(0,i)
                length += diff * diff
                if(self.__LEARNMETHOD.getLearningMethod() == "SUBSTRACTIVE"):
                    self.__CORRECT.add(best, i, diff)
                else:
                    self.__WORK.set(0, i, self.__LEARNRATE * self.__TRAIN[tset, i] * input.getNormFact() + wptr.get(0, i))
            diff = input.getSynth() - wptr.get(0, self.__INPUTNEURONCOUNT)
            length += diff * diff
            if(self.__LEARNMETHOD.getLearningMethod() == "SUBSTRACTIVE"):
                self.__CORRECT.add(best, self.__INPUTNEURONCOUNT, diff)
            else:
                self.__WORK.set(0, self.__INPUTNEURONCOUNT, self.learRate * input.getSynth() + wptr.get(0, self.__INPUTNEURONCOUNT))
            if(length > self.__GLOBALERROR):
                self.__GLOBALERROR = length
            if(self.__LEARNMETHOD.getLearningMethod() == "ADDITIVE"):
                self.normalizeWeight(self.__WORK, 0)
                for i in range(self.__INPUTNEURONCOUNT + 1):
                    self.__CORRECT.add(best, i, self.__WORK.get(0, i) - wptr.get(0, i))
        self.__GLOBALERROR = math.sqrt(self.__GLOBALERROR)
    
    def forceWin(self):
        best = which = 0
        outputWeights = self.__SOM.getOutputWeights()
        dist = 1.7976931348623157e+308
        for tset in range(len(self.__TRAIN)):
            best = self.__SOM.winner(self.__TRAIN[tset])
            output = self.__SOM.getOutput()
            if(output[best] < dist):
                dist = output[best]
                which = tset
        input =NormalizeInput(self.__TRAIN[which], self.__SOM.getNormalizationType())
        best = self.__SOM.winner_(input)
        output = self.__SOM.getOutput()
        dist = 4.9e-324
        i = self.__OUTPUTNEURONCOUNT
        while(i > 0):
            i -= 1
            if(self.__WON[i] != 0):
                continue
            if(output[i] > dist):
                dist = output[i]
                which = i
        for j in range(input.getInputMatriz().getColumns()):
            outputWeights.set(which, j, input.getInputMatriz().get(0, j))
        self.normalizeWeight(outputWeights, which)

    def getBestError(self):
        return self.__BESTERROR

    def getTotalError(self):
        return self.__TOTALERROR

    def initialize(self):
        self.__SOM.getOutputWeights().randomize(-1, 1)
        for i in range(self.__OUTPUTNEURONCOUNT):
            self.normalizeWeight(self.__SOM.getOutputWeights(), i)
        
    def iteration(self):
        self.evaluateErrors()
        self.__TOTALERROR = self.__GLOBALERROR
        if(self.__TOTALERROR < self.__BESTERROR):
            self.__BESTERROR = self.__TOTALERROR
            self.copyWeights(self.__SOM, self.__BESTNET)
        winners = 0
        for i in range(len(self.__WON)):
            if(self.__WON[i] != 0):
                winners += 1
        if((winners < self.__OUTPUTNEURONCOUNT) and (winners < len(self.__TRAIN))):
            self.forceWin()
            return
        self.adjustWeights()
        if(self.__LEARNRATE > 0.01):
            self.__LEARNRATE *= self.__REDUCTION
        
        
    def normalizeWeight(self, matriz, row):
        _len = MatrizMath.vectorLength(matriz.getRow(row))
        _len = max(_len, 1.E-30)
        _len = 1.0 / _len
        for i in range(self.__INPUTNEURONCOUNT):
            matriz.set(row, i, matriz.get(row, i) * _len)
        matriz.set(row, self.__INPUTNEURONCOUNT, 0)

    def setAttribute(self, item, data):
        globals()["__"+item.upercase()] = data
        
    def getAttribute(self, item, data):
        return globals()["__"+item.upercase()];