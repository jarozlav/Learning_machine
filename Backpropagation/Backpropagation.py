from random import *
from Matriz.Matriz import Matriz
from Matriz.MatrizMath import MatrizMath

class Backpropagation:

    def __init__(self, patterns = None,  inputCount = 0, hiddenCount = 0, outputCount = 0):
        if((inputCount > 0) and (hiddenCount > 0) and (outputCount > 0)):
            self.inputCount = inputCount
            self.hiddenCount = hiddenCount
            self.outputCount = outputCount
        else:
            raise Exception("Las neuronas de las apas deben ser mayor a 0 (cero)")
        self.patterns = MatrizMath.transpose(patterns)
        self.WInputHidden = Matriz([[0 for c in range(self.inputCount)] for r in range(self.hiddenCount)])
        self.WHiddenOutput = Matriz([[0 for c in range(self.hiddenCount)] for r in range(self.outputCount)])
        self.identy = MatrizMath.identity(self.getOutputCount())
        self.identy = MatrizMath.transpose(self.identy)

    def getInputCount(self):
        return self.inputCount

    def getHiddenCount(self):
        return self.hiddenCount

    def getOutputCount(self):
        return self.outputCount

    def getPatterns(self):
        return self.patterns

    def getWeightInput(self):
        return self.WInputHidden

    def getWeightOutput(self):
        return self.WHiddenOutput

    def getIterations(self):
        return self.iterations

    def setWeights(self, weightInput, weightOutput):
        self.setWeightInput(weightInput)
        self.setWeightOutput(weightOutput)
        
    def setWeightInput(self, weightInput):
        self.WInputHidden = weightInput

    def setWeightOutput(self, weightOutput):
        self.WHiddenOutput = weightOutput

    def setIterations(self, iterations):
        self.iterations = iterations

    def setPatterns(self, pattern):
        self.patterns = patterns

    def initialization(self):
        self.randomize(self.WInputHidden)
        self.randomize(self.WHiddenOutput)

    def simulation_net(self, pattern):
        netInput = MatrizMath.cross_multiply(self.getWeightInput(), pattern)
        deltanetInput = MatrizMath.matriz_sigmoidal(netInput)
        netOutput = MatrizMath.cross_multiply(self.getWeightOutput(), deltanetInput)
        return MatrizMath.matriz_sigmoidal(netOutput)

    def trainNet(self, learnRate, error, iterations):
        print ("Inicio de entrenamiento")
        self.initialization()
        iteration = 0
        globalErrors = [0 for i in range(self.getOutputCount())]
        haveError = True
        self.setIterations(0)
        while((iteration < iterations) and haveError):
            haveError = False
            for i in range(self.getPatterns().getColumns()):
                netInput = MatrizMath.cross_multiply(self.getWeightInput(), self.getPatterns().getColumn(i))
                fnetInput = MatrizMath.matriz_sigmoidal(netInput)
                netOutput = MatrizMath.cross_multiply(self.getWeightOutput(), fnetInput)
                fnetOutput = MatrizMath.matriz_sigmoidal(netOutput)
                outputErrors = MatrizMath.substract(self.identy.getColumn(i), fnetOutput)
                sqrtError = MatrizMath.squartError(outputErrors)
                globalErrors[i] = sqrtError
                if(sqrtError > error):
                    haveError = True
                    netDeltaInput = MatrizMath.matriz_delta(netInput)
                    netDeltaOutput = MatrizMath.matriz_delta(fnetOutput)
                    deltaOutput = MatrizMath.multiply(outputErrors, netDeltaOutput)
                    WOutputT = MatrizMath.transpose(self.getWeightOutput())
                    temp = MatrizMath.cross_multiply(WOutputT, deltaOutput)
                    deltaInput = MatrizMath.multiply(temp, netDeltaInput)
                    m_WO = MatrizMath.cross_multiply(deltaOutput, MatrizMath.transpose(fnetInput))
                    deltaWOutput = MatrizMath.scalar_multiply(m_WO, learnRate)
                    self.setWeightOutput(MatrizMath.add(self.getWeightOutput(), deltaWOutput))
                    m_WI = MatrizMath.cross_multiply(deltaInput, MatrizMath.transpose(self.getPatterns().getColumn(i)))
                    deltaWInput = MatrizMath.scalar_multiply(m_WI, learnRate)
                    self.setWeightInput(MatrizMath.add(self.getWeightInput(), deltaWInput))
            iteration += 1
        self.setIterations(iteration)
        globalError = 0
        print("La red ha sido entrenada")
        print("Iteraciones: "+ str(self.getIterations()))
        if not haveError:
            for i in range(len(globalErrors)):
                globalError += globalErrors[i]
            print("Valor de error alcanzado")
            print("Error global: "+ str(globalError))
        else:
            print("Valor de error no alcanzado")

    def randomize(self, matriz):
        for r in range(matriz.getRows()):
            for c in range(matriz.getColumns()):
                self.set(matriz, r, c, uniform(-1,1))

    def set(self, matriz, row, column, value):
        matriz.set(row, column, value)
