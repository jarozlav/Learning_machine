from Backpropagation import Backpropagation
from Matriz.Matriz import Matriz
from Matriz.MatrizMath import MatrizMath
from Matriz.Tools.Tools import Tools
#from colorama import init, Fore
#from termcolor import colored

class Run:
    inputs = 35
    hidden = 18
    output = 10
    name_file = "prueba.txt"
    
    warning = 'red'
    
    __TOOLS = None

    def __init__(self):
        self.__TOOLS = Tools()
        self.banner()
        self.run()
        self.bye()
        
    def banner(self):
        banner = '\tBienvenido\n'
        banner += 'Red tipo: \tBackpropagation\n'
        banner += 'Capas: \t\t3\n'
        banner += 'version: \t1.0\n'
        banner += 'Reconoce: \tPatrones\n'
        banner += 'Patrones: \t'+self.name_file+'\n'
        banner += 'Autor: \t\n'
        print banner
    
    def bye(self):
        bye = 'Borrando pesos\n'
        bye += 'Red apagada'
        print bye
        
    def run(self):
        lines = self.__TOOLS.getLinesinArray(self.name_file)
        print "Los datos han sido cargados de: " + self.name_file
        self.names, patterns = self.__TOOLS.getNamesAndPatterns(lines)
        self.whileTrain(patterns, self.inputs, self.hidden, self.output)
        self.whileRecognize()
        
    def whileTrain(self, patterns, inputs, hiddens, outputs):
        self.red = Backpropagation(Matriz(patterns), inputs, hiddens, outputs)
        train = self.inputDataTrain()
        self.red.trainNet(train[0], train[1], train[2])
        
    def inputDataTrain(self):
        train = []
        learnRate = self.getInputLearnRate()
        train.append(learnRate)
        errorRate = self.getInputErrorRate()
        train.append(errorRate)
        iterations = self.getInputIterations()
        train.append(iterations)
        return train

    def getInputIterations(self):
        iterations = self.__TOOLS.getInputNumber("Cuantas iteraciones realizara?")
        return int(iterations)

    def getInputErrorRate(self):
        return self.__TOOLS.getInputDecimal("Cual es el margen de error?")

    def getInputLearnRate(self):
        return self.__TOOLS.getInputDecimal("Cual es el rango de aprendizaje")
    
    def whileRecognize(self):
        _next = True
        while(_next):
            pattern = self.__TOOLS.inputToList(35)
            tmp = Matriz(pattern)
            resp = self.red.simulation_net(tmp)
            tmp_resp = MatrizMath.transpose(resp)
            print "Resultados segun la red"
            for index in range(len(self.names)):
                print self.names[index] + ": "+ str(tmp_resp.get(0, index))
            winner = MatrizMath.moreProbable(tmp_resp.getRow(0), self.names)
            print "El patron reconocido es: "+ winner
            _next = self.__TOOLS.getNext()
        

