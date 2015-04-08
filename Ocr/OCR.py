
from Som.Matriz.Tools.Tools import Tools
from numpy import *
from random import random
import threading
from Som.SelfOrganizingMap import SelfOrganizingMap
from Som.TrainSelfOrganizingMap import TrainSelfOrganizingMap
from Som.NormalizeType import NormalizeType
from Som.NormalizeInput import NormalizeInput
from Som.LearningMethod import LearningMethod
from Som.Matriz.Matriz import Matriz
from Som.Matriz.MatrizMath import MatrizMath
from SampleData import SampleData
from Tools_ocr import Tools_ocr


class OCR():
    name_file = "sample.dat"
    DOWNSAMPLE_WIDTH = 5
    DOWNSAMPLE_HEIGHT = 7
    MAX_ERROR = 0.01
    halt = False
    List_letters = []
    trainThread = None
    net = None
    
    __TOOLS = None
    __TOOLS_OCR = None
    
    def __init__(self):
        self.__TOOLS = Tools()
        self.__TOOLS_OCR = Tools_ocr()
        self.banner()
        self.run()
        self.bye()
        
    #Mensaje de bienvenida y datos de la red
    def banner(self):
        banner = '\tBienvenido\n'
        banner += 'Red tipo: \tSelfOrganizingMap\n'
        banner += 'Capas: \t\t2\n'
        banner += 'version: \t1.0\n'
        banner += 'Reconoce: \tPatrones\n'
        banner += 'Patrones: \t'+self.name_file+'\n'
        banner += 'Autor: \t\n'
        print banner
    
    #Despedida de la red
    def bye(self):
        bye = 'Borrando pesos\n'
        bye += 'Red apagada'
        print bye

    #Inicializa la actividad de la red
    def run(self):
        try:
            lines = self.__TOOLS.getLinesinArray(self.name_file)
            self.List_letters = self.__TOOLS_OCR.toListLetters(lines, self.DOWNSAMPLE_WIDTH, self.DOWNSAMPLE_HEIGHT)
            print "Los datos han sido cargados de: " + self.name_file
            self.train()
            self.recognize()
        except Exception as _except:
            raise _except
    
    #Trabaja con hilos
    #def train(self):
    #    print "Fase de entrenamiento"
    #    if(self.trainThread == None):
    #        self.run()
    #    else:
    #        self.halt = True

    #Fase de entrenamiento
    def train(self):
        try:
            print "Fase de entrenamiento"
            base = self.List_letters[0]
            inputNeuron = base.getRows() * base.getColumns()
            outputNeuron = len(self.List_letters)
            self.net = SelfOrganizingMap(inputNeuron, outputNeuron, "MULTIPLICATIVE")
            _set = self.__TOOLS_OCR.sampleDataToMatriz(self.List_letters)
            train = TrainSelfOrganizingMap(self.net, _set, "SUBSTRACTIVE", 0.5)
            self.whileTrain(train)
        except Exception as _except:
            raise _except
    
    #Ciclo de entrenamiento
    def whileTrain(self, train):
        tries = 1
        train.iteration()
        while((train.getTotalError() > self.MAX_ERROR) and not self.halt):
            self.update(tries, train.getTotalError(), train.getBestError())
            train.iteration()
            tries += 1
        self.halt = True
        self.update(tries, train.getTotalError(), train.getBestError())

    #Imprime en pantalla los datos
    def update(self, tries, totalError, bestError):
        if(self.halt):
            self.trainThread = None
            print "El entraenamiento ha sido completado"
        print "Los intentos son: "+str(tries)
        print "El error total es: "+str(totalError)
        print "El mejor error es: "+str(bestError)

    #Fase de reconocimiento
    def recognize(self):
        print "Fase de reconocimiento"
        if(self.net == None):
            print "La red debe ser entrenada primero"
            return
        self.whileRecognize()

    def whileRecognize(self):
        _next = True
        while(_next):
            pattern = self.__TOOLS.inputToList(35)
            pattern = array(pattern)
            best = self.net.winner(pattern)
            _map = self.mapNeuron()
            print "La letra es: "+ str(_map[best])+" y la neurona activada es: "+str(best)
            _next = self.__TOOLS.getNext()

    #Crea el mapa de neuronas de la red SOM
    def mapNeuron(self):
        return self.makeMapNeuron(self.DOWNSAMPLE_WIDTH, self.DOWNSAMPLE_HEIGHT, self.List_letters)
    
    #Crea el mapa de neuronas de las letras de los patrones
    def makeMapNeuron(self, width, height, listletters):
        _map = ['?' for y in range(len(listletters))]
        for i in range(len(listletters)):
            sample = listletters[i]
            _input = self.__TOOLS_OCR.boolToDecimalList(sample)
            best = self.net.winner(_input)
            _map[best] = sample.getLetter()
        return _map
        

