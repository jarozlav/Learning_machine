#!/usr/bin/python
# -*- coding: utf-8 -*-

from Matriz.Tools.Tools import Tools
from Matriz.Matriz import Matriz
from Matriz.MatrizMath import MatrizMath
from NormalizeType import NormalizeType



class NormalizeInput:
    
    __TOOLS = None
    __NORMALIZE = None
    __INPUTMATRIZ = None
    
    def __init__(self, pattern, tipe):
        self.__TOOLS = Tools()
        '''El pattern debe ser un array '''
        if(not (type(pattern[0]) in self.__TOOLS.getNumbers())):
            raise Exception("El patron debe ser un vector")
        self.__NORMALIZE = NormalizeType(tipe)
        self.calculateFactors(pattern)
        self.__INPUTMATRIZ = self.createInputMatriz(pattern, self.getSynth())

    #Crea un vector de tipo matriz con un elemento extra
    def createInputMatriz(self, pattern, extra):
        m= Matriz(1, len(pattern)+1)
        for i in range(len(pattern)):
            m.set(0, i, pattern[i])
        m.set(0, len(pattern), extra)
        return m

    #Crea los factores de calculo para la normalizacion
    def calculateFactors(self, pattern):
        inputMatriz= Matriz.createColumnMatriz(pattern)
        lenMatriz = MatrizMath.vectorLength(Matriz(pattern))
        lenMatriz = max(lenMatriz, 1.E-30)
        numInputs = len(pattern)
        if(self.__NORMALIZE.getNormalizeType() == 'MULTIPLICATIVE'):
            self.normfac = 1.0/ lenMatriz
            self.synth = 0.0
        else:
            self.normfac = 1.0/math.sqrt(numInputs)
            d = numInputs - math.pow(lenMatriz, 2)
            if(d > 0):
                self.synth = math.sqrt(d) * self.normfact
            else:
                self.synth = 0.0

    def getNormFact(self):
        return self.normfac

    def getSynth(self):
        return self.synth

    def getInputMatriz(self):
        return self.__INPUTMATRIZ

    def getNormalizeType(self):
        return self.__NORMALIZE.getNormalizeType()
    
    def setAttribute(self, item, data):
        globals()["__"+item.upercase()] = data
        
    def getAttribute(self, item, data):
        return globals()["__"+item.upercase()];
