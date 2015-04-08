#!/usr/bin/python
# -*- coding: utf-8 -*-

from Matriz.Matriz import Matriz
from Matriz.MatrizMath import MatrizMath
from numpy import *
from NormalizeType import NormalizeType
from NormalizeInput import NormalizeInput

class SelfOrganizingMap:
    
    __INPUTNEURONCOUNT = 0
    __OUTPUTNEURONCOUNT = 0
    __OUTPUTWEIGHTS = None
    __OUTPUT = None
    __NORMALIZATIONTYPE = None

    def __init__(self, inputCount, outputCount, tipe = 'MULTIPLICATIVE'):
        self.__INPUTNEURONCOUNT = inputCount #int
        self.__OUTPUTNEURONCOUNT = outputCount #int
        self.__OUTPUTWEIGHTS = Matriz(self.__OUTPUTNEURONCOUNT, self.__INPUTNEURONCOUNT + 1)#Matriz
        self.__OUTPUT = [0.0 for i in range(self.__OUTPUTNEURONCOUNT)] #list
        try:
            norm_type = NormalizeType(tipe)
        except Exception as _except:
            raise
        self.__NORMALIZATIONTYPE = norm_type.getNormalizeType()
    
    #Normaliza el patron y obtiene la neurona ganadora
    def winner(self, pattern):
        normalize = NormalizeInput(pattern, self.__NORMALIZATIONTYPE)
        return self.winner_(normalize)

    #Obtiene la neurona ganadora
    def winner_(self, normalizeInput):
        win = 0
        biggest = 0.0000000000001E-1022
        for i in range(self.__OUTPUTNEURONCOUNT):
            matriz = self.__OUTPUTWEIGHTS.getRow(i)
            self.__OUTPUT[i] = MatrizMath.dotProduct(normalizeInput.getInputMatriz(), matriz) * normalizeInput.getNormFact()
            self.__OUTPUT[i] = (self.__OUTPUT[i] + 1.0)/ 2.0
            if(self.__OUTPUT[i] > biggest):
                biggest = self.__OUTPUT[i]
                win = i
            if(self.__OUTPUT[i] < 0):
                self.__OUTPUT[i] = 0
            if(self.__OUTPUT[i] > 1):
                self.__OUTPUT[i] = 1
        return win;

    def getInputNeuronCount(self):
        return self.__INPUTNEURONCOUNT

    def getNormalizationType(self):
        return self.__NORMALIZATIONTYPE

    def getOutput(self):
        return self.__OUTPUT

    def getOutputNeuronCount(self):
        return self.__OUTPUTNEURONCOUNT

    def getOutputWeights(self):
        return self.__OUTPUTWEIGHTS

    def setOutputWeights(self, outputWeights):
        self.__OUTPUTWEIGHTS = ouputWeights

    def setAttribute(self, item, data):
        globals()["__"+item.upercase()] = data
        
    def getAttribute(self, item, data):
        return globals()["__"+item.upercase()];