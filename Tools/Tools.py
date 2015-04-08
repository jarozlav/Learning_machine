#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import *

class Tools:
    '''
    Esta clase contiene las utilerias mas utilizadas
    '''
    
    __INTEGERS = [int, int32]
    __FLOATS = [double, float, float64]
    __BOOLS = [bool_, bool]
    __ARRAYS = [ndarray, list]
    __NUMBERS = [int, int32, double, float, float64]
    
    #Retorna los tipos de datos enteros
    def getIntegers(self):
        return self.__INTEGERS
    
    #Retorna los tipos de datos flotantes
    def getFloats(self):
        return self.__FLOATS

    #Retorna los tipos de datos enteros y flotantes    
    def getNumbers(self):
        return self.__NUMBERS
    
    #Retorna los tipos de datos booleanos
    def getBools(self):
        return self.__BOOLS
    
    #Retorna los tipos de datos arrays o listas
    def getArrays(self):
        return self.__ARRAYS
    
    #Lines = [[A][10101010101010101010101010101010101],
    #       [B][10101010101010101010101010101010101]]
    def getLinesinArray(self, nameFile):
        import os
        absolute = self.__back_dir(os.path.dirname(__file__))
        relative = 'Datos/'+ nameFile
        path = os.path.join(absolute, relative)
        return [(l.strip()).split(":") for l in (open(path).readlines())]
    
    def __back_dir(self, path, back = 1):
        list_path = path.split('/')
        new_path = ''
        length = len(list_path) - back
        for index in range(length):
            new_path += list_path[index] + '/'
        return new_path
    
    
    #Input = [10101010101010101010101010101010101]
    def inputToList(self, lenght = 35):
        _input = self.__an_input(lenght)
        input_ = []
        for i in _input:
            input_.append(float(i))
        return input_
    
    #Input = "10101010101010101010101010101010101"
    def __an_input(self, lenght = 35):
        correct = False
        _input = ""
        while(not correct):
            _input = self.getInput("Ingresa el patron a reconocer, solo 0's y 1's")
            if(len(_input) == lenght):
                correct = True
            else:
                print "Ingresa una cadena de 1's y 0's que representen el patron de 35 digitos"
        return _input
    
    def getNext(self):
        correct = False
        _input = ""
        valid = ['Si', 'S', 'si', 's', 'No', 'N', 'no', 'n']
        yes = ['Si', 'S', 'si', 's']
        no = ['No', 'N', 'no', 'n']
        while(not correct):
            _input = self.getInput("Quieres reconocer otro patron?")
            if(_input in valid):
                if(_input in yes):
                    return True
                else:
                    return False
            else:
                print "Ingresa alguna de estas opciones"
                print str(valid)
                correct = False
        
    
    def getInput(self, message):
        print message
        return raw_input()
    
    def getInputDecimal(self, message):
        decimal = 0
        correct = False
        while(not correct):
            decimal = self.getInputNumber(message)
            if(decimal > 0.0 and decimal <= 1.0):
                correct = True
            else:
                correct = False
                print("Ingresa un numero > 0 y <= 1")
        return decimal
        

    def getInputNumber(self, message):
        correct = False
        number = 0
        while(not correct):
            number_ = self.getInput(message)
            try:
                number = float(number_)
                correct = True
            except Exception as _except:
                correct = False
            if not correct:
                print "No ingresaste un digito\n Intenta otra vez"
        return number
    
    def getNamesAndPatterns(self, lines):
        listletters = []
        patterns = []
        for line in lines:
            listletters.append(line[0])
            patterns.append(self.textToList(line[1]))
        return listletters, patterns
    
    def textToList(self, texts):
        _list = []
        for text in texts:
            _list.append(text)
        return _list
