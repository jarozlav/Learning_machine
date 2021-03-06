#!/usr/bin/python
# -*- coding: utf-8 -*-

#Importacion de la librerias
from Tools.Tools import Tools
from numpy import *
from random import random

#Definicion de la clase
class Matriz:
    '''
    Esta clase sirve como objeto para la manipulacion
    de las tablas de pesos y patrones de los enlaces de las capas
    '''

    #Se definen los tipos para validaciones
    #Estos son los unicos tipos de datos que se aceptan
    
    __TOOLS = None
    __MATRIZ = None
    
    #Constructor global
    #Acepta [int or bool or double], [[int or bool or double]], (int, int)
    def __init__(self, *args, **keyargs):
        self.__TOOLS = Tools()
        if(args):
            if(len(args) > 0):
                if(type(args[0]) in self.__TOOLS.getArrays()):
                    if(type(args[0][0]) in self.__TOOLS.getNumbers()):
                        self.f_Array_Matriz(args[0])
                    elif(type(args[0][0][0]) in self.__TOOLS.getBools()):
                        self.f_Matriz(args[0])
                    else:
                        self.f_Matriz_D(args[0])
                elif(type(args[0]) in self.__TOOLS.getIntegers() and type(args[1]) in self.__TOOLS.getIntegers()):
                    self.f_Row_Column(args[0],args[1])
                else:
                    raise Exception("No se pasaron los parametros correctos")
        elif(keyargs):
            if(type(keyargs.get('sourceMatriz')) in self.__TOOLS.getArrays()):
                if(type(keyargs.get('sourceMatriz')[0]) in self.__TOOLS.getNumbers()):
                    self.f_Array_Matriz(keyargs.get('sourceMatriz'))
                elif(type(keyargs.get('sourceMatriz')[0,0]) in self.__TOOLS.getBools()):
                    self.f_Matriz(keyargs.get('sourceMatriz'))
                else:
                    self.f_Matriz_D(keyargs.get('sourceMatriz'))
            elif (keyargs.get('row') in self.__TOOLS.getIntegers() and keyargs.get('column') in self.__TOOLS.getIntegers()):
                self.f_Row_Column(keyargs.get('row'),keyargs.get('column'))
            else:
                raise Exception("No se pasaron los parametros correctos")
        

    #Constructor que recibe [int or bool or double]
    def f_Array_Matriz(self, array):
        self.configure(len(array), 1)
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                self.set(r, c, array[r])

    #Constructor que recibe (int, int)
    def f_Row_Column(self, row, column):
        self.configure(row, column)
        
    #Constructor que recibe una [[int or double]]
    def f_Matriz_D(self, sourceMatriz):
        self.configure(len(sourceMatriz), len(sourceMatriz[0]))
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                self.set(r, c, sourceMatriz[r][c])

    #Constructor que recibe una [[bool]]
    def f_Matriz(self, sourceMatriz):
        self.configure(len(sourceMatriz), len(sourceMatriz[0]))
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                if(sourceMatriz[r][c]):
                    self.set(r, c, 1)
                else:
                    self.set(r, c, -1)
    
    #Inicializa la matriz
    def configure(self, row, column):
        self.__MATRIZ= zeros((row,column),dtype=double)
        
    #Asigna un valor en el elemento fila-columna
    def set(self, row, column, value):
        self.__MATRIZ[row][column]=value

    #Obtiene el valor del elemento fila-columna
    def get(self, row, column):
        return self.__MATRIZ[row][column]
    
    #Suma un valor al elemento que esta en esa fila-columna
    def add(self, row, column, value):
        try:
            self.validate(row, column, "add")
            newValue=self.get(row, column) + value
            self.set(row, column, newValue)
        except Exception as _except:
            raise _except
        

    #Valida que las filas y las columnas esten dentro del rango de la matriz
    def validate(self, row, column, function):
        if(row >= self.getRows() or row <0):
            raise Exception ("Matriz -> F("+function+") La fila no esta dentro del rango permitido")
        if(column >= self.getColumns() or column <0):
            raise Exception("Matriz- > F("+function+") La columna no esta dentro del rango permitido")

    #Crea una matriz de una sola columna
##     [[1],
##     [2],
##     [3]]
    @classmethod
    def createColumnMatriz(self, input):
        d= zeros((len(input),1),dtype=double)
        for i in range(len(d)):
            d[i][0]=input[i]
        return Matriz(d)

    #Crea una matriz de una sola fila
##    [[1, 2, 3]]
    @classmethod
    def createRowMatriz(self, input):
        d= zeros((1,len(input[0])),dtype=double)
        for i in range(len(d[0])):
            d[0,i] = input[0,i]
        return Matriz(d)

    #Pone a la matriz en ceros
    def clear(self):
        for r in range(0,self.getRows()):
            for c in range(self.getColumns()):
                self.set(r, c, 0)

    #Clona la matriz generando una nueva
    def clone(self):
        return Matriz(self.__MATRIZ)

    #Verifica que 2 matrices sean iguales
    def equals(self, matriz):
        return self.__equals1(matriz, 10)

    #Verifica que 2 matrices sean iguales con precision
    def __equals1(self, matriz, precision):
        if(precision < 0):
            raise Exception("La presicion debe ser mayor a cero")
        test = math.pow(10.0,precision)
        if(test == float("inf") or test > 0x7fffffffffffffff):
            raise Exception("La precision "+ precision + " decimales no es soportada")
        precision = math.pow(10, precision)
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                if(self.get(r,c)*precision != matriz.get(r,c) * precision):
                    return False
        return True

    #Rellena la matriz con el array
    def fromPackedArray(self, array, index):
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                self.__MATRIZ[r,c]=array[index]
                index += 1
        return index;

    #Nos dice si la matriz es un vector ya sea:
    #Una fila muchas columnas
    #Muchas filas una columna
    def isVector(self):
        return (False, True)[int((self.getRows() == 1 or self.getColumns() == 1))]

    #Indica si la matriz esta en ceros
    def isZero(self):
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                if(self.__MATRIZ[r,c] != 0):
                    return False;
        return True

    #Rellena la matriz con valores al azar
    def randomize(self, minimo, maximo):
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                self.__MATRIZ[r,c]=random()*(maximo - minimo) + minimo

    #Suma todos los elementos de la matriz
    def sum(self):
        result =0
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                result += self.__MATRIZ[r,c]
        return result

    #Devuelve un vector con los elementos de la matriz
    def toPackedArray(self):
        result = []
        index = 0
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                result.append(self.__MATRIZ[r,c])
        return result

    #Retorna la matriz
    def getMatriz(self):
        return self.__MATRIZ

    #Retorna la fila especificada de la matriz
    def getRow(self, row):
        from numpy import *
        if(row < 0 and row > self.getRows()):
            raise Exception("No se puede obtener la fila #"+ row)
        newMatriz = zeros((1,self.getColumns()), dtype=double)
        for c in range(self.getColumns()):
            newMatriz[0,c]=self.__MATRIZ[row,c]
        return Matriz(newMatriz)

    #Retorna la cantidad de filas que tiene la matriz
    def getRows(self):
        return len(self.__MATRIZ)

    #Retorna la columna especificada de la matriz
    def getColumn(self, column):
        from numpy import *
        if(column < 0 and column > self.getColumns()):
            raise Exception("No se puede obtener la columna #"+ column)
        newMatriz = zeros((self.getRows(),1), dtype=double)
        for r in range(self.getRows()):
            newMatriz[r][0]=self.__MATRIZ[r][column]
        return Matriz(newMatriz)

    #Retorna la cantidad de columnas que tiene la matriz
    def getColumns(self):
        return len(self.__MATRIZ[0])

    #Devuelve el tamaño de la matriz (filas * columnas)
    def size(self):
        return size(self.__MATRIZ)
    
    def setAttribute(self, item, data):
        globals()["__"+item.upercase()] = data
        
    def getAttribute(self, item, data):
        return globals()["__"+item.upercase()];
