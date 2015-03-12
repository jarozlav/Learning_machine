#! /usr/bin/python
# -*- coding: utf-8 -*-

#Importacion de las librerias 
from Matriz import *

#Definicion de la clase
class MatrizMath:

    #Suma 2 matrices
    @classmethod
    def add(self, matrizA, matrizB):
        if(matrizA.getRows() != matrizB.getRows()):
            raise Exception("Las matrices tiene filas diferentes:\n\tMatriz A: "
                            +str(matrizA.getRows())+"\n\tMatriz B: "+str(matrizB.getRows()))
        elif (matrizA.getColumns() != matrizB.getColumns()):
            raise Exception("Las matrices tiene columnas diferentes:\n\tMatriz A: "
                            +str(matrizA.getColumns())+"\n\tMatriz B: "+str(matrizB.getColumns()))
        matrizC = matrizA.getMatriz() + matrizB.getMatriz()
        return Matriz(matrizC)

    #Copia la matriz (source) a la matriz (target)
    @classmethod
    def copy(self, source, target):
        if(source.getRows() != target.getRows()):
            raise Exception("Las matrices deben tener las mismas filas")
        if(source.getColumns() != target.getColumns()):
            raise Exception("Las matrices deben tener las mismas columnas")
        for r in range(source.getRows()):
            for c in range(source.getColumns()):
                target.set(r, c, source.get(r, c))

    #Elimina la columna especificada (deleted)
    @classmethod
    def deleteCoumn(self, matriz, deleted):
        if (deleted >= matriz.getColumns()):
            raise Exception("No se puede eliminar la columna #"+str(deleted)+" de la matriz, porque solo tiene "+str(matriz.getColumns())+" columnas")
        newMatriz = zeros((matriz.getRows(), matriz.getColumns()-1), dtype= double)
        for r in range(matriz.getRows()):
            targetColumn = 0
            for c in range(matriz.getColumns()):
                if(c != deleted):
                    newMatriz[r, targetColumn] = matriz.get(r, c)
                    targetColumn += 1
        return Matriz(newMatriz)

    #Elimina la columna especificada (deleted)
    @classmethod
    def deleteRow(self, matriz, deleted):
        if (deleted >= matriz.getRows()):
            raise Exception("No se puede eliminar la fila #"+str(deleted)+" de la matriz, porque solo tiene "+str(matriz.getRows())+" filas")
        newMatriz = zeros((matriz.getRows()-1, matriz.getColumns()), dtype= double)
        targetRow = 0
        for r in range(matriz.getRows()):
            if(r != deleted):
                for c in range(matriz.getColumns()):
                    newMatriz[targetRow, c] = matriz.get(r, c)
                targetRow += 1
        return Matriz(newMatriz)

    #Divide una matriz entre un divisor(int or double)
    @classmethod
    def divide(self, matriz, divisor):
        types=[int, double, float32, float64]
        if(type(divisor) in types ):
            return Matriz(matriz.getMatriz() / divisor)
        return 0.0
        

    #Producto punto de 2 matrices
    @classmethod
    def dotProduct(self, matrizA, matrizB):
        if(not matrizA.isVector() or not matrizB.isVector()):
            raise Exception("Para realizar el producto punto ambas matrices deben ser vectores")
        a = matrizA.toPackedArray()
        b = matrizB.toPackedArray()
        if(len(a) != len(b)):
            raise Exception("Para realizar el producto punto ambas matrices deben tener la misma longitud")
        return dot(a,b)

    #Genera la matriz identidad de tamaño (size)
    @classmethod
    def identity(self, size):
        if(size < 1):
            raise Exception("La matriz identidad debe tener mas de 1 elemento")
        return Matriz(identity(size))

    #Multiplica un matriz por un escalar
    @classmethod
    def scalar_multiply(self, matriz, escalar):
        types=[int, double, float, float64]
        #print type(escalar)
        if(type(escalar) in types):
            return Matriz(escalar * matriz.getMatriz())
        return 0.0

    #Multiplicacion cruz de 2 matrices
    @classmethod
    def cross_multiply(self, matrizA, matrizB):
        if(matrizA.getColumns() != matrizB.getRows()):
            raise Exception("Para realizar el producto cruz la matriz A debe tener tantas columnas como "+
                            "las filas de la matriz B")
        return Matriz(dot(matrizA.getMatriz(), matrizB.getMatriz()))

    #Resta la matrizB a la matrizA
    @classmethod
    def substract(self, matrizA, matrizB):
        if(matrizA.getRows() != matrizB.getRows()):
            raise Exception("Las matrices deben tener la misma cantidad de filas")
        if(matrizA.getColumns() != matrizB.getColumns()):
            raise Exception("Las matrices deben tener la misma cantidad de columnas")
        return Matriz(matrizA.getMatriz() - matrizB.getMatriz())

    #Multiplica elemento por elemento de las matrices
    @classmethod
    def multiply(self, matrizA, matrizB):
        result = Matriz(matrizA.getRows(), matrizB.getColumns())
        for r in range(matrizA.getRows()):
            for c in range(matrizA.getColumns()):
                multiply = matrizA.get(r, c) * matrizB.get(r, c)
                result.set(r, c, multiply)
        return result

    #Traspone la matriz
    @classmethod
    def transpose(self, matriz):
        return Matriz(matriz.getMatriz().T)

    #Determina la distancia vectorial del vector tipo matriz
    @classmethod
    def vectorLength(self, matriz):
        try:
            if(not matriz.isVector()):
                raise Exception("Solo se puede obtener la distancia vectorial de un vector")
            v = matriz.toPackedArray()
        except Exception as _except:
            print "Me enviaste un array"
            return 0
        sum_squart = 0
        for i in range(len(v)):
            sum_squart += math.pow(v[i], 2)
        return math.sqrt(sum_squart)

    #Obtiene el error cuadratico
    @classmethod
    def squartError(self, errors):
        temp = 0
        for r in range(errors.getRows()):
            temp += math.pow(errors.get(r, 0), 2)
        return temp * 0.5

    #Obtiene el sigmoidal de acuerdo a n
    @classmethod
    def sigmoidal(self, n):
        exp = (-1) * n
        densidad = (1 + math.pow(math.e, exp))
        return (1 / densidad)

    #Obtiene una matriz de sigmoidales de acuerdo a la matriz pasada
    @classmethod
    def matriz_sigmoidal(self, matriz):
        result = Matriz(matriz.getRows(), matriz.getColumns())
        for r in range(matriz.getRows()):
            for c in range(matriz.getColumns()):
                result.set(r, c, MatrizMath.sigmoidal(matriz.get(r, c)))
        return matriz

    #Sigmoidal prima
    @classmethod
    def delta_sigmoidal(self, n):
        return MatrizMath.sigmoidal(n) * (1 - MatrizMath.sigmoidal(n))

    #Matriz de sigmoidales primas
    @classmethod
    def matriz_delta(self, matriz):
        result = Matriz(matriz.getRows(), matriz.getColumns())
        for r in range(matriz.getRows()):
            for c in range(matriz.getColumns()):
                result.set(r, c, MatrizMath.delta_sigmoidal(matriz.get(r, c)))
        return result
