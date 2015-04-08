#from numpy import *

class SampleData:
    
    grid = None
    letter = ''

    def __init__(self, letter, rows, columns):
        self.grid = [[0 for c in range(columns)] for r in range(rows)]
        self.letter = letter[0]

    #Limpia la matriz del patron que representa la letra
    def clear(self):
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                self.set(r, c, False)

    #Clona la matriz
    def clone(self):
        newSample= SampleData(self.letter, self.getRows(), self.getColumns())
        for r in range(self.getRows()):
            for c in range(self.getColumns()):
                newSample.set(r, c, self.get(r, c))
        return newSample

    #Compara letras
    def compareTo(self, sample):
        if(self.getLetter() > sample.getLetter()):
            return 1
        else:
            return -1

    def getGrid(self):
        return self.grid

    #Width
    def getRows(self):
        return len(self.grid)
    
    #height
    def getColumns(self):
        return len(self.grid[0])

    def getLetter(self):
        return self.letter

    def set(self, r, c, v):
        self.grid[r][c] = v

    def get(self, r, c):
        return self.grid[r][c]
