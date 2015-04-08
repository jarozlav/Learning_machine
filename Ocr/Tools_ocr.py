from SampleData import SampleData

    
class Tools_ocr:    
    
    #Lines = [[A][10101010101010101010101010101010101],
    #       [B][10101010101010101010101010101010101]]
    def toListLetters(self, lines, width, height):
        listletters = []
        for line in lines:
            listletters.append(self.__toSampleData(line, width, height))
        return listletters
    
    #Line = [A][10101010101010101010101010101010101]
    def __toSampleData(self, line, width, height):
        samp = SampleData(line[0], width, height)
        index = 0
        for c in range(samp.getColumns()):
            for r in range(samp.getRows()):
                samp.set(r, c, line[1][index] == '1')
                index += 1
        return samp
    
    #Convierte los patrones (SampleData) a una matriz
    def sampleDataToMatriz(self, listletters):
        _set = []
        for index in range(len(listletters)):
            _set.append(self.boolToDecimalList(listletters[index]))
        return _set
    
    #_input = [0.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5-0.50.5]
    def boolToDecimalList(self, sample):
        _input = [0 for y in range(sample.getColumns() * sample.getRows())]
        idx = 0
        for c in range(sample.getColumns()):
            for r in range(sample.getRows()):
                _input[idx] = (-0.5, 0.5)[sample.get(r,c)] #sample.get(r, c) ? 0.5: -0.5
                idx += 1
        return _input