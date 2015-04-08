class RunNaiveBayes():
    '''
    Esta clase permite normalizar el texto
    quitar contracciones
    cargar las stopwords
    '''
    
    stopwords = None
    namefilestopwords = None
    namefiletrain = '.train.txt'
    namefiletest = '.test.txt'
    printer = Tools.printer

    def __init__(self,train_set=None, autorun=False, withstopwords=False, filestopwords=None, viewprocess=True):
        self.autorun = autorun
        self.withstopwords = withstopwords
        self.namefilestopwords = filestopwords
        if not viewprocess:
            self.printer= Tools.notviewprocess
        self._banner()
        if autorun:
            self._configure(True, train_set)
        else:
            self._norun(train_set)

    def _norun(self, train_set):
        self._help()
        self._configure(False, train_set)

    def _help(self):
        _help = 'El algoritmo termina en el proceso de entrenamiento\n'
        _help += "Ahora debes comprobar que clasifica correctamente con la funcion:\n"
        _help += "run = RunNaiveBayes()\n"
        _help += "run.Test(test)\n"
        _help += "Para clasificar texto se usa la funcion:\n"
        _help += "run.Classify()\n"
        self.printer(_help, '', False)

    def _banner(self):
        banner = '\t\t\t\tBienvenido\n'
        banner += '\t\t\tAlgoritmo: \tNaive Bayes\n'
        banner += '\t\t\tversion: \t1.0\n'
        banner += '\t\t\tClasifica: \tOraciones\n'
        banner += '\t\t\tAutor: \t\n'
        self.printer(banner, 'hi')

    def _bye(self):
        bye = '\t\t\tBorrando pesos\n'
        bye += '\t\t\tAlgoritmo apagado'
        self.printer(bye, 'bye')

    def _configure(self, autorun, train_set=None):
        separator = "################################################################################"
        if train_set is not None:
            self.printer(separator, 'title')
            self.printer("\t\t\t\tCargar Datos", 'title')
            self.printer(separator+'\n', 'title')
            self.train = self.validTrainList(train_set)
            self._stopwords()
            self._whiletrain()
        if autorun:
            if train_set is None:
                self.printer(separator, 'title')
                self.printer("\t\t\t\tCargar Datos", 'title')
                self.printer(separator+'\n', 'title')
                self.loadData()
                self.printer(separator, 'title')
                self.printer("\t\t\t\tEntrenamiento", 'title')
                self.printer(separator+'\n', 'title')
                self._whiletrain()
            self.printer(separator, 'title')
            self.printer("\t\t\t\t\tTesteo", 'title')
            self.printer(separator+'\n', 'title')
            self._whileTest(self.namefiletest)
            #self.TestwithArchive()
            self.printer(separator, 'title')
            self.printer("\t\t\t\tClasificacion", 'title')
            self.printer(separator+'\n', 'title')
            self.Classify()
            #self.printer(separator, 'title')
            #self.printer("\t\t\t\tActualizar datos", 'title')
            #self.printer(separator, 'title')
            #self.update()
            self.printer(separator+'\n', 'title')
            self._bye()

    def loadData(self):
        self.Train()
        self._stopwords()

    def Stopwords(self, stopwords=None, ask=False):
        if type(stopwords) == list:
            self.printer("Asignando stopwords", 'load')
            self.stopwords=stopwords
            self.printer("Stopwords asignados\n", 'done')
        else:
            if ask:
                self.withstopwords = True
                self._stopwords()
            else:
                self.stopwords = self.loadStopwords(self.withstopwords, self.namefilestopwords)

    def _stopwords(self):
        if not self.autorun:
            if self.withstopwords:
                self.namefilestopwords = self.getNamefileStopwords()
        if self.withstopwords:
            self.stopwords = self.loadStopwords(self.withstopwords, self.namefilestopwords)

    def Train(self, train=None):
        if train is None:
            self.train()
        else:
            self.train = self.validTrainList(train)
        self._whiletrain()

    def train(self):
        if not self.autorun:
            self.namefiletrain = self.getNamefileTrain()
        self.train = self.loadTrain(self.namefiletrain)

    def wantStopwords(self):
        return Tools.getBools("Quiere usar un archivo [Stopwords] para filtrar palabras?")

    def getNamefileTrain(self):
        self.__function = self.validaNameFileTrain
        self.__namefilefunction = self.namefiletrain
        return self.getNameFile("Ingresa el nombre del archivo de entrenamiento o presiona [Enter]")

    def getNamefileStopwords(self):
        self.__function = self.validaNameFileStopwords
        self.__namefilefunction = self.namefilestopwords
        return self.getNameFile("Ingresa el nombre del archivo que contiene las stopwords o presiona [Enter]")

    def getNamefileTest(self):
        self.__function = self.validaNameFileTest
        self.__namefilefunction = self.namefiletest
        return self.getNameFile("Ingresa el nombre del archivo para el testeo o presiona [Enter]")

    def getNamefileClassify(self):
        self.__function = self.validaNameFileClassify
        self.__namefilefunction = ""
        enter = False
        namefile = ''
        while not enter:
            namefile = self.getNameFile("Ingresa el nombre del archivo con texto a clasificar")
            if namefile == '':
                self.printer("No has ingresado el nombre del archivo\n", 'warning')
                enter = False
            else:
                enter = True
        return namefile

    def getNameFile(self, text):
        correct = False
        while not correct:
            namefile = Tools.getInput(text)
            if namefile != '':
                correct = self.__function(namefile)
            else:
                namefile = self.__namefilefunction
                correct = True
        return namefile

    def validaNameFileTrain(self, namefile):
        return self.validaNameFile(namefile, 'Train/')

    def validaNameFileStopwords(self, namefile):
        return self.validaNameFile(namefile, 'Stopwords/')

    def validaNameFileTest(self, namefile):
        return self.validaNameFile(namefile, 'Test/')

    def validaNameFileClassify(self, namefile):
        return self.validaNameFile(namefile, 'Classify/')

    def validaNameFile(self, namefile, relative_path):
        import os
        path = 'Data/'+relative_path
        allarchives = os.listdir(path)
        listofarchives = [archive for archive in allarchives if os.path.isfile(path+archive) and archive[-3:] == 'txt']
        exist = namefile in listofarchives
        if not exist:
            self.printer("El archivo: "+namefile+" no existe", 'warning')
            self.printer("Ingresa el nombre de alguno de estos: ", 'warning')
            self.printer(str(listofarchives), 'warning')
        return exist

    #carga archivos y remplaza contracciones 
    def loadTrain(self, nameFile):
        self.printer("Cargando datos de entrenamiento", 'load')
        inList = Tools.load(nameFile, relative_path='Train/', sep=': ')
        self.printer("Datos de entrenamiento cargados de /Data/Train/"+nameFile, 'done')
        return self.validTrainList(inList)

    def validTrainList(self, inList):
        if (Tools.validFormat(inList, 'train')):
            self.printer("Los datos de entrenamiento cumplen con el formato", 'done')
            self.printer("[['string', 'string'], ...]", 'done')
            if not Tools.inOrder(inList[0]):#permitir que corrija uno a uno toda la lista
                self.printer("Los datos estan invertidos", 'warning')
                self.printer("Invirtiendo", 'load')
                inList = Tools.reverseInList(inList)
                self.printer("Datos invertidos", 'done')
            self.printer("Datos correctos", 'done')
            self.printer("Remplazando contracciones en las oraciones", 'load')
            inList_process = Tools.preprocessInList(inList)
            self.printer("Las contracciones de las oraciones han sido remplazadas\n", 'done')
            return Tools.getListofTuples(inList_process)
        else:
            raise Exception("Los datos del entrenamiento no cubren el formato")

    def loadStopwords(self, withstopwords, namefile):
        self.printer("Cargando las palabras del stopwords", 'load')
        if namefile is None:
            from nltk.corpus import stopwords
            self.printer("Stopwords cargados de nltk.corpus.stopwords(english)\n", 'done')
            return set(stopwords.words('english'))
        else:
            self.printer("Stopwords cargados de Data/Stopwords/"+nameFile+'\n', 'done')
            return Tools.load(namFile,relative_path='Stopwords/', sep='\n')

    def _whiletrain(self):
        self.printer("Creando el algoritmo Naive Bayes", 'load')
        self.classifier = NaiveBayesClassifier(self.train, self.stopwords)
        self.printer("Naive Bayes creado\n", 'done')
        self.printer("Entrenando el algoritmo", 'load')
        self.classifier.train()
        self.printer("Algoritmo entrenado\n", 'done')

    def Test(self, test=None, withtest=True):
        if test == None:
            if withtest:
                self.testManualorFile()
        else:
            self.test(test)

    def test(self, test):
        self.printer("Validando datos 'Test'", 'load')
        listoftest = self.__testPatterns(test)
        self.__whileTest(listoftest)

    def testManualorFile(self):
        archive = Tools.getBools("Desea testear desde un archivo?")
        if archive:
            self.TestwithArchive()
        else:
            self.Testmanual()

    def TestwithArchive(self):
        namefile = self.getNamefileTest()
        self._whileTest(namefile)

    def Testmanual(self):
        tupleTest = self.validaInputTest()
        inTuple = self.__testPattern(tupleTest)
        if inTuple is not None:
            self.__whileTest(inTuple)
        else:
            self.printer("Los datos estan clasificados incorrectamente", 'warning')

    def validaInputTest(self):
        correct = False
        tupletest = None
        while not correct:
            text = Tools.getInput("Ingresa la tupla a testear, separados por @\tFormato: oracion@clase")
            tupletest = text.split('@')
            if len(tupletest) == 2:
                correct = True
            else:
                self.printer("No ingresaste los datos en forma correcta", 'warning')
                self.printer("Formato: oracion@clase\n", '')
                correct = False
        return tupletest

    def _whileTest(self, nameFile):
        self.printer("Validando datos 'Test'", 'load')
        listoftest = self._loadTest(nameFile)
        if listoftest is not None:
            self.__whileTest(listoftest)
        else:
            self.printer("Los datos estan clasificados incorrectamente", 'warning')

    def __whileTest(self, listoftest):
        count = 1
        for (sentence, label) in listoftest:
            self.printer("Validando test {0}".format(count), 'load')
            self.printer("({0}, {1})".format(sentence, label), '')
            labeled = self.classifier.classify(sentence)
            if labeled != label:
                self.printer("Oracion clasificada incorrectamente", 'warning')
                self.printer("\t\t{0} != {1}".format(labeled, label), 'warning')
                self.printer("Actualizar conjunto de entrenamiento", 'load')
                self.classifier.update([(sentence, label)])
                self.printer("El conjunto de entrenamiento ha sido actualizado", 'done')
            self.printer("Oracion clasificada correctamente\n", 'done')
            count += 1
        self.printer("Datos del test han sido validados\n", 'done')

    def validaLabels(self, inList):
        correct = True
        for pattern in inList:
            correct = self.validaLabel(pattern[1])
            if not correct:
                correct = False
                break
        return correct

    def validaLabel(self, label):
        inside = label in self.classifier.labels()
        return label in self.classifier.labels()

    def _loadTest(self, nameFile):
        inList = Tools.load(nameFile, relative_path='Test/', sep=': ')
        self.printer("Datos de validacion han sido cargados de /Data/Test/"+nameFile, 'done')
        return self.__testPatterns(inList)

    def __testPatterns(self, inList):
        if (Tools.validFormat(inList, 'train')):
            self.printer("Los datos de validacion cumplen con el formato", 'done')
            self.printer("[['string', 'string'], ..]", 'done')
            if not Tools.inOrder(inList[0]):#permitir que corrija uno a uno toda la lista
                self.printer("Los datos estan invertidos", 'warning')
                self.printer("Invirtiendo...", 'load')
                inList = Tools.reverseInList(inList)
                self.printer("Datos invertidos", 'done')
            self.printer("Datos correctos", 'done')
            self.printer("Remplazando contracciones en las oraciones...", 'load')
            inList_process = Tools.preprocessInList(inList)
            self.printer("Las contracciones de las oraciones han sido remplazadas\n", 'done')
            corrects = self.validaLabels(inList_process)
            if corrects:
                tuples = Tools.getListofTuples(inList_process)
            else:
                tuples = None
            return tuples#Tools.getListofTuples(inList_process)
        else:
            raise Exception("Los datos de testeo no cubren el formato")

    def __testPattern(self, pattern):
        if (Tools._validtrain(pattern)):
            self.printer("La tupla cumple con el formato", 'done')
            self.printer("[['string', 'string'], ..]", 'done')
            if not Tools.inOrder(pattern):#permitir que corrija uno a uno toda la lista
                self.printer("Los datos estan invertidos", 'warning')
                self.printer("Invirtiendo...", 'load')
                pattern = Tools.reverse(pattern)
                self.printer("Datos invertidos", 'done')
            self.printer("Datos correctos", 'done')
            self.printer("Remplazando contracciones en la oracion...", 'load')
            pattern[1] = Tools.preprocess(pattern[1])
            self.printer("Las contracciones de la oracion han sido remplazadas\n", 'done')
            corrects = self.validaLabel(pattern[1])
            if corrects:
                tuples = [Tools.getTuples(pattern)]
            else:
                tuples = None
            return tuples#Tools.getTuples(inList_process)
        else:
            raise Exception("Los datos del entrenamiento no cubren el formato")

    def Classify(self, classify=None):
        self.printer("Fase de clasificacion:", '')
        if classify is None:
            self._whileClassification()
        else:
            return self._classifyPatterns(classify)

    def _whileClassification(self):
        _next = True
        while (_next):
            self._manualOrinFile()
            _next = Tools.haveNext()

    def _manualOrinFile(self):
        textinput = 'Desea clasificar datos de un archivo?'
        archive = Tools.getBools(textinput)
        if archive:
            classified = self.classifyArchive()
        else:
            classified = self.classifyPattern()
        return classified

    def classifyArchive(self):
        classifieds = None
        self.printer("Clasificacion mediante archivos", '', False)
        namefile = self.getNamefileClassify()
        self.printer("Cargando oraciones de /Data/Classify/"+namefile, 'load')
        try:
            inList = Tools.load(namefile, relative_path='Classify/')
        except Exception:
            self.printer("Verifica que el archivo este en la carpeta: Data/Classify/\n", 'warning')
            return
        self.printer("Oraciones cargadas", 'done')
        self.printer("Validando el formato de las oraciones", 'load')
        if(Tools.validFormat(inList, 'classify')):
            self.printer("Oraciones validas", 'done')
            classifieds = self._classifyPatterns(inList)
        else:
            raise Exception("Los datos de clasificacion no cubren el formato")
        return classifieds

    def _classifyPatterns(self, patterns):
        self.printer("Clasificacion de oraciones", 'load')
        self.printer('', '', False)
        classifieds = []
        count = 1
        for pattern in patterns:
            self.printer("Clasificacion N {0}".format(count), 'load')
            classified = self._classifyPattern(pattern[0])
            classifieds.append(classified)
            self.printer("Resultado obtenido: ", '')
            self.printer(str(classified)+'\n', '')
            count += 1
        return classifieds

    def classifyPattern(self):
        self.printer("Clasificacion de oracion", 'load')
        pattern = Tools.getInput("Ingresa la oracion a clasificar")
        classified = self._classifyPattern(pattern, True)
        self.printer("Resultado obtenido: ", '')
        self.printer(str(classified)+'\n', '')
        return classified

        #pattern = sentence
    def _classifyPattern(self, pattern, optimized = False):
        self.printer(str(pattern), '')
        #validar si tiene contraciones
        self.printer("Borrando contracciones de la oracion ingresada", 'load')
        pattern = Tools.preprocess(pattern)
        self.printer("Contracciones borradas", 'done')
        self.printer("Clasificando oracion", 'load')
        label = self.classifier.classify(pattern, optimized)
        self.printer("Oracion clasificada", 'done')
        return Tools.getTuples([pattern, label])
    
