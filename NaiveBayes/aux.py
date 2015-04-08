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

    def _banner(self):
        banner = '\t\t\t\tBienvenido\n'
        banner += '\t\t\tAlgoritmo: \tNaive Bayes\n'
        banner += '\t\t\tversion: \t1.0\n'
        banner += '\t\t\tClasifica: \tOraciones\n'
        banner += '\t\t\tAutor: \t\n'
        self.printer(banner, 'hi')

    def _help(self):
        _help = 'El algoritmo termina en el proceso de entrenamiento\n'
        _help += "Ahora debes comprobar que clasifica correctamente con la funcion:\n"
        _help += "run = RunNaiveBayes()\n"
        _help += "run.Test(test)\n"
        _help += "Para clasificar texto se usa la funcion:\n"
        _help += "run.Classify()\n"
        self.printer(_help, '', False)

    def _bye(self):
        bye = '\t\t\tBorrando pesos\n'
        bye += '\t\t\tAlgoritmo apagado'
        self.printer(bye, 'bye')

    def _norun(self, train_set):
        self._help()
        self._configure(False, train_set)

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

    ###############   PUBLIC  ###############
    def Train(self, train=None):
        if train is None:
            self.train()
        else:
            self.train = self.validTrainList(train)
        self._whiletrain()

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

    def Test(self, test=None, withtest=True):
        if test == None:
            if withtest:
                self.testManualorFile()
        else:
            self.test(test)

    def Classify(self, classify=None):
        self.printer("Fase de clasificacion:", '')
        if classify is None:
            self._whileClassification()
        else:
            return self._classifyPatterns(classify)

    ###############   PRIVATE  ###############
    
    def train(self):
        if not self.autorun:
            self.namefiletrain = self.getNamefileTrain()
        self.train = self.loadTrain(self.namefiletrain)

    def _stopwords(self):
        if not self.autorun:
            if self.withstopwords:
                self.namefilestopwords = self.getNamefileStopwords()
        if self.withstopwords:
            self.stopwords = self.loadStopwords(self.withstopwords, self.namefilestopwords)

    def test(self, test):
        self.printer("Validando datos 'Test'", 'load')
        listoftest = self.__testPatterns(test)
        self.__whileTest(listoftest)

    #def wantStopwords(self):
    #    return Tools.getBools("Quiere usar un archivo [Stopwords] para filtrar palabras?")
    
    ###############     CARGAS
    
    def loadData(self):
        self.Train()
        self._stopwords()
    
    def loadTrain(self, nameFile):
        self.printer("Cargando datos de entrenamiento", 'load')
        inList = Tools.load(nameFile, relative_path='Train/', sep=': ')
        self.printer("Datos de entrenamiento cargados de /Data/Train/"+nameFile, 'done')
        return self.validTrainList(inList)
    
    def loadStopwords(self, withstopwords, namefile):
        self.printer("Cargando las palabras del stopwords", 'load')
        if namefile is None:
            from nltk.corpus import stopwords
            self.printer("Stopwords cargados de nltk.corpus.stopwords(english)\n", 'done')
            return set(stopwords.words('english'))
        else:
            self.printer("Stopwords cargados de Data/Stopwords/"+nameFile+'\n', 'done')
            return Tools.load(namFile,relative_path='Stopwords/', sep='\n')

    def _loadTest(self, nameFile):
        inList = Tools.load(nameFile, relative_path='Test/', sep=': ')
        self.printer("Datos de validacion han sido cargados de /Data/Test/"+nameFile, 'done')
        return self.__testPatterns(inList)

    ###############   Obtiene el nombre de los archivos ###############
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
    
    ###############     VALIDACIONES
    ###############   Valida que el nombre de los archivos exista en el directorio  ###############
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

    ###############   Valida la lista de entrenamiento  ###############
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
        
    ##############  Valida la entrada individual del test
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

    ###############     PROCESOS

    def _whiletrain(self):
        self.printer("Creando el algoritmo Naive Bayes", 'load')
        self.classifier = NaiveBayesClassifier(self.train, self.stopwords)
        self.printer("Naive Bayes creado\n", 'done')
        self.printer("Entrenando el algoritmo", 'load')
        self.classifier.train()
        self.printer("Algoritmo entrenado\n", 'done')
        
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
        
    def _whileClassification(self):
        _next = True
        while (_next):
            self._manualOrinFile()
            _next = Tools.haveNext()

    ##############      REDIRECCIONES
    
    ##  Validar mediante?
    def testManualorFile(self):
        archive = Tools.getBools("Desea testear desde un archivo?")
        if archive:
            self.TestwithArchive()
        else:
            self.Testmanual()

    ##  Clasificar mediante?
    def _manualOrinFile(self):
        textinput = 'Desea clasificar datos de un archivo?'
        archive = Tools.getBools(textinput)
        if archive:
            classified = self.classifyArchive()
        else:
            classified = self.classifyPattern()
        return classified
    
    ##############      SUBPROCESOSS

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class Tools():
    #############
    ##Load files#
    #############
    @classmethod
    def _read_data(self, dataset, format=None):
        """Reads a data file and returns an iterable that can be used
        as testing or training data.
        """
        # Attempt to detect file format if "format" isn't specified
        #dataset = 'Data/' + dataset
        if not format:
            format_class = formats.detect(dataset)
            if not format_class:
                raise FormatError('Could not automatically detect format for the given '
                                  'data source.')
        else:
            registry = formats.get_registry()
            if format not in registry.keys():
                raise ValueError("'{0}' format not supported.".format(format))
            format_class = registry[format]
        return format_class(dataset, **self.format_kwargs).to_iterable()
    
    @classmethod
    def load(self, nameFile, relative_path='', sep=': '):
        '''
        carga los datos de un archivo en el formato
        etiqueta: sentence
            o
        contraccion sin_contraccion
        '''
        absolute_path = 'Data/'
        path = absolute_path + relative_path + nameFile
        return [(l.strip()).split(sep) for l in (open(path).readlines())]
    
    ###################
    ##Input data user##
    ###################
    
    @classmethod
    def haveNext(self):
        textinput = 'Desea continuar?'
        return Tools.getBools(textinput)
    
    @classmethod
    def getBools(self, textinput):
        yes = ['Si', 'S', 'si', 's']
        nots = ['No', 'N', 'no', 'n']
        valids = yes + nots
        return Tools.getBool(valids, yes, textinput)
    
    @classmethod
    def getBool(self, valids, yes, textinput):
        correct = False
        _input = ""
        while(not correct):
            _input = Tools.getInput(textinput)
            if(_input in valids):
                if(_input in yes):
                    return True
                else:
                    return False
            else:
                Tools.printer("Ingresa alguna de estas opciones", 'warning')
                Tools.printer(str(valids)+'\n', '', False)
                correct = False
                
    @classmethod
    def getInput(self, message):
        Tools.printer(message, '', False)
        data = raw_input()
        print ""
        return data
    
    ############################
    ##Tranform list and tuples##
    ############################
    @classmethod
    def getListofTuples(self, patterns):
        """
        param patterns: Lista de listas en formato
        [['sentence', 'label'],...]
        """
        tuples = []
        for pattern in patterns:
            tuples.append(Tools.getTuples(pattern))
        return tuples
    
    @classmethod
    def getTuples(self, pattern):
        return tuple(pattern)

    ### Reverse a list ###
    @classmethod
    def reverseInList(self, patterns):
        '''
        param patterns: una lista de lista de la forma
        [['label', 'sentences'],...]
        
        rtype: una lista de lista de la forma
        [['sentences', 'label'],...]
        '''
        new_patterns = []
        for pattern in patterns:
            new_patterns.append(Tools.reverse(pattern))
        return new_patterns
    
    @classmethod
    def reverse(self, pattern):
        return [pattern[1], pattern[0]]
    
    ###Valid format
    @classmethod
    def validFormat(self, patterns, typeinput):
        valid = True
        for pattern in patterns:
            if typeinput == 'train':
                valid = Tools._validtrain(pattern)
            elif typeinput == 'classify':
                valid = Tools._validclassify(pattern)
            else:
                print "Error"
            if not valid:
                break
        return valid
    
    @classmethod
    def _validtrain(self, pattern):
        typestring = [basestring, str]
        if len(pattern) == 2:
            if type(pattern[0]) in typestring and type(pattern[1]) in typestring:
                return True
        return False
    
    @classmethod
    def _validclassify(self, pattern):
        typestring = [basestring, str]
        if len(pattern) == 1:
            if type(pattern[0]) in typestring:
                return True
        return False
    
    @classmethod
    def _validregex(self, pattern):
        if len(pattern) == 2:
            if pattern[0] == basestring and pattern[1] == basestring:
                return True
        return False
    
    @classmethod
    def inOrder(self, pattern):
        if len(pattern[0]) > 10:
            return True
        return False
    
    #######################
    ##Replce contractions##
    #######################
    @classmethod
    def preprocessInList(self, patterns):
        '''
        param patterns = Es una lista de listas de la forma
        [['sentence', 'label']]
        donde sentence puede contener contracciones
        
        rtype: Una lista de lista de la forma
        [['sentence', 'label']]
        donde sentence no contiene contracciones
        '''
        for pattern in patterns:
            pattern[0] = Tools.preprocess(pattern[0])
        return patterns
    
    @classmethod
    def preprocess(self, sentence):
        replacer = RegexpReplacer()
        #have = self.haveContractions(sentence, replacer.getRegex())
        #if have:
        #    sentence = replacer.replace(sentence)
        sentence = replacer.replace(sentence)
        return sentence
    
    ######################
    ##Print data to user##
    ######################
    
    @classmethod
    def notviewprocess(self, text, event, turn=True):
        pass
    
    @classmethod
    def printer(self, text, event, turn = 1):
        
        gray = '\033[1;30m'
        red = '\033[1;31m'
        green = '\033[1;32m'
        yellow = '\033[1;33m'
        blue = '\033[1;34m'
        magenta = '\033[1;35m'
        cyan = '\033[1;36m'
        white = '\033[1;37m'
        crimson = '\033[1;38m'
        default = '\033[1;m'
        
        warning = red
        done = green
        loadding = blue
        clear = default
        title = magenta
        data = gray
        hiandbye = cyan
        
        open_ = white + '[' #+ default
        close_ = white + ']   ' #+ default
        inside_l = loadding + 'loadding'
        inside_d = done + '  done  '
        inside_w = warning + 'warning '
        inside_da = data + '  data  '
        inside = cyan + '  ...   '
        toprint = ''
        
        if event == 'load':
            if turn:
                toprint = open_ + inside_l + close_
            toprint += loadding + text + '...'
        elif event == 'done':
            if turn:
                toprint = open_ + inside_d + close_
            toprint += done + text
        elif event == 'warning':
            if turn:
                toprint = open_ + inside_w + close_
            toprint += warning + text
        elif event == 'error':
            toprint += error + text
        elif event == 'data':
            if turn:
                toprint = open_ + inside_da + close_
            toprint += data + text
        elif event == 'title':
            toprint += title + text
        elif event == 'hi' or event == 'bye':
            toprint += hiandbye + text
        else:
            if turn:
                toprint = open_ + inside + close_
            toprint += text
        print toprint + default