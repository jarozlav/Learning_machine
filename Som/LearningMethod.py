#!/usr/bin/python
# -*- coding: utf-8 -*-


class LearningMethod:
    __ADDITIVE = False
    __SUBSTRACTIVE = False

    def __init__(self, _type):
        if(type(_type) == str):
            if(_type == 'ADDITIVE' or _type == 'additive'):
                self.__ADDITIVE = True
            elif (_type == 'SUBSTRACTIVE' or _type == 'substractive'):
                self.__SUBSTRACTIVE = True
            else:
                raise Exception("Tipo de aprendizaje no valido")
        elif(type(_type) == int):
            if(_type == 0):
                self.__ADDITIVE = True
            elif (_type == 1):
                self.__SUBSTRATIVE = True
            else:
                raise Exception("Tipo de aprendizaje no valido")
        else:
            raise Exception("Tipo de aprendizaje no valido")

    def getLearningMethod(self):
        if(self.__ADDITIVE):
            return "ADDITIVE"
        elif(self.__SUBSTRACTIVE):
            return "SUBSTRACTIVE"
        else:
            return ""

        
