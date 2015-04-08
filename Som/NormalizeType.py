#!/usr/bin/python
# -*- coding: utf-8 -*-

class NormalizeType:
    __Z_AXIS = False
    __MULTIPLICATIVE = False

    def __init__(self, _type):
        if(type(_type) == str):
            if(_type == 'Z_AXIS' or _type == 'z_axis'):
                self.__Z_AXIS = True
            elif (_type == 'MULTIPLICATIVE' or _type == 'multiplicative'):
                self.__MULTIPLICATIVE = True
            else:
                raise Exception("No se construyo NormalizeType")
        elif(type(_type) == int):
            if(_type == 0):
                self.__MULTIPLICATIVE = True
            elif (_type == 1):
                self.__Z_AXIS = True
            else:
                raise Exception("No se construyo NormalizeType")
        else:
            raise Exception("No se construyo NormalizeType")

    def getNormalizeType(self):
        if(self.__Z_AXIS):
            return "Z_AXIS"
        elif(self.__MULTIPLICATIVE):
            return "MULTIPLICATIVE"
        else:
            return ""
