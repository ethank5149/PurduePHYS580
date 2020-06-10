########################################################################################################################
#     ========     |  Constants                                                                                        #
#     \\           |  Basic classes for physical constants                                                             #
#      \\          |                                                                                                   #
#      //          |  Author: Ethan Knox                                                                               #
#     //           |  Website: https://www.github.com/ethank5149                                                       #
#     ========     |  MIT License                                                                                      #
########################################################################################################################
########################################################################################################################
# License                                                                                                              #
# Copyright 2020 Ethan Knox                                                                                            #
#                                                                                                                      #
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated         #
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the  #
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to      #
# permit persons to whom the Software is furnished to do so, subject to the following conditions:                      #
#                                                                                                                      #
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the #
# Software.                                                                                                            #
#                                                                                                                      #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE #
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS  #
# OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR  #
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.     #
########################################################################################################################


class Air:
    def __init__(self):
        self.density = 1.225
        self.viscosity = 2.0e-5


class Water:
    def __init__(self):
        self.density = 1000
        self.viscosity = 1.0e-3

class Physics:
    def __init__(self):
        self.G = 6.67408e-11

class Sun:
    def __init__(self):
        self.radius = 695700e3
        self.mass = 1988500e24
        self.volume = 1412000e21
        self.density = 1408
        self.GM = 132712e15
        self.luminosity = 382.8e24
        self.period = 609.12*60*60

class Earth:
    def __init__(self):
        self.radius = 6371e3
        self.mass = 5.9724e24
        self.volume = 1.083e21
        self.density = 5514
        self.GM = 0.3986e15
        self.period = 23.9345*60*60
