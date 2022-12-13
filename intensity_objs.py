# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 20:27:41 2022

@author: victo
"""
import raytracer
import numpy as np
# simulation of intensities to help with PID control

class photo_diode():
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size

class light_source():
    def __init__(self, pos, ray_number):
        self.pos = pos
        self.rays = []
        self.n = ray_number
        for i in range(self.n):
            ray_direction = [0, np.sin(2*np.pi * (i/self.n)), np.cos(2*np.pi * (i/self.n))]
            self.rays.append(raytracer.Ray(self.pos, ray_direction))