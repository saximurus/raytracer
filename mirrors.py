# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 13:30:42 2021

@author: victo
"""

import numpy as np
import raytracer


def reflect(vecIn, vecNormal):
    '''
    reflection function - takes the component of the input vector
    which is parallel to the normal and makes it negative

    Parameters
    ----------
    vecIn : unit vector
        -the input ray
    vecNormal : unit vector
        -the ray normal to the surface of the reflector

    Returns
    -------
    vecOut : unit vector
        the vector describing the direction of the ray after reflection

    '''
    vecOut = vecIn - 2 * (np.dot(vecIn, vecNormal) * vecNormal)
    return vecOut

class SphericalReflection(raytracer.OpticalElement):
    'A spherical reflector element - essentially the same as a refractor but it applies reflection instead of Snell\'s law'
    def __init__(self, pos, curvature, r_ap):
        raytracer.OpticalElement.__init__(self, refraction = False)
        self.pos = pos
        self.curvature = curvature
        self.reflection = True
        self.r_ap = r_ap
        
    def propagate_ray(self, ray):
        intercept = self.intercept(ray)
        if type(intercept) == type(None):
            return
        normal = raytracer.SphericalRefraction.normal(self, intercept, ray)
        direction = ray.k()
        newDirection = reflect(direction, normal)
        ray.append(intercept, newDirection)
        

