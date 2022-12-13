# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:28:06 2021
implement snell's law in 3d
@author: victo
"""
import numpy as np
import scipy.optimize as op

# current function used in ray tracer: snell.Snell() - bottom function


# attempted numerical solution to find snells law in three dimensions using
# scipy's optimization algorithms

# sometimes the runtime to find the value is too long so it gives up
# this means a ray can go flying off, largely accurate other than that
# the vector implementation is quicker and more consistent though

def SnellNumerical(vecIn, vecNormal, n1, n2):
    '''vecIn = unit vector of direction of incoming light
    vecNormal = unit normal vector into surface of refraction
    n1, n2 = refractive indeces of materials before and after surface
    
    => numerically calculates the new direction of light after refraction
    '''
    vecIn = vecIn / np.sqrt(sum(vecIn**2)) # normalize vectors should the user not have read doc
    vecNormal = vecNormal / np.sqrt(sum(vecNormal**2))
    theta1 = np.arccos(np.dot(vecIn, vecNormal))
    if np.sin(theta1)*(n1/n2) > 1:
        return None
    theta2 = np.arcsin(np.sin(theta1)*n1/n2)
    
    def _finding_k2_eq(k2, k1, normal, theta):
        eq1 = np.dot(k2, np.cross(k1, normal))
        eq2 = np.dot(k2, normal) - np.cos(theta)
        eq3 = np.sqrt(sum(k2**2)) - 1
        return np.array([eq1, eq2, eq3])
    
    vecOut = op.fsolve(_finding_k2_eq, vecNormal, (vecIn, vecNormal, theta2))
    vecOut /= np.sqrt(sum(vecOut**2))
    return vecOut

    

# attempted analytic solution to snells law in three dimensions
# angles produced aren't accurate

def SnellAnalytical(vecIn, vecNormal, n1, n2):
    '''vecIn = unit vector of direction of incoming light
    vecNormal = unit normal vector into surface of refraction
    n1, n2 = refractive indeces of materials before and after surface
    
    => analytically calculates the new direction of light after refraction
    '''
    vecIn = vecIn / np.sqrt(sum(vecIn**2)) # normalize vectors should the user not have read doc
    vecNormal = vecNormal / np.sqrt(sum(vecNormal**2))
    theta1 = np.arccos(np.dot(vecIn, vecNormal))
    if np.sin(theta1) > n2/n1:
        return None
    theta2 = np.arcsin(np.sin(theta1)*n1/n2)
    
    #attempt at solving an equation with way too many unknowns
    a = np.cross(vecIn, vecNormal)
    n = vecNormal
    t = (np.cos(theta2) + (n[0]*a[2]/a[0]) - n[2]) / (n[1] - n[0]*a[1]/a[0]) # math in lab book, constant relating y-component of vecOut to z-component
    m = 1 + t**2 + ((t**2) * (a[1]**2) + 2 * t * a[1] * a[2] + a[2]**2) / a[0]**2
    k_z = np.sqrt(1/m)
    k_y = t * k_z
    k_x = -((k_y * a[1] + k_z * a[2])/a[0])
    vecOut = np.array([k_x, k_y, k_z])
    vecOut /= np.sqrt(sum(vecOut**2))
    return vecOut
    


# https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
# derivation of formula in report

#reflection function would ideally be called from mirrors.py
#however this causes circular imports as raytracer.py imports this module
#which would import mirrors.py
#which would import raytracer.py

def reflect(vecIn, vecNormal):
    'a function in the case of total internal reflection'
    vecOut = vecIn - 2 * (np.dot(vecIn, vecNormal) * vecNormal)
    return vecOut

def Snell(vecIn, vecNormal, n1, n2):
    '''vecIn = unit vector of direction of incoming light
    vecNormal = unit normal vector into surface of refraction
    n1, n2 = refractive indeces of materials before and after surface
    
    => analytically calculates the new direction of light after refraction
    with a formula found at https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
    '''
    vecIn = vecIn / np.sqrt(sum(vecIn**2)) # normalize vectors should the user not have read doc and the inputs aren't unit vectors
    vecNormal = vecNormal / np.sqrt(sum(vecNormal**2))
    
    theta1 = np.arccos(np.dot(vecIn, vecNormal))
    if np.sin(theta1) > n2/n1:
        #case of total internal reflection
        vecOut = reflect(vecIn, vecNormal)
        return vecOut
    
    comp_perp = (n1/n2) * (vecIn - (np.dot(vecNormal, vecIn) * vecNormal)) # component of output vector perpendicular to normal
    comp_para = np.sqrt(1 - ((n1/n2)**2 * (1 - (np.dot(vecNormal, vecIn)**2)))) * vecNormal # componenet of output vector parallel to normal
    vecOut = comp_para + comp_perp
    return vecOut

# functions for reflectivity

def Fresnel(vecIn, vecNormal, n1, n2, polarization = 'random'):
    """
    Implements the Fresnel equations to find the value for the reflectivity of the surface
    The angle between the vectors must be between 0 and pi/2
    
    Parameters
    ----------
    vecIn : TYPE
        the direction unit vector of the ray
    vecNormal : TYPE
        The normal vector to the surface.
        Should be pointing into the surface
        w.r.t. the ray's direction.
    n1 : float
        refractive index of the material before the surface
    n2 : TYPE
        refractive index of the material after the surface

    Returns
    -------
    Reflectivity of the surface for that ray
    between 0 and 1

    """
    theta = np.arccos(np.dot(vecIn, vecNormal))
    
    if theta > np.pi/2:
        vecNormal = -vecNormal
    
    theta = np.arccos(np.dot(vecIn, vecNormal))
    #reflectivity calculated in roundabout way due to floating point errors for small values
    
    R_s_numerator = n1 * np.cos(theta) - n2 * np.sqrt(1-((n1/n2)*np.sin(theta))**2)
    R_s_denominator = n1 * np.cos(theta) + n2 * np.sqrt(1-((n1/n2)*np.sin(theta))**2)
    R_s = (R_s_numerator / R_s_denominator) ** 2
    
    R_p_numerator = n2 * np.sqrt(1-((n1/n2)*np.sin(theta))**2) - n1 * np.cos(theta)
    R_p_denominator = n2 * np.sqrt(1-((n1/n2)*np.sin(theta))**2) + n1 * np.cos(theta)
    R_p = (R_p_numerator / R_p_denominator) ** 2
    
    if polarization == 'random':
        return 0.5 * (R_s + R_p)
    else:
        raise NotImplementedError('currently only considers unpolarized light')

