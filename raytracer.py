# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 22:42:00 2021

@author: victo
"""

import numpy as np

import snell

class Ray:
    'a ray object'
    def __init__(self, pos = [0, 0, 0], direction = [0, 0, 1]):
        self.pos = np.array(pos)
        self.direction = np.array(direction)
        self.poslist = [np.array(pos)]
        self.directions = [np.array(direction)]
    
    def p(self):
        'returns the current position of the ray'
        return self.pos
    
    def k(self):
        'returns the current direction of the ray'
        return self.direction
    
    def append(self, pos, direction):
        'update the current position and direction and add these to the ray\'s history'
        self.pos = pos
        self.direction = direction
        self.poslist.append(pos)
        self.directions.append(direction)
        
    def vertices(self):
        'returns a list of the previous positions of the ray'
        return self.poslist
    
class RayBundle:
    '''
    A circular bundle of rays all pointed in the same direction
    
    Will be circular on the x-y plane at the bundle's starting z-axis position
    
    Parameters:
        pos = starting position of the central ray of the bundle
        
        direction = starting direction of all of the rays in the bundle
        
        rayDensity = the number of rays per square unit on x-y plane - may be
        slightly increased to make an even distribution of rays with a specific radius
        
        radius = radius of the circular cross section of the beam - will not be adjusted
    '''
    
    def __init__(self, pos = [0, 0, 0], direction = [0, 0, 1], rayDensity = 1, radius = 1):
        self.pos = pos
        self.direction = direction
        ray_positions = self._find_bundle_start_positions(rayDensity, radius)
        self.rays = [Ray(pos = position, direction = self.direction) for position in ray_positions]
        

    def _find_bundle_start_positions(self, d, r):
        'finds the starting positions of the rays in the bundle'
        area = np.pi * r ** 2
        n = d*area # finds number of rays
        layer_number = int(np.ceil(n/6)) # finds approximate number of layers in the circle to produce same pattern as in assignment sheet
        points = []
        try:
            scale_factor = r / (layer_number - 1) # pattern is built on assumption that each layer is 1 unit farther from the center as the previous - this is the factor to scale back to radius
        except ZeroDivisionError:
            #scale factor is irrelevant anyway for layer 1
            pass
        for i in range(layer_number):
            if i == 0:
                layer = [self.pos] # creates center beam
            else:
                # next line produces pattern of beam from assignment sheet
                layer = [np.array([(self.pos[0] + np.cos(m)*scale_factor*i), (self.pos[1] + np.sin(m)*scale_factor*i), self.pos[2]]) for m in np.linspace(0, 2*np.pi, 6*i + 1)[:-1]] # sorry, terrible to read but amazing code golf
            points += layer
        return points
    
    def update_pos(self):
        central_ray = self.rays[0]
        self.pos = central_ray.p()
    
    def spread(self):
        'finds the current RMS spread of the rays from the center of the beam'
        total = 0
        for ray in self.rays:
            r = ray.p()-self.pos
            r_squared = sum(r**2)
            total += r_squared
        rms = np.sqrt(total / len(self.rays))
        return rms
    
    def propagate(self, elements):
        for element in elements:
            for ray in self.rays:
                element.propagate_ray(ray)
        self.update_pos()
        
    
class OpticalElement:
    def __init__(self, refraction = False):
        self.refraction = refraction
        self.curvature = 0

    def propagate_ray(self, ray):
        'propagates a ray through the optical element'
        intercept = self.intercept(ray)
        if type(intercept) == type(None): # no intercept - no interaction
            return
        
        if self.refraction:
            normal = self.normal(intercept, ray)
            newDirection = snell.Snell(ray.k(), normal, self.n1, self.n2)
            if type(newDirection) == type(None): # this is how the snell module handled total internal reflection - updated to include TIR
                ray.append(pos = intercept, direction = np.array([0, 0, 0])) # terminates ray
                return
            ray.append(intercept, newDirection)
            return
        ray.append(intercept, direction = ray.k())
    
    def find_l(self, ray):
        '''
        finds the factor by which the ray's direction vector
        must be multiplied in order to reach the intercept point.
        '''
        if self.curvature == 0: # special case of plane surface: R = infinity so this part prevents divide by zero error
            l = (self.pos[2] - ray.p()[2]) / ray.k()[2]
            if l < 0 or l == 0:
                return None
            return l
        R = 1 / self.curvature
        circle_center = self.pos.copy()
        try:
            circle_center[2] += R # defines center of curvature
            
        except OverflowError: # this problem came up with values close to 0 for curvature
            l = (self.pos[2] - ray.p()[2]) / ray.k()[2]
            if l < 0 or l == 0:
                return None
            return l
        
        r = ray.p() - circle_center # r = position vector of ray relative to center of curvature
        
        if (np.dot(r, ray.k())**2 - (np.dot(r, r) - R**2)) < 0: # in this case there will be no intercept
            return None
        
        l1 = - np.dot(r, ray.k()) + np.sqrt(np.dot(r, ray.k())**2 - (np.dot(r, r) - R**2)) # l is the solution to a quadratic eqn, therefore two possible values
        l2 = - np.dot(r, ray.k()) - np.sqrt(np.dot(r, ray.k())**2 - (np.dot(r, r) - R**2))
        
        if l1 < 0 and l2 < 0: # no negative l's allowed, our ray only propagates forward
            return None
        
        elif round(l1, 6) < 0 or round(l2, 6) < 0:
            l = max(l1, l2)
        
        
        elif (self.curvature > 0 and ray.k()[2] > 0) or (self.curvature < 0 and ray.k()[2] < 0): # makes sure the ray hits the right 'side' of the sphere
            l = min(l1, l2)
        elif (self.curvature < 0 and ray.k()[2] > 0) or (self.curvature > 0 and ray.k()[2] < 0):
            l = max(l1, l2)
        
        if l < 1e-6: # l == 0 or l < 0 was not sensitive to floating point/rounding errors
        #prevents backwards ray tracing or accidentally repeating the same element
            return None
        
        return l
    
    def intercept(self, ray):
        '''
        finds the intercept of the ray with the optical element
        '''
        if np.all(ray.k() == np.array([0, 0, 0])):
            #terminated ray
            #should no longer be necessary due to implementation of total internal reflection
            return None
        
        l = self.find_l(ray)
        if type(l) == type(None):
            return None
        
        intercept = ray.p() + l * ray.k()
        
        try:
            if (intercept[1] > self.pos[1] + self.r_ap) or (intercept[1] < self.pos[1] - self.r_ap):
                return None
        except AttributeError: # object doesn't have an aperture radius
            return intercept
        
        return intercept
    
    

class SphericalRefraction(OpticalElement):
    '''an optical element with a spherical curved surface
    
    Parameters
    ------
    pos: list/array
    the position of the element (intercept of surface and optical axis; NOT center of curvature)
    
    curvature: int/float
    1/R where R is the radius of the spherical surface
        
    n1: int/float
    refractive index of material before surface
    
    n2: int/float
    refractive index of material behind surface
    
    r_ap: the aperture radius of the surface
    '''
    def __init__(self, pos = [0, 0, 0], curvature = 1, n1 = 1, n2 = 1, r_ap = 1):
        OpticalElement.__init__(self, refraction = True)
        self.pos = np.array(pos)
        self.curvature = curvature
        self.n1 = n1
        self.n2 = n2
        self.r_ap = r_ap
    
    def normal(self, intercept, ray):
        'finds the vector normal to the surface at the intercept point'
        
        if self.curvature == 0:
            normal = np.array([0, 0, np.sign(ray.k()[2])])
            return normal
        
        R = 1 / self.curvature
        circle_center = self.pos.copy()
        try:
            circle_center[2] += R
        except OverflowError: # problems with curvatures close to zero
            normal = np.array([0, 0, np.sign(ray.k()[2])])
            return normal
        
        normal = circle_center - intercept
        normal /= np.sqrt(sum(normal**2)) #normalize hehehe
        
        if np.sign(R) != np.sign(ray.k()[2]):
            normal = -normal # z component of normal should be pointing along ray direction - not at center of curvature
        return normal
    
    
class OutputPlane(OpticalElement):
    'A plane perpendicular to the optical axis at the specified distance (z)'
    def __init__(self, z):
        OpticalElement.__init__(self)
        self.pos = np.array([0, 0, z])
    #propagate/intercept methods are inherited
    
    

class Lens(OpticalElement):
    'A composition of multiple refractive  to simulate a lens object'
    # a lens HAS refractive surfaces (composition) and IS AN optical element (inheritance)
    def __init__(self, pos = [0, 0, 100], width = 5, front_curv = .02, back_curv = -.02, n1 = 1, n2 = 1.5, r_ap = 100):
        
        OpticalElement.__init__(self, refraction = True)
        
        self.pos = pos
        self.width = width
        self.front_curv = front_curv
        self.back_curv = back_curv
        self.n1 = n1
        self.n2 = n2
        
        self.front = SphericalRefraction(pos, front_curv, n1, n2, r_ap)
        self.back = SphericalRefraction(np.array([pos[0], pos[1], pos[2] + width]), back_curv, n2, n1, r_ap)
    
    def find_focus(self):
        '''
        finds focus by tracing a ray close to the optical axis
        
        only works for lenses whose optical axis is (x, y) = (0, 0)
        
        a better method is to find the position of the minimum RMS beam spread
        of a ray bundle that has passed through the lens
        '''
        try:
            test_ray = Ray(pos = [0, 0.0001/max(abs(self.front_curv), abs(self.back_curv)), 0])
        except ZeroDivisionError:
            print('no focus: double plane lens')
            return None
        
        self.front.propagate_ray(test_ray)
        self.back.propagate_ray(test_ray)
        x, y, z = test_ray.p()
        v_x, v_y, v_z = test_ray.k()
        z_focus = z - (y/v_y) * v_z
        return z_focus
    
    def set_refractive_indeces(self, n1, n2):
        self.n1 = n1
        self.n2 = n2
        self.front = SphericalRefraction(self.pos, self.front_curv, n1, n2, self.r_ap)
        self.back = SphericalRefraction(np.array([self.pos[0], self.pos[1], self.pos[2] + self.width]), self.back_curv, n2, n1, self.r_ap)
    
    def propagate_ray(self, ray):
        self.front.propagate_ray(ray)
        self.back.propagate_ray(ray)
        
    
    
#graphics functions


def raytrace(ax, vertices, color = 'blue'):
    'traces the vertices of each ray on a plot'
    ys = [point[1] for point in vertices]
    zs = [point[2] for point in vertices]
    ax.plot(zs, ys, linestyle = 'solid', marker = '.', color = color)
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('y (mm)')
    

def spot_diagram(ax, bundle):
    'plots the projections of the ray positions on the x-y plane'
    for ray in bundle.rays:
        ax.plot(ray.p()[0], ray.p()[1], '.', color = 'blue')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')


