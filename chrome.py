# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:27:21 2021

@author: victo
"""

#chromatic aberration and rainbow extensions

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import mirrors
import raytracer
import snell
import copy
import scipy.ndimage as nd

def weighting_func(wavelength, mean, spread):
    'normal distribution function to find weighting value for each RGB value at a specific wavelength of light'
    x = wavelength
    exponent = ((x-mean)/spread)**2
    return np.e ** (-(exponent)/2)


def wavelength_to_rgb(wavelength):
    'finds the rgb values for a specific wavelength'
    
    #largely a cosmetic function and not very important to scientific results
    
    l = wavelength
    
    #weighting func is a Gaussian
    #values for mean are loosely based on actual values from a quick Google
    #standard deviations are based on what looked good
    red = weighting_func(l, 700e-9, 50e-9)
    
    red += 0.6 * weighting_func(l, 380e-9, 25e-9) # red at low wavelengths to make violet
    #don't go too low or it will look red and not violet
    
    green = 3 * weighting_func(l, 600e-9, 50e-9) # the factor of three is arbitrary but it looked good
    blue = weighting_func(l, 500e-9, 40e-9)
    result = np.array([red, green, blue])
    result /= max(result)
    red, green, blue = result
    return red, green, blue


def n_glass(wavelength):
    '''
    implements Sellmeier equation for borosilicate glass
    
    wavelength is in SI units
    returns refractive index at that wavelength
    '''
    l = wavelength * 10**6
    A = 1.43131380
    B = 0.84014624
    C = 1.28897582e-2
    D = 1.28897582
    E = 100
    
    #source for values: G. Ghosh, “Sellmeier coefficients and dispersion of thermo-optic coeffi-cients for some optical glasses,”Applied optics (2004), vol. 36, no. 7, pp.1540–1546, 1997.
    n_squared = A + B / (1 - C/l**2) + D/(1 - E/l**2)
    return np.sqrt(n_squared)

water_n_table = np.loadtxt('water_rf_data.txt')
water_n_table[:,0] *= 1e-6

water_n_dict = {water_n_table[:,0][i]: water_n_table[:,1][i] for i in range(len(water_n_table))}
#dictionary used so wavelengths can be the index

def n_water(wavelength):
    'finds refractive index of water at a specific wavelength using a table of values - source in report'
    if wavelength in water_n_table[:,0]:
        n = water_n_dict[wavelength]
        return n
    else:
        # linear interpolation for in-between values of wavelength
        
        for i, l in enumerate(water_n_table[:,0]):
            #set i to be the index of the wavelength just below the desired one
            if wavelength < l:
                # the variable i is saved when the loop is broken
                break
        slope = (water_n_table[:,1][i]-water_n_table[:,1][i-1])/(water_n_table[:,0][i]-water_n_table[:,0][i-1]) # finds the gradient between the two points to either side of the desired wavelength
        n = slope * (wavelength - water_n_table[:,0][i]) + water_n_table[:,1][i] # linear interpolation
        
        return n

    
        
class color_Ray(raytracer.Ray):
    'a color version of the classic Ray'
    def __init__(self, pos = [0, 0, 0], direction = [0, 0, 1], wavelength = 600e-9):
        raytracer.Ray.__init__(self, pos, direction)
        self.wavelength = wavelength
        self.intensity = 1.
    
    def angle(self, axis):
        ''''
        finds the component about the 'axis' parameter of
        the angle between ray.k() and the optical axis
        i. e. to find altitude and azimuth of rays incident on a detector
        
        useful for rainbows
        
        axis: 0 = x-axis, 1 = y-axis
        '''
        tan_angle = self.k()[axis]/self.k()[2]
        angle = np.arctan(tan_angle)
        return angle
    



class color_RayBundle(raytracer.RayBundle):
    '''
    a color version of the classic RayBundle
    - has several rays with different wavelengths at each position
    '''
    def __init__(self, pos = [0, 0, 0], direction = [0, 0, 1], rayDensity = 1, radius = 1, color = 'white'):
        raytracer.RayBundle.__init__(self, pos, direction, rayDensity, radius)
        ray_positions = self._find_bundle_start_positions(rayDensity, radius)
        self.color = color_RayBundle._color_to_wavelengths(color)
        self.rays = []
        self.k = direction
        wavelengths = list(np.linspace(480e-9, 800e-9, 5))
        wavelengths = [430e-9] + wavelengths
        wavelengths = np.array(wavelengths)
        for position in ray_positions:
            for wavelen in wavelengths:
                self.rays.append(color_Ray(pos = position, direction = direction, wavelength = wavelen))
        
    def _color_to_wavelengths(color):
        '''
        finds the wavelength for an rgb value
        largely unused as most things were done with white light
        
        kinda useless but fun to play with ngl
        '''
        color_dict = {'white': np.array([1, 1, 1]),
                      'red': np.array([1, 0, 0]),
                      'green': np.array([0, 1, 0]),
                      'blue': np.array([0, 0, 1])}
        try:
            color = color_dict[color]
        except:
            print('Not a valid color')
            return None
        
        return np.array([color[0]*700e-9, color[1]*600e-9, color[2]*500e-9])
    

class ColorSphericalRefraction(raytracer.SphericalRefraction):
    'Spherical refractive surface that considers the wavelength of the ray'
    def __init__(self, pos, curvature, r_ap, material, facing):
        self.material = material
        self.facing = facing
        raytracer.SphericalRefraction.__init__(self, pos, curvature, r_ap = r_ap)
    
    def set_indeces(self, ray):
        try:
            l = ray.wavelength
        except AttributeError:
            l = 500e-9
        if self.material == 'glass':
            n = n_glass(l)
        if self.material == 'water':
            facing = self.check_facing(ray)
            self.facing = facing
            n = n_water(l)
            
        if self.facing == 'in':
            self.n2 = n
            self.n1 = 1
        if self.facing == 'out':
            self.n1 = n
            self.n2 = 1
        
    def propagate_ray(self, ray):
        'propagation taking into account the refractive index of the material'
        
        self.set_indeces(ray)
        
        raytracer.SphericalRefraction.propagate_ray(self, ray)
        
        try:
            reflectivity = self.get_reflectivity(ray)
            
            ray.intensity *= (1. - reflectivity)
        except:
            pass
    
    
    def check_facing(self, ray):
        'method for checking whether a ray is entering or exiting a raindrop'
        if ray.k()[2] < 0 and self.curvature > 0:
            return 'out'
        elif ray.k()[2] > 0 and self.curvature > 0:
            return 'in'
        elif ray.k()[2] < 0 and self.curvature < 0:
            return 'in'
        elif ray.k()[2] > 0 and self.curvature < 0:
            return 'out'
        else: # just prevents an error in the case that ray.k() == 0
            return self.facing
    
    def get_reflectivity(self, ray):
        self.set_indeces(ray)
        
        normal = self.normal(self.intercept(ray), ray)
        R = snell.Fresnel(ray.k(), normal, self.n1, self.n2)
        if R > 1:
            normal = -normal
            R = snell.Fresnel(ray.k(), normal, self.n1, self.n2)
        return R
    
    
class ColorLens(raytracer.Lens):
    'a color version of the ray tracer Lens'
    def __init__(self, pos = [0, 0, 100], width = 5, front_curv = 0.02, back_curv = -0.02, r_ap = 100, material = 'glass'):
        self.pos = pos
        self.width = width
        self.front_curv = front_curv
        self.back_curv = back_curv
        self.material = material
        self.front = ColorSphericalRefraction(pos, front_curv, r_ap, material = 'glass', facing = 'in')
        self.back = ColorSphericalRefraction(np.array([pos[0], pos[1], pos[2] + width]), back_curv, r_ap, material = 'glass', facing = 'out')

class Raindrop(raytracer.Lens):
    'a lens starting out with a refractor in front and a reflector in the back'
    def __init__(self, pos = [0, 0, 3], radius = 1):
        self.refraction = True
        self.pos = np.array(pos)
        pos = np.array(pos)
        self.radius = radius
        self.front = ColorSphericalRefraction(pos, 1/radius, radius, 'water', 'in')
        self.back = ColorSphericalRefraction(np.array([pos[0], pos[1], pos[2] + radius * 2]), -1/radius, radius, 'water', 'out')
        #this has been commented out as it is not total internal reflection which causes the reflection
        #the transmittivity/reflectivity at each surface could be a further extension of the project
        #self.back = mirrors.SphericalReflection(np.array([pos[0], pos[1], pos[2] + radius * 2]), -1/radius, radius)
        
        self.front_reflector = mirrors.SphericalReflection(np.array(pos), 1/radius, radius)
        self.back_reflector = mirrors.SphericalReflection(np.array([pos[0], pos[1], pos[2] + radius * 2]), -1/radius, radius)
        
        
    def propagate_ray(self, ray):
        '''
        propagates the ray in the manner required for a rainbow to form
        
        this is not a complete model of reflectivity/transmittivity within
        the raindrop and is only accurate for the most commonly seen rainbows
        '''
        self.front.propagate_ray(ray)
        try:
            reflectivity = self.back.get_reflectivity(ray)
            self.back_reflector.propagate_ray(ray)
            ray.intensity *= reflectivity  
        except:
            self.back_reflector.propagate_ray(ray)
        
        self.front.propagate_ray(ray)
    
    def propagate_properly(self, ray_bundle):
        
        ray_families = []
        
        for ray in ray_bundle.rays:
            iterations = 0
            ray_family = [ray]
            while True:
                if iterations > 7:
                    break
                new_rays = []
                
                intensities = [r.intensity for r in ray_family]
                if max(intensities) < 0.0001:
                    break
                
                for ray_parent in ray_family:
                    
                    l_front = self.front.find_l(ray_parent)
                    l_back = self.back.find_l(ray_parent)
                    
                    def propagate_front():
                        ray_child = copy.deepcopy(ray_parent)
                        reflectivity = self.front.get_reflectivity(ray_parent)
                        ray_child.intensity = ray_parent.intensity * reflectivity
                        self.front.propagate_ray(ray_parent)
                        self.front_reflector.propagate_ray(ray_child)
                        new_rays.append(ray_child)
                    
                    def propagate_back():
                        ray_child = copy.deepcopy(ray_parent)
                        reflectivity = self.back.get_reflectivity(ray_child)
                        ray_child.intensity = ray_parent.intensity * reflectivity
                        self.back.propagate_ray(ray_parent)
                        self.back_reflector.propagate_ray(ray_child)
                        new_rays.append(ray_child)
                    
                    if type(l_front) == type(None) and type(l_back) == type(None):
                        continue
                    
                    elif type(l_back) == type(None):
                        propagate_front()
                    
                    elif type(l_front) == type(None):
                        propagate_back()
                    
                    elif l_front < l_back or l_front == l_back:
                        propagate_front()
                    
                    elif l_front > l_back:
                        propagate_back()
                
                ray_family += new_rays
                iterations += 1
                
            ray_families += ray_family[1:]
            
        ray_bundle.rays += ray_families
                        
                
    def draw(self, ax):
        'graphics function - draws circle on plot representing the surfaces of the raindrop'
        R = self.radius
        front_pos = self.pos
        
        pos = front_pos + R # not entirely sure why this gives the correct position lol
        #the function works and that's the important part
        
        z_arr = np.linspace(pos[2] - R, pos[2] + R)
        
        y_positives = np.sqrt(R**2 - (z_arr - pos[2])**2)
        y_negatives = -1 * y_positives
        
        y_positives += self.pos[1]
        y_negatives += self.pos[1]
        
        ax.plot(z_arr, y_positives, color = 'xkcd:sky blue')
        ax.plot(z_arr, y_negatives, color = 'xkcd:sky blue')


class Observer(raytracer.OutputPlane):
    'an output plane with an aperture radius - meant to represent an eye/human observer'
    def __init__(self, pos, r_ap):
        raytracer.OutputPlane.__init__(self, pos[2])
        self.r_ap = r_ap
        self.pos = np.array(pos)


def color_spot_diagram(ax, bundle):
    'graphics function - creates spot diagram using wavelengths to determine spot color'
    for ray in bundle.rays:
        ax.plot(ray.p()[0], ray.p()[1], ',', color = wavelength_to_rgb(ray.wavelength))
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

def additive_rgb_diagram(ray_bundle, ax_size, size, method = 'position', use_intensity = False):
    '''
    graphics function - creates spot diagram and allows pixels within overlapping
    spots to take on RGB values of both ray colors.
    
    This allows wavelengths to recombine to white light

    Parameters
    ----------
    ray_bundle : the ray bundle for which the spot diagram is being made
    
    ax_size : Tuple (x width, y width)
        The width of the plot in the units plotted, e.g. degrees/mm
        
    size : Tuple (x width, y width)
        The physical size of the plot in inches
        
    method : 'position' or 'axis', optional
        Whether to plot by the position of the rays or their angle
        with respect to either the x or y axes and relative to the
        optical axis. 
        
        The default is 'position'.
    
    use_intensity : bool
        Whether or not to consider the intensity of the ray when plotting

    Returns
    -------
    img_out : array
        The array of RGBA values that describe the image of the plot

    '''
    buf_fig, buf_ax = plt.subplots(figsize = size, dpi = 240)
    # buf_ prefix to describe temporary figure/ax - used to plot and then extract buffer
    wavelengths = []
    # for loop generates list of all wavelengths represented in ray bundle:
    for ray in ray_bundle.rays:
        if ray.wavelength not in wavelengths:
            wavelengths.append(ray.wavelength)
    
    # dpi is 240 pixels per inch
    width = 240 * size[0]
    height = 240 * size[1]
    background_color = np.array(col.to_rgb('xkcd:black'))
    buf_ax.axis('off')
    buf_ax.set_xlim(-ax_size[0]/2, ax_size[0]/2)
    buf_ax.set_ylim(-ax_size[1]/2, ax_size[1]/2)
    buf_fig.patch.set_facecolor(background_color) # gotta love the xkcd Color Survey
    buf_fig.patch.set_edgecolor(background_color)
    buf_fig.canvas.draw()
    tot_array = np.frombuffer(buf_fig.canvas.buffer_rgba(), np.uint8).reshape(height, width, -1).copy()
    tot_array = tot_array.astype('float')
    #intensities = [ray.intensity for ray in ray_bundle.rays]
    
    for i, wavelength in enumerate(wavelengths):
        'create a plot for every wavelength and extract the image buffer'
        
        for ray in ray_bundle.rays:
            if wavelength == ray.wavelength:
                
                if use_intensity:
                    #if intensity is used then the color of the ray changes
                    plot_color = (np.array(wavelength_to_rgb(wavelength)) - background_color) * ray.intensity
                    plot_color = np.clip(plot_color, 0, 1)
                    
                else:
                    plot_color = wavelength_to_rgb(wavelength)
                    
                if method == 'position':
                    buf_ax.plot(ray.p()[0], ray.p()[1], 'o', color = plot_color)
                    buf_ax.set_xlabel('x (mm)')
                    buf_ax.set_ylabel('y (mm)')
                elif method == 'angle':
                    if ray.k()[2] < 0:
                        angle_pos = np.array([ray.angle(axis = 0), ray.angle(axis = 1)])
                        angle_pos *= (180/np.pi)
                        buf_ax.plot(angle_pos[0], angle_pos[1], 'o', color = plot_color)
                        buf_ax.set_xlabel(r'Azimuth ($\degree$)')
                        buf_ax.set_ylabel(r'Altitude ($\degree$)')
                    
                    
        #ensuring the size/background of the plot for each color/wavelength is the same:
        
        buf_ax.set_xlim(-ax_size[0]/2, ax_size[0]/2)
        buf_ax.set_ylim(-ax_size[1]/2, ax_size[1]/2)
        buf_ax.axis('off')
        buf_fig.patch.set_facecolor('black')
        buf_fig.patch.set_edgecolor('black')
        buf_fig.canvas.draw()
        
        img = np.frombuffer(buf_fig.canvas.buffer_rgba(), np.uint8).reshape(height, width, -1).copy()
        img = img.astype('float')
        
        if use_intensity:
            tot_array += img
        elif not use_intensity:
            tot_array = np.maximum(img, tot_array)
    
    #ensuring the array is the tight size
    tot_array = np.clip(tot_array, 0, 255)
    img_out = tot_array.astype(np.uint8) # making sure this gets interpreted correctly
    
    plt.close() # close the temporary plot so it doesn't interfere with the actual plots
    return img_out

    



def investigate(raindrop, n):
    'testing function'
    #also demonstrates why the radius of ray bundles hitting raindrops in chrome_plots.py is < 1.0 - preventing TIR and higher order rainbows
    rays = []
    
    for i in np.linspace(0, 1, n):
        pos_rays = []
        for wavelength in np.linspace(400e-9, 800e-9, 10):
            pos_rays.append(color_Ray(pos = [0, i, 0], wavelength = wavelength))
        rays.append(pos_rays)
    detector_positions = [2, 7]
    detectors = [raytracer.OutputPlane(pos) for pos in detector_positions]
    
    fig, ax = plt.subplots(n, 1, tight_layout = True, figsize = [4, 2*n])
    ax[0].set_title('Investigating a raindrop')
    
    for i, ray_list in enumerate(rays):
        for ray in ray_list:
            raindrop.propagate_ray(ray)
            for detector in detectors:
                detector.propagate_ray(ray)
            raytracer.raytrace(ax[i], ray.vertices(), color = wavelength_to_rgb(ray.wavelength))
        raindrop.draw(ax[i])
    
    #plt.savefig('test_raindrop2.pdf')
    plt.show()