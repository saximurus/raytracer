# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 14:20:54 2021

@author: victo
"""

import numpy as np
import scipy.optimize as op
import raytracer
import matplotlib.pyplot as plt

n_glass = 1.5168

def beam_spread(curvatures, focal_length):
    '''
    finds the RMS spread of the rays from the center of the beam
    for a specified lens at a specified focal length
    
    Parameters
    -----
    curvatures: tuple
    specifies the front and back curvatures of the lens to be used
    
    focal length: float/int
    specifies the distance from the lens at which the spread will be measured
    
    returns RMS beam spread
    '''
    
    lens_pos = 100
    detector_pos = lens_pos + focal_length
    
    lens = raytracer.Lens(np.array([0, 0, lens_pos]), front_curv = curvatures[0], back_curv = curvatures[1], n2 = n_glass)
    detector = raytracer.OutputPlane(detector_pos)
    
    ray_bundle = raytracer.RayBundle(rayDensity = 0.5, radius = 5)
    
    ray_bundle.propagate([lens, detector])
    
    return ray_bundle.spread()


def find_best_curv(focal_length):
    'returns the curvatures of the front and back of the lens that best minimize the spread of the beam at the focal point'
    
    curvature_guess = np.array([0.16, -0.003])
    #multiple attempts at optimization; lines that are commented out are attempts at using other functions
    #best_curv = op.minimize(beam_spread, curvature_guess, args = focal_length, bounds = ((0, np.inf),(-np.inf, 0)), tol = 1e-8)
    #attempted to use op.fmin_tnc but accidentally used same line for testing :/
    best_curv = op.basinhopping(beam_spread, curvature_guess, 5, 1, minimizer_kwargs = {'args': focal_length, 'bounds': ((0, np.inf), (-np.inf, 0))}, niter_success = 500)
    return best_curv


def contour_plot(curv_range, focal_length, n, name = 'lens_opt_contours'):
    '''
    Creates a contour plot where the height is the RMS beam spread 
    of a particular lens and the axes denote the curvatures of the lens
    
    Also saves the output values to a csv file

    Parameters
    ----------
    curv_range : array-like, 2 positive values
        Minimum curvature and maximum curvature for the contour plot
        The back lens will take on the negative values
    focal_length : int/float
        The distance from the lens at which the RMS beam spread will be measured.
    n : int
        The number of points to calculate for the contour plot.
        The number of points on each side is sqrt(n)
    name : string, optional
        Name of the output file containing the contour plot
        (NOT HEIGHT VALUES), no '.pdf' required.
        The default is 'lens_opt_contours'
    '''
    side_length = int(np.sqrt(n))
    
    front_curvs = np.linspace(curv_range[0], curv_range[1], side_length)
    back_curvs = np.linspace(-curv_range[0], -curv_range[1], side_length)
    
    heights = []
    
    for i, back_curv in enumerate(back_curvs):
        height_row = []
        for front_curv in front_curvs:
            height = beam_spread([front_curv, back_curv], focal_length)
            height_row.append(height)
        heights.append(height_row)
        print('%.3f percent done' % (i * 100/side_length))
    
    heights = np.array(heights)
    
    ######
    # Don't forget to change the name each time the function is run !!!
    #
    np.savetxt('beam_spread_contour_height_FOR_MARKING.csv', heights, delimiter = ',')
    #
    ######
    
    fig, ax = plt.subplots()
    
    cont_levels = [0.005, 0.020, 0.080, 0.32]
    level_colors = [(0.7, 0.05, 0.0), (0, 0.4, 0.5), (0, 0.25, 0.5), (0, 0, 0.5)]
    
    cs = ax.contour(front_curvs, back_curvs, heights,  levels = cont_levels, colors = level_colors)
    ax.set_xlabel(r'Curvature of front ($mm^{-1}$)')
    ax.set_ylabel(r'Curvature of back ($mm^{-1}$)')
    ax.set_title('Contour Plot of Beam Spread')
    
    for i, val in enumerate(cont_levels):
        #sets a label for each contour height
        cs.collections[i].set_label(str(val) + 'mm')
    
    ax.legend(loc = 'lower right')
    plt.savefig(name + '.pdf', bbox_inches = 'tight')



def saved_contour_plot(data_file, file_name):
    """
    Creates a contour plot out of saved data and saves the plot.
    The values of the sides are preset to fit the data
    that came with the project.

    Parameters
    ----------
    data_file : string
        The name of the file containing the saved data
        
    file_name : string + '.pdf'
        The name of the file the plot gets written to. Important to save
        as pdf to preserve detail.

    """

    data = np.loadtxt(data_file, delimiter = ',')
    
    side_length = len(data[0])
    
    curv_range = [0.001, 0.020]
    
    front_curvs = np.linspace(curv_range[0], curv_range[1], side_length)
    back_curvs = np.linspace(-curv_range[0], -curv_range[1], side_length)
    
    cont_levels = [0.005, 0.020, 0.080, 0.32]
    level_colors = [(0.7, 0.05, 0.0), (0, 0.4, 0.5), (0, 0.25, 0.5), (0, 0, 0.5)]
    
    fig, ax = plt.subplots(tight_layout = True, figsize = [8, 8])
    cs = ax.contour(front_curvs, back_curvs, data, levels = cont_levels, colors = level_colors)
    ax.set_xlabel(r'Curvature of front ($mm^{-1}$)')
    ax.set_ylabel(r'Curvature of back ($mm^{-1}$)')
    ax.set_title('Figure 11: Contour Plot of Beam Spread')
    
    
    for i, val in enumerate(cont_levels):
        cs.collections[i].set_label(str(val) + 'mm')
    ax.legend(loc = 'lower right')
    
    
    plt.savefig(file_name, bbox_inches = 'tight')




#testing function

def cost_func(curv, curvs, focal_length, facing = 'output'):
    'creates cost function for one side of the lens at a constant curvature. Facing determines which lens has the varied curvature.'
    spreads = []
    if facing == 'output':
        for curv_back in curvs:
            spreads.append(beam_spread(np.array([curv, curv_back]), focal_length))
    if facing == 'input':
        for curv_front in curvs:
            spreads.append(beam_spread(np.array([curv_front, curv]), focal_length))
    plt.plot(curvs, spreads, 'x')
    
