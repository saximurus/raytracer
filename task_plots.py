# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 23:36:51 2021

@author: victo
"""

# run through all the plots required for the ray tracing project

# each task has its own function - plots for each task should be independently reproducable on their own set of axes

import numpy as np
import matplotlib.pyplot as plt
import raytracer
import lens_investigation # task 15 was large enough to get its own file
import lens_optimization as lensOpt

n_glass = 1.5168

def propagate_all(elements, rays):
    for element in elements:
        for ray in rays:
            element.propagate_ray(ray)

def task9(subplot):
    'creates the plot for task 9'
    lens = raytracer.SphericalRefraction(pos = [0, 0, 100], curvature = 0.03, n1 = 1, n2 = 1.5, r_ap = 100)
    detector = raytracer.OutputPlane(250)

    rays = [raytracer.Ray(pos = [0, i, 0]) for i in np.linspace(-0.1, 0.1, 5)] # creates evenly spaced rays
    elements = [lens, detector]

    propagate_all(elements, rays)

    for ray in rays:
        raytracer.raytrace(subplot, ray.vertices())
        subplot.set_title('Task 9 - Tracing a few rays')

def task11(subplots):
    '''
    creates plots for task 11 - imaging tests
    subplots - a list/array of two matplotlib Axes objects
    '''
    
    
    ''
    #imaging test plot for a single refractive surface
    lens_front = raytracer.SphericalRefraction(pos = [0, 0, 100], curvature = 0.03, n1 = 1, n2 = 1.5, r_ap = 100)
    detector = raytracer.OutputPlane(500)
    
    elements = [lens_front, detector]
    rays = [raytracer.Ray(direction = [0, np.sin(i), np.cos(i)]) for i in np.linspace(-0.01, 0.01, 10)]
    
    propagate_all(elements, rays)
    
    for ray in rays:
        raytracer.raytrace(subplots[0], ray.vertices())
        subplots[0].set_title('Task 11 - Imaging Test')
    
    #thin lens formula test plot
    lens_front = raytracer.SphericalRefraction(pos = [0, 0, 200], curvature = 0.03, n1 = 1, n2 = 1.5, r_ap = 100)
    lens_back = raytracer.SphericalRefraction(pos = [0, 0, 205], curvature = 0, n1 = 1.5, n2 = 1, r_ap = 100)
    detector = raytracer.OutputPlane(500)
    
    elements = [lens_front, lens_back, detector]
    rays = [raytracer.Ray(direction = [0, np.sin(i), np.cos(i)]) for i in np.linspace(-0.01, 0.01, 10)]
    
    propagate_all(elements, rays)
    
    for ray in rays:
        raytracer.raytrace(subplots[1], ray.vertices())
        subplots[1].set_title('Task 11 - Thin Lens Imaging Test\nExpected f = 66.67mm')

def task12(subplot):
    'creates the plot for task 12 - testing the thin lens formula'
    ray_bundle = raytracer.RayBundle(rayDensity = .5, radius = 5)
    
    lens = raytracer.SphericalRefraction(pos = [0, 0, 100], curvature = 0.03, n1 = 1, n2 = 1.5, r_ap = 100)
    detector = raytracer.OutputPlane(200)
    
    elements = [lens, detector]
    propagate_all(elements, ray_bundle.rays)
    
    for ray in ray_bundle.rays:
        raytracer.raytrace(subplot, ray.vertices())
        subplot.set_title('Task 12 - Raytracing a uniform beam\nwith radius 5mm')
        
        
def task13(subplots):
    '''
    creates the plots for task 13 - showing the spot diagram at the start and the focal point
    
    requires 2 subplots
    '''
    ray_bundle = raytracer.RayBundle(rayDensity = 1, radius = 5)
    
    lens = raytracer.SphericalRefraction(pos = [0, 0, 100], curvature = 0.03, n1 = 1, n2 = 1.5, r_ap = 100)
    detector = raytracer.OutputPlane(200)
    
    elements = [lens, detector]
    
    raytracer.spot_diagram(subplots[0], ray_bundle)
    subplots[0].set_title('Task 13 - Spot Diagrams\nAt z = 0mm:')
    
    propagate_all(elements, ray_bundle.rays)
    
    raytracer.spot_diagram(subplots[1], ray_bundle)
    subplots[1].set_title('Task 13 - Spot Diagrams\nAt focal plane (z = 200mm)')
    
def task15(subplots):
    '''
    create the plots for task 15 - investigating the performance of a plano-convex lens - a plot of RMS spread against initial beam radius
    
    requires 2 subplots
    '''
    lens_investigation.plano_convex_trace('input', subplots[0])
    lens_investigation.plano_convex_trace('output', subplots[1])
    
    lens_investigation.spot_sizes([5, 10], 10, 'input', subplots[2])
    lens_investigation.spot_sizes([5, 10], 10, 'output', subplots[2])
    
    subplots[2].legend(['Convex side facing input', 'Convex side facing output'])
    subplots[2].set_title('Task 15 - Performance of a plano-convex lens\ndepending on orientation')
    
def lens_opt(subplot):
    'optimizes a lens and plots the performance of the lens'
    curv = lensOpt.find_best_curv(100) # can return different values for different initial guesses - see lens_optimization module
    lens_investigation.spot_sizes([5, 10], 10, 'ownlens', subplot, lens = raytracer.Lens(front_curv = curv.x[0], back_curv = curv.x[1], n2 = n_glass), f = 100)
    print('Optimized Curvatures\nfront: %.5e mm^-1\nback: %.5e mm^-1' % tuple(curv.x))
    

def add_fig_title(fig_number, ax):
    'adds Figure x: to the title of a plot'
    title = ax.get_title()
    title = 'Figure ' + str(fig_number) + ': ' + title
    ax.set_title(title)
    
def main():
    'creates a pdf with all plots required in the project - Figure titles added for report'
    
    fig, axes = plt.subplots(5, 2, figsize = [10, 15], tight_layout = True)

    task9(axes[0][0])
    print('Task 9 Plot Complete')
    
    task11([axes[0][1], axes[1][0]])
    print('Task 11 PLots Complete')
    
    task12(axes[1][1])
    print('Task 12 Plot Complete')
    
    task13([axes[2][0], axes[2][1]])
    print('Task 13 Plots Complete')
    
    # might take a while
    task15([axes[3][0], axes[3][1], axes[4][0]])
    print('Task 15 Plots Complete')

    # will take a while
    lens_investigation.spot_sizes([5, 10], 10, 'input', axes[4][1])
    lens_opt(axes[4][1])

    axes[4][1].legend(['Plano-convex lens facing input', 'Optimized lens'])
    axes[4][1].set_title('Performance of the optimized\nlens compared to the plano-convex lens')

    for i, ax in enumerate(axes.flatten()):
        add_fig_title(i + 1, ax)
    
    plt.savefig('task_plots.pdf')
    print('Done')



main()










