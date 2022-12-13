# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:55:21 2021

@author: victo
"""




import numpy as np
import raytracer


n_glass = 1.5168

def plano_convex_trace(facing, subplot):
    'plots the ray trajectories of the lens specified in task 15'
    
    if facing == 'input': # the convex part of the lens is facing the incoming rays
        lens = raytracer.Lens(back_curv = 0, n2 = n_glass)
    elif facing == 'output': # the convex part of the lens is facing the output plane
        lens = raytracer.Lens(front_curv = 0, n2 = n_glass)
    
    ray_bundle = raytracer.RayBundle(rayDensity = 0.5, radius = 5)
    
    detector = raytracer.OutputPlane(300)
    
    ray_bundle.propagate([lens, detector])
    
    for ray in ray_bundle.rays:
        raytracer.raytrace(subplot, ray.vertices())
    
    subplot.set_title('Task 15 - Ray trajectories for a plano-convex\nlens with convex side facing ' + facing)

def spot_sizes(r_range, beam_number, facing, subplot, lens = None, f = 'find f'):
    'plots the performance of either a plano-convex or a specified lens'
    if facing == 'input':
        lens = raytracer.Lens(back_curv = 0, n2 = n_glass) # other parameters are specified by standard lens in raytracer module
    elif facing == 'output':
        lens = raytracer.Lens(front_curv = 0, n2 = n_glass)
    elif facing == 'ownlens':
        pass
    
    if type(f) == int: # allows for the focal length to be specified instead of found paraxially
        focal_detector = raytracer.OutputPlane(lens.pos[2] + f)
    else:
        focal_detector = raytracer.OutputPlane(lens.find_focus())
    
    elements = [lens, focal_detector]
    
    radii = np.linspace(r_range[0], r_range[1], beam_number)
    ray_bundles = [raytracer.RayBundle(rayDensity = 1, radius = i) for i in radii]
    
    for ray_bundle in ray_bundles:
        ray_bundle.propagate(elements)
    
    spreads = np.array([ray_bundle.spread() for ray_bundle in ray_bundles])
    subplot.plot(radii, spreads)
    subplot.set_xlabel('Initial beam radius (mm)')
    subplot.set_ylabel('RMS beam spread\nat focal plane (mm)')


