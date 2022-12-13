# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:41:58 2021

@author: victo
"""
import matplotlib.pyplot as plt

import numpy as np
import raytracer
import chrome

def plot1(ax, file_type):
    'plots the spot diagram of a ray bundle at the focus point of a lens with a refractive index dependent on wavelength'
    
    lens = chrome.ColorLens(front_curv = 0.08, back_curv = -0.08)
    focal_detector = raytracer.OutputPlane(lens.find_focus())
    ray_bundle = chrome.color_RayBundle(rayDensity = 1, radius = 5)
    ray_bundle.propagate([lens, focal_detector])
    if file_type == 'pdf':
        chrome.color_spot_diagram(ax, ray_bundle)
    if file_type == 'png':
        img = chrome.additive_rgb_diagram(ray_bundle, [6, 6], [6, 6])
        ax.imshow(img)
        ax.axis('off')
    ax.set_title('Spot diagram showing chromatic\naberration at the focus point')


def plot2(ax):
    'traces a single color ray through a raindrop to show classic rainbow diagram'
    
    raindrop = chrome.Raindrop()
    detector = raytracer.OutputPlane(2)
    rays = [chrome.color_Ray(pos = [0, 0.75, 0], wavelength = i) for i in np.linspace(400e-9, 800e-9, 20)]
    for ray in rays:
        raindrop.propagate_ray(ray)
        detector.propagate_ray(ray)
        raytracer.raytrace(ax, ray.vertices(), color = chrome.wavelength_to_rgb(ray.wavelength))
    raindrop.draw(ax)
    ax.set_title('A ray traced\nthrough a raindrop')

def plot3(ax):
    'traces several non-color rays through a raindrop to demonstrate bunching of rays at maximum angle'
    rays = []
    for y in np.linspace(0, 1, 40):
        rays.append(raytracer.Ray(pos = [0, y, 0]))
    rays = np.array(rays)
    raindrop = chrome.Raindrop()
    detector = raytracer.OutputPlane(1)
    for ray in rays:
        raindrop.propagate_ray(ray)
        detector.propagate_ray(ray)
        raytracer.raytrace(ax, ray.vertices())
    raindrop.draw(ax)
    ax.set_title('Several rays traced\nthrough a raindrop')
    
def plot4(ax):
    'plots the variation of exiting angle with starting y position for different wavelengths'
    rays = []
    wavelengths = np.linspace(450e-9, 750e-9, 5)
    for wavelength in wavelengths:
        wavelength_rays = []
        for y in np.linspace(0, 0.99, 100):
            wavelength_rays.append(chrome.color_Ray(pos = [0, y, 0], wavelength = wavelength))
        rays.append(wavelength_rays)
    rays = np.array(rays)
    raindrop = chrome.Raindrop()
    
    for ray_list in rays:
        angles = []
        color = chrome.wavelength_to_rgb(ray_list[0].wavelength)
        for ray in ray_list:
            raindrop.propagate_ray(ray)
            angles.append(ray.angle(axis = 1))
        angles = np.array(angles)
        angles *= (180/np.pi)
        
        ax.plot(np.linspace(0, 1, 100), angles, color = color)
        ax.set_xlabel('Initial height of beam (mm)')
        ax.set_ylabel(r'Angle from optical axis ($\degree$)')
    ax.set_title('Variation in angle\nof different wavelengths')
    legend_labels = [('%.f nm' % (i*1e9)) for i in wavelengths]
    ax.legend(legend_labels)

def plot5(ax):
    
    'plots the maximum exit angle against wavelength'
    
    rays = []
    
    for wavelength in np.linspace(380e-9, 800e-9, 50):
        wavelength_rays = []
        for y in np.linspace(0, 1, 100):
            wavelength_rays.append(chrome.color_Ray(pos = [0, y, 0], wavelength = wavelength))
        rays.append(wavelength_rays)
    rays = np.array(rays)
    raindrop = chrome.Raindrop()
    
    max_angles = []
    for ray_list in rays:
        angles = []
        for ray in ray_list:
            raindrop.propagate_ray(ray)
            angles.append(ray.angle(axis = 1))
        angles = np.array(angles)
        angles *= (180/np.pi)
        max_angles.append(max(angles))
        
    ax.plot(np.linspace(380, 700, 50), max_angles)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(r'Maximum Angle (\degree)')
    ax.set_title('Maximum exiting angle\nfor different wavelengths')


def plot6(ax):
    '''
    plots the angles of the rays making up a collimated beam of white light exiting a single raindrop
    -produces same plot as plot7 but siginificantly faster
    '''
    
    raindrop = chrome.Raindrop()
    ray_bundle = chrome.color_RayBundle(rayDensity = 40, radius = 0.92)
    #radius is 0.9 due to edge effects of 1.0 - some wavelengths experience total internal reflection at too high an input. These rays would contribute to higher order rainbows, not the primary ones being simulated.
    raindrop.propagate_properly(ray_bundle)
    img = chrome.additive_rgb_diagram(ray_bundle, [120, 120], [8, 8], method = 'angle', use_intensity = True)
    ax.imshow(img)
    ax.axis('off')
    #ax.set_title('Rainbow')


def plot7(ax):
    '''
    plots the altitude and azimuth of rays hitting a small observation disk
    -meant to imitate a camera/human eye and show what a rainbow looks like
    -produces the same plot as plot6 as the rays hitting the detector are
     a subset of the rays produced in plot6

    '''
    storm_radius = 60
    storm_depth = 40
    
    _dummy_bundle = raytracer.RayBundle(rayDensity = 0.002, radius = storm_radius)
    
    # using the positions of the dummy bundle to find the positions for the raindrops
    positions = [ray.p() for ray in _dummy_bundle.rays]
    
    
    for i in range(int(storm_depth/4)):
        depth_poses = [np.array([ray.p()[0], ray.p()[1], ray.p()[2] + 4*(i+1)]) for ray in _dummy_bundle.rays]
        positions += depth_poses # taking advantage of list addition - this is not an array
    
    raindrops = [(chrome.color_RayBundle(pos = [position[0], position[1], 10], rayDensity = 20, radius = 0.9), chrome.Raindrop(pos = [position[0], position[1], position[2] + 0.8*storm_radius], radius = 1)) for position in positions]
    # each 'raindrop' in the list is a tuple of a raindrop object and the associated ray bundle that will hit it
    
    detector = chrome.Observer([0, 0, 20], 5)
    # detector is an output plane with an aperture radius so only select rays hit it - meant to function like eye
    
    hit_rays = [] # a list of rays that have hit the detector
    for i, pair in enumerate(raindrops):
        ray_bundle = pair[0]
        raindrop = pair[1]
        
        raindrop.propagate_properly(ray_bundle)
        ray_bundle.propagate([detector])
        for ray in ray_bundle.rays:
            if ray.p()[2] == detector.pos[2]:
                hit_rays.append(ray)
        
        if int(i/25) == i/25:
            print('%.3f percent propagated' % (i * 100 / len(raindrops)))
            
    ray_bundle.rays = hit_rays
    total_img = chrome.additive_rgb_diagram(ray_bundle, [100, 100], [6, 6], method = 'angle')
    ax.imshow(total_img)
    ax.axis('off')
    ax.set_title('Visible Rainbow')


'''
#plotting the spot diagram for chromatic aberration:

fig, ax = plt.subplots(1, 1, figsize = [8, 8])

plot1(ax, 'png')

old_title = ax.get_title()
new_title = 'Figure 12:' + old_title
ax.set_title(new_title)

#fig.savefig('chromatic_aberration.png')
plt.show()


#plotting the plots explaining how the rainbow works
fig2, axes = plt.subplots(2, 2, tight_layout = True, figsize = [8, 8])

plot2(axes[0][0])
plot3(axes[0][1])
plot4(axes[1][0])
plot5(axes[1][1])

for i, ax in enumerate(axes.flatten()):
    old_title = ax.get_title()
    new_title = 'Figure ' + str(i + 13) + ': ' + old_title
    ax.set_title(new_title)


#fig2.savefig('rainbow_explanation.pdf')
plt.show()

'''
#plotting the visualization of the rainbow

fig3, ax = plt.subplots(1, 1, figsize = [8, 8], tight_layout = True)

plot6(ax)

#old_title = ax.get_title()
#new_title = 'Figure 17: ' + old_title
#ax.set_title(new_title)

fig3.savefig('higher_order_rainbows.png', bbox_inches = 'tight', pad_inches = 0)

plt.show()

def testing():
    fig, ax = plt.subplots()
    raindrop = chrome.Raindrop()
    ray_bundle = chrome.color_RayBundle(pos = [0, 0.8, 0], rayDensity = 1, radius = 0.9)
    raindrop.propagate_properly(ray_bundle)
    ray_bundle.propagate([raytracer.OutputPlane(1), raytracer.OutputPlane(7)])
    ax.set_xlim(2, 5)
    ax.set_ylim(-2, 2)
    for ray in ray_bundle.rays:
        raytracer.raytrace(ax, ray.vertices())
    
#testing()



'''
READ BEFORE UNCOMMENTING BELOW:

This bit takes a while - it's a full simulation of large numbers of raindrops and large numbers of rays per raindrop;
large numbers ensure that enough rays hit the detector to 'see' the rainbow.

This will produce largely the same plot as plot6 (with a few rays possibly missing)
since the rays caught by the detector are a subset of the rays
produced by the raindrops. All raindrops are identical therefore the angles
of many of the rays produced will be identical.


To reduce the computational workload, reduce the rayDensity of the color ray bundle.
This will also reduce the number of rays hitting the detector.
The figure in the report was created with a rayDensity of 30 rays/mm^2.

More a proof of concept than anything else - but it does work
'''

#fig4, ax = plt.subplots(1, 1, figsize = [8, 8])

#plot7(ax)

#fig4.savefig('visible_rainbow.png')

#plt.show()
