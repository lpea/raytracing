# raytracing
Simple raytracer greatly inspired by [Seb Lague's video tutorial](https://www.youtube.com/watch?v=Qz0KTGYJtUk) and [Alan Wolfe's amazing blog post series](https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/).

Originally written in Python, the algorithm was also ported to C++ (CPU only) and C++ with GLSL shaders using the Ogre framework.

## Dependencies

### Python

The script Python/rt.py depends on the `opencv` and `numpy` python packages.

### C++

The project in C++/ requires the OpenCV library (`opencv-devel` package on Fedora).

### Shaders

The project in Shader/ requires the Ogre library (`ogre-devel` package on Fedora).

## General Principle

An image is formed by simulating the path of the light that hits each pixel. The key principle is that the path of the light is calculated backwards compared to the actual physical phenomenon. Instead of going from a light source to the sensor, virtual rays go from the sensor and bounce on objects before reaching a light source.

For each pixel a direction is computed, along which a ray is cast until it hits the first object along its path. Its light and color information is updated based on the color of the object, then its direction is changed to account for reflection. Again, the algorithm determines which object is hit along its new path, and so on, until either no object is found on its path or a certain number of bounces 

### Diffuse vs. specular reflections

Depending on an object's property (such as the material it is made of, a paint coating, the roughness of its surface, etc.) it might reflect light either in a very "mirror-like" way, which makes it look shiny or metallic, or in a diffuse way, meaning that the light is scattered with no privileged direction. The first type of reflection is called *specular* and the second is *diffuse*. A material may exhibit a combination of both specular and diffuse reflections.

The type of reflection determines the color of the light. Usually, only the diffuse component is colored and the specular is essentially white (think of the reflections on a cherry for example). Metals however can have distinctive colored specular reflections.

## Basic Algorithm

### Single image formation

For each pixel:
1. Determine the direction of the ray to cast (depends on the pixel location and field of view).
2. Find out the closest object from the scene that intersects with the light ray.
3. Update the light content (intensity, color) of the ray.
4. Calculate the new direction after reflection.
5. Go back to step 2. or break if no object is hit or after a given number of bounces.

### Image averaging

Since diffuse reflections randomly change the direction of the ray after hitting an object, a single image contains a lot of noise and does not accurately reflect the color and luminosity of objects. Therefore, several images must be generated and combined together in order to accumulate enough information for each pixel, a bit like a photoreceptor accumulates photons.

## Parameter tuning

A few parameters can have a dramatic effect on the final result.

The **number of frames** that are averaged together has a significant effect on the image "smoothness": a high number of frames lowers the noise level. It has no effect on the light level (in the sense of "exposure").

The **surface of emissive objects** and their **intensity of emission**, obviously. Less light or a lower illumination means that the scene illumination will be lower.

An **Exposure** correction can/must be applied after tone mapping.

## Traps I've fallen into

* I initially set the intensity of the light source to 1.0, thinking it was some sort of hard limit, making the scene way too dark. But the intensity can be set arbitrarily high as the "color" at that point is on an open-ended scale.
* The albedo of the source material was set to white, which allowed a same ray to hit the source multiple times, boosting the rays "strength". The albedo of a source must be set to black to prevent this effect.
* The random sampling of a vector on the unit sphere was wrongly implemented by sampling theta and phi and converting from spherical coordinates to cartesian, but this method does not give a uniform sampling.
* The random jitter used for anti-aliasing was sampled on a square, not a circle.
* The orientation of the normal was not calculated correctly for planar surfaces, it must depend on the side of the surface the ray comes from.
* Exposure correction, tone mapping and sRGB conversion were applied on rendered frames before averaging them together but it must be applied to the displayed result only.

## TODO

* Fuse **emission color** and **emission intensity**?
* What difference between **specular probability** and **roughness**?
* Rectangular surfaces are implemented using two triangles. Maybe there's a more efficient method?

## Resources

* https://www.youtube.com/watch?v=Qz0KTGYJtUk
* https://github.com/SebLague/Ray-Tracing
* https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
* https://raytracing.github.io/
