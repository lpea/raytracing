# raytracing
Simple raytracer in Python

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

The **number of frames** that are averaged together has a significant effect on the image "smootheness": a high number of frames lowers the noise level. It has no effet on the light level (in the sense of "exposure").

The **maximum number of bounces** determines how many bounces are calculated for a single light ray. After a certain number of bounces, we can consider that additional ones do not contribute in a significative way to its light intensity and color. Increasing this value allows to better calculate multiple reflections. However changing this value also affects the light intensity, like changing the exposure time or sensitivity on a camera.
