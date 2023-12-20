#!/bin/env python

import cv2
from datetime import datetime
import math
from multiprocessing import Pool, Lock
import numpy as np
import os
import random
import threading
import time


def random_vector_sphere():
    x = random.gauss(0.0, 1.0)
    y = random.gauss(0.0, 1.0)
    z = random.gauss(0.0, 1.0)
    return np.array([x, y, z]) / math.hypot(x, y, z)


def random_vector_circle():
    theta = random.random() * 2 * math.pi
    return np.array([np.cos(theta), np.sin(theta), 0])


def random_vector_hemisphere(normal):
    v = random_vector_sphere()
    return np.sign(normal.dot(v)) * v


def cos_weighted_random_vector_hemisphere(normal):
    # https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#Cosine-WeightedHemisphereSampling
    return normalize(normal + random_vector_sphere())


def normalize(v):
    return v / np.sqrt(v.dot(v))


def reflect(v, normal):
    # Calculate the reflected vector of v given the normal of a surface
    return v - 2 * v.dot(normal) * normal


def lerp(v0, v1, t):
    # t = 0 => v0, t = 1 => v1.
    return (1 - t) * v0 + t * v1


class Sphere:
    def __init__(self, pos, radius, material):
        self.pos = pos
        self.radius = radius
        self.r2 = self.radius * self.radius
        self.mat = material

    def intersect(self, ray):
        v = ray.orig - self.pos
        d = ray.dir
        r2 = self.r2

        # (v + t * d)**2 = r**2
        # d.d * t**2 + 2 * d.v + v.v = r**2
        # d.d = 1
        # discriminant = 4 * d.v - 4 * v.v - r**2)
        # => t = - d.v - sqrt(d.v - v.v - r**2)

        b = d.dot(v)
        # if -b < -r:
        #     # Sphere behind plane, no intersection possible
        #     return None

        c = v.dot(v) - r2
        delta = b**2 - c
        if delta > 0:
            return -b - np.sqrt(delta)  # Smaller root
        return None

    def get_normal(self, pt):
        # Assumes that pt is on the sphere
        return (pt - self.pos) / self.radius


class Plane:
    def __init__(self, pos, normal, material):
        self.pos = pos
        self.normal = normalize(normal)
        self.mat = material

    def intersect(self, ray):
        v = self.pos - ray.orig
        p = ray.dir.dot(self.normal)
        if p != 0:
            return v.dot(self.normal) / p
        else:
            return None

    def get_normal(self, pt):
        # FIXME: should depend on the side of the plane the ray comes from
        return self.normal


def Color(x, y, z):
    return np.array([x, y, z])


BLACK = Color(0.0, 0.0, 0.0)
WHITE = Color(1.0, 1.0, 1.0)
RED = Color(0.0, 0.0, 0.75)
PINK = Color(0.75, 0.5, 0.75)
LIGHT_PINK = Color(1.0, 0.75, 1.0)
LIGHTER_PINK = Color(1.0, 0.875, 1.0)
GREEN = Color(0.0, 0.75, 0.0)
LIGHT_GREEN = Color(0.5, 0.75, 0.5)
LIGHTER_GREEN = Color(0.75, 1.0, 0.75)
BLUE = Color(0.75, 0.0, 0.0)
LIGHT_BLUE = Color(0.75, 0.5, 0.5)
YELLOW = Color(0.0, 0.75, 0.75)
LIGHT_YELLOW = Color(0.5, 0.75, 0.75)
LIGHTER_YELLOW = Color(0.75, 1.0, 1.0)
PURPLE = Color(0.75, 0.0, 0.75)
ORANGE = Color(0.0, 0.5, 0.75)


class Material:
    def __init__(
        self,
        emission_color=WHITE,
        emission_intensity=0.0,
        albedo=BLACK,
        specular_prob=0.0,
        roughness=0.0,
        specular_color=WHITE,
    ):
        # emission color: how much the material glows in RGB
        self.emission_color = emission_color
        self.emission_intensity = emission_intensity
        self.emission_light = emission_color * emission_intensity

        # albedo: what color the material is under white light
        self.albedo = albedo

        # Probability of specular reflection when a ray hits the material
        self.specular_prob = specular_prob
        # A specular ray with roughness = 1.0 will behave as a diffuse ray
        self.roughness = roughness
        self.specular_color = specular_color


class PassiveMaterial(Material):
    # Non-emitting Material
    def __init__(self, *args):
        Material.__init__(self, BLACK, 0.0, *args)


class MatteMaterial(PassiveMaterial):
    # Diffuse material without specular reflection
    def __init__(self, albedo):
        PassiveMaterial.__init__(self, albedo)


def Pos(x, y, z):
    return np.array([x, y, z])


def Dir(x, y, z):
    return np.array([x, y, z])


# fmt: off
planets = [
    Sphere(Pos(-5, -5, 8), 5, Material(WHITE, 1.0, BLACK, 0.0, 0.0)),  # Sun
    Sphere(Pos(3, 2, 4), 1, Material(WHITE, 1.0, BLACK, 0.0, 0.0)),  # Sun #2
    Sphere(Pos(2, -2, 7), 1, PassiveMaterial(LIGHT_GREEN, 0.7, 0.3, LIGHT_GREEN)),
    Sphere(Pos(3, 0, 10), 4, PassiveMaterial(YELLOW, 0.0, 1.0, YELLOW)),
    Sphere(Pos(-2, 2, 20), 5, PassiveMaterial(RED, 0.0, 1.0, RED)),
    Sphere(Pos(-2, 1, 2), 1, PassiveMaterial(PURPLE, 0.0, 1.0, PURPLE)),
]

ground = [
    Sphere(Pos(-10, -10, 0), 10, Material(WHITE, 1.0, BLACK, 0.0, 0.0)),  # Sun
    Sphere(Pos(0, 100, 5), 100, PassiveMaterial(LIGHT_GREEN, 0.0, 1.0)),  # Ground
    Sphere(Pos(-0.6, 0, 1.5), 0.3, PassiveMaterial(RED, 0.8, 0.2, RED)),
    Sphere(Pos(0.6, 0, 2), 0.3, PassiveMaterial(YELLOW, 0.8, 0.5, YELLOW)),
    Plane(Pos(0, 0, 10), Dir(0, 0, -1), PassiveMaterial(WHITE, 0.0, 1.0)),
]

cornell_box_1 = [
    ## Ceiling
    Plane(Pos(0, -1.8, 0), Dir(0, 1, 0), Material(WHITE, 2.0, BLACK)),
    ## Passive Walls
    Plane(Pos(0, 0, 5), Dir(0, 0, -1), MatteMaterial(WHITE)),  # Back
    Plane(Pos(-2, 0, 0), Dir(1, 0, 0), MatteMaterial(RED)),  # Left
    Plane(Pos(2, 0, 0), Dir(-1, 0, 0), MatteMaterial(GREEN)),  # Right
    Plane(Pos(0, 1, 0), Dir(0, -1, 0), MatteMaterial(WHITE)),  # Floor
    Plane(Pos(0, 0, -0.2), Dir(0, 0, 1), MatteMaterial(BLACK)),  # Rear
    ## Matte Spheres (for RGB)
    Sphere(Pos(-1, 0.7, 3), 0.3, MatteMaterial(LIGHTER_YELLOW)),
    Sphere(Pos(0, 0.7, 3), 0.3, MatteMaterial(LIGHTER_PINK)),
    Sphere(Pos(1, 0.7, 3), 0.3, MatteMaterial(LIGHTER_GREEN)),
]

cornell_box_2 = [
    ## Ceiling
    Plane(Pos(0, -1.5, 0), Dir(0, 1, 0), Material(WHITE, 2.0, BLACK)),
    ## Passive Walls
    Plane(Pos(0, 0, 3.7), Dir(0, 0, -1), MatteMaterial(WHITE)),  # Back
    Plane(Pos(-1.5, 0, 0), Dir(1, 0, 0), MatteMaterial(RED)),  # Left
    Plane(Pos(1.5, 0, 0), Dir(-1, 0, 0), MatteMaterial(GREEN)),  # Right
    Plane(Pos(0, 1, 0), Dir(0, -1, 0), MatteMaterial(WHITE)),  # Floor
    Plane(Pos(0, 0, -0.2), Dir(0, 0, 1), MatteMaterial(BLACK)),  # Rear
    ## Shiny Spheres
    Sphere(Pos(-1.1, 0.6, 3), 0.4, PassiveMaterial(LIGHT_YELLOW, 0.1, 0.2, Color(0.9, 0.9, 0.9))),
    Sphere(Pos(0, 0.6, 3), 0.4, PassiveMaterial(PINK, 0.3, 0.2, Color(0.9, 0.9, 0.9))),
    Sphere(Pos(1.1, 0.6, 3), 0.4, PassiveMaterial(BLUE, 0.5, 0.5, RED)),
    ## Green Spheres
    Sphere(Pos(-1.25, -0.5, 3.5), 0.2, PassiveMaterial(GREEN, 1.0, 0, Color(0.3, 1.0, 0.3))),
    Sphere(Pos(-0.625, -0.5, 3.5), 0.2, PassiveMaterial(GREEN, 1.0, 0.25, Color(0.3, 1.0, 0.3))),
    Sphere(Pos(0, -0.5, 3.5), 0.2, PassiveMaterial(GREEN, 1.0, 0.5, Color(0.3, 1.0, 0.3))),
    Sphere(Pos(0.625, -0.5, 3.5), 0.2, PassiveMaterial(GREEN, 1.0, 0.75, Color(0.3, 1.0, 0.3))),
    Sphere(Pos(1.25, -0.5, 3.5), 0.2, PassiveMaterial(GREEN, 1.0, 1.0, Color(0.3, 1.0, 0.3))),
]
# fmt: on

OBJECTS = cornell_box_1


class Ray:
    MAX_BOUNCES = 4

    def __init__(self, origin, direction):
        self.orig = origin
        self.dir = normalize(direction)
        self.color = np.array([0.0, 0.0, 0.0])
        self.throughput = np.array([1.0, 1.0, 1.0])
        self._last_hit_obj = None

    def find_next_hit(self):
        min_distance = np.inf
        hit_obj = None
        for obj in OBJECTS:
            distance = obj.intersect(self)
            if distance is not None and 0 < distance < min_distance:
                min_distance = distance
                hit_obj = obj
        # Safeguard against double hit (only works for concave objects obviously)
        if self._last_hit_obj == id(hit_obj):
            raise RuntimeError("Same object hit twice!")
        self._last_hit_obj = id(hit_obj)
        return hit_obj, min_distance

    def update(self, hit_obj, distance):
        # Update origin
        hit_point = self.orig + distance * self.dir
        normal = hit_obj.get_normal(hit_point)  # FIXME: orientation may be reversed
        # Move point slightly along the normal to prevent double hit on the same object.
        self.orig = hit_point + 0.001 * normal

        mat = hit_obj.mat

        # Update color
        # When a ray hits an object, emissive * throughput is added to the pixel's color
        self.color += mat.emission_light * self.throughput

        # Test for specular reflection
        is_specular = random.random() < mat.specular_prob

        # Sample direction for diffuse reflection
        # Cosine weighted distribution
        diffuse_dir = normalize(normal + random_vector_sphere())

        # When a ray hits an object, the throughput is multiplied by the object's albedo
        # if it's a diffuse reflection or the specular color if it's a specular ray.
        if is_specular:
            specular_dir = reflect(self.dir, normal)
            # Square the roughness to make it perceptually linear
            self.dir = normalize(lerp(specular_dir, diffuse_dir, mat.roughness**2))
            self.throughput *= mat.specular_color
        else:
            self.dir = diffuse_dir
            self.throughput *= mat.albedo

    def shoot(self):
        for _ in range(Ray.MAX_BOUNCES):
            obj, distance = self.find_next_hit()
            if obj is None:
                break
            self.update(obj, distance)


def ACESFilm(x):
    # Tone mapping curve
    # https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)


def lin_to_sRGB(c):
    c = np.clip(c, 0, 1)
    c_hi = pow(c, 1.0 / 2.4) * 1.055 - 0.055
    c_lo = c * 12.92
    return np.where(c < 0.0031308, c_lo, c_hi)


def sRGB_to_linear(c):
    c = np.clip(c, 0, 1)
    c_hi = pow((c + 0.055) / 1.055, 2.4)
    c_lo = c / 12.92
    return np.where(c < 0.04045, c_lo, c_hi)


def render_frame(i):
    print(f"Rendering frame {i}")
    t_start = time.time()
    cam_width = 320
    cam_height = 240
    cam_position = np.array([0, 0, 0])
    fov = 60 * math.pi / 180
    focal = (cam_width / 2) / math.tan(fov / 2)

    u0 = (cam_width - 1) / 2
    v0 = (cam_height - 1) / 2

    new_im = np.zeros((cam_height, cam_width, 3))
    for v in range(cam_height):
        for u in range(cam_width):
            # Anti-aliasing: add a small jitter on the ray direction
            jitter = random_vector_circle()
            ray_dir = np.array([u - u0, v - v0, focal])

            ray = Ray(cam_position, ray_dir + jitter)
            ray.shoot()

            color = ray.color
            # Apply tone mapping: convert unbounded HDR color range to SDR color range
            # color = ACESFilm(color)
            # Convert from linear to sRGB for display
            # color = lin_to_sRGB(color)
            new_im[v, u] = color

    t_end = time.time()
    print(f"Frame {i} rendered in {t_end - t_start:.3f} s")
    return new_im


class Renderer:
    PARALLEL_JOBS = 4  # 4 physical cores (= num cpu / 2)

    def __init__(self):
        self.i = 0
        self.avg_im = None
        self.lock = Lock()  # For parallel execution
        self.save_folder = datetime.now().isoformat()
        os.mkdir(self.save_folder)

    def process_new_frame(self, frame):
        with self.lock:
            if self.avg_im is None:
                self.avg_im = frame
            else:
                # Weighted average
                weight = 1 / (self.i + 1)
                self.avg_im = self.avg_im * (1 - weight) + frame * weight
            self.i += 1
            if self.i % 4 == 0:
                self.save()

    def show(self):
        if self.avg_im is not None:
            cv2.imshow("render", self.avg_im)
            cv2.waitKey(1)

    def save(self, name=None):
        if self.avg_im is not None:
            im = np.clip(self.avg_im * 255, 0, 255).astype(np.uint8)
            if name is None:
                name = (
                    f"{self.save_folder}/{datetime.now().isoformat()}_pass_{self.i}.png"
                )
            cv2.imwrite(name, im)

    def render(self, n_frames):
        for i in range(n_frames):
            new_im = render_frame(i)
            self.process_new_frame(new_im)
            self.show()

        self.save()

    def render_par(self, n_frames):
        with Pool(Renderer.PARALLEL_JOBS) as p:
            res = [
                p.apply_async(render_frame, [i], callback=self.process_new_frame)
                for i in range(n_frames)
            ]
            [r.wait() for r in res]


if __name__ == "__main__":
    r = Renderer()
    r.render_par(100)
    r.show()
    print("Rendering complete")
    cv2.waitKey()
