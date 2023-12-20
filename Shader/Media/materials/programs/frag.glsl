#version 460

// https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language
// reference: https://registry.khronos.org/OpenGL-Refpages/gl4/html/

// External "auto" parameters
// are either declared in the .material file, e.g.
// >>> param_named_auto vp_width viewport_width
// or bound programmatically, e.g.
// >>> auto def_params = fragment_program->getDefaultParameters();
// >>> def_params->setNamedAutoConstant("vp_width", Ogre::GpuProgramParameters::ACT_VIEWPORT_WIDTH);
uniform float time = 0; // used to initialize the random seed
uniform float vp_width = 640;
uniform float vp_height = 480;
uniform vec3 camera_pos = vec3(0.0);
//uniform mat4 wvp_matrix;

// Output
// gl_FragColor is removed after version 420
layout(location = 0) out vec4 FragColor;

// Scene to render
// 1: matte spheres
// 2: shiny spheres
// 3: specular spheres
const int scene = 1;
// Number of samples to calculate and average
const int num_passes = 200;
// Number of times a ray can bounce after a hit
const int max_bounces = 4;
// Apply color "post processing" (exposure adjustment, tone-mapping...)
const bool post_processing = true;
// Use quads instead of planes
const bool use_quads = false;


const float PI = 3.141592;
const float inf = 10000.0; // far, far away

struct Ray {
  vec3 orig;
  vec3 dir;
  vec3 color;
  vec3 throughput;
};

struct Material {
  vec3 emissive;
  vec3 albedo;
  // TODO implement specular reflections
  float specular_proba;
  float roughness;
  vec3 specular_color;
};

struct Hit {
  float t;
  vec3 point;
  vec3 normal;
  Material mat;
};

/////////////////////////////////////////////////////////////
// See https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/

// initialize a random number state based on frag coord and time
uint rngState = uint(uint(gl_FragCoord.x) * uint(1973) + uint(gl_FragCoord.y) * uint(9277) + uint(time*1000) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed)
{
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float RandomFloat01(inout uint state)
{
  return float(wang_hash(state)) / 4294967296.0;
}

vec3 RandomUnitVector(inout uint state)
{
  float z = RandomFloat01(state) * 2.0f - 1.0f;
  float a = RandomFloat01(state) * 2 * PI;
  float r = sqrt(1.0f - z * z);
  float x = r * cos(a);
  float y = r * sin(a);
  return vec3(x, y, z);
}
/////////////////////////////////////////////////////////////

vec2 randomUnitVectorOnCircle(inout uint state)
{
    float theta = RandomFloat01(state) * 2 * PI;
    return vec2(cos(theta), sin(theta));
}

bool intersectSphere(in Ray ray, in vec4 sphere, inout Hit hit)
{
  vec3 center = sphere.xyz;
  float radius = sphere.w;
  vec3 v = ray.orig - center;
  float b = dot(ray.dir, v);
  float c = dot(v, v) - radius * radius;
  float delta = b * b - c;

  if (delta > 0) {
    float t = -b - sqrt(delta);
    if (t >= 0 && t < hit.t) {
      hit.t = t;
      hit.point = ray.orig + t * ray.dir;
      hit.normal = (hit.point - center) / radius;
      return true;
    }
  }
  return false;
}

bool intersectPlane(in Ray ray, in vec3 pt, in vec3 normal, inout Hit hit)
{
  vec3 v = pt - ray.orig;
  float p = dot(ray.dir, normal);
  if (p != 0) {
    float t = dot(v, normal) / p;
    if (t >= 0 && t < hit.t) {
      hit.t = t;
      hit.point = ray.orig + t * ray.dir;
      hit.normal = (p < 0) ? normal : -normal;
      return true;
    }
  }
  return false;
}

bool intersectTriangle(in Ray ray, in vec3 a, in vec3 b, in vec3 c, inout Hit hit)
{
  vec3 ab = b - a;
  vec3 ac = c - a;
  vec3 normal = cross(ab, ac);

  float det = -dot(ray.dir, normal);
  float invdet = 1.0 / det;
  vec3 AO = ray.orig - a;
  vec3 DAO = cross(AO, ray.dir);
  float u = dot(ac, DAO) * invdet;
  float v = -dot(ab, DAO) * invdet;
  float t = dot(AO, normal) * invdet;
  if (abs(det) > 0 && t > 0 && t < hit.t &&
      u >= 0.0 && v >= 0.0 && (u + v) <= 1.0 ) {
      hit.t = t;
      hit.point = ray.orig + t * ray.dir;
      hit.normal = normalize(det < 0 ? -normal : normal);
      return true;
  }
  return false;
}

bool intersectQuad(in Ray ray, in vec3 a, in vec3 b, in vec3 c, in vec3 d, inout Hit hit)
{
  if (intersectTriangle(ray, a, b, c, hit)) {
    return true;
  } else {
    return intersectTriangle(ray, c, d, a, hit);
  }
}

Hit getHit(in Ray ray)
{
  Hit hit;
  hit.t = inf;

  // Origin is the bottom-left corner, X right, Y up, Z forward

  // Ceiling + lighting
  if (use_quads) {
    if (intersectPlane(ray, vec3(0, 1.25, 0), vec3(0, -1, 0), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.7, 0.7, 0.7);
      hit.mat.specular_proba = 0.0;
    }
    if (intersectQuad(ray,
                      vec3(-0.5, 1.24, 2.25),
                      vec3(0.5, 1.24, 2.25),
                      vec3(0.5, 1.24, 1.75),
                      vec3(-0.5, 1.24, 1.75),
                      hit)) {
      hit.mat.emissive = vec3(1.0, 0.9, 0.7) * 20;
      hit.mat.albedo = vec3(0.0, 0.0, 0.0);
      hit.mat.specular_proba = 0.0;
    }
  } else {
    if (intersectPlane(ray, vec3(0, 1.25, 0), vec3(0, -1, 0), hit)) {
      hit.mat.emissive = vec3(1.0, 0.9, 0.7) * 2;
      hit.mat.albedo = vec3(0.0, 0.0, 0.0);
      hit.mat.specular_proba = 0.0;
    }
  }
  // Back
  if (intersectPlane(ray, vec3(0, 0, 2.5), vec3(0, 0, -1), hit)) {
    hit.mat.emissive = vec3(0.0);
    hit.mat.albedo = vec3(0.7, 0.7, 0.7);
    hit.mat.specular_proba = 0.0;
  }
  // Left
  if (intersectPlane(ray, vec3(-1.25, 0, 0), vec3(1, 0, 0), hit)) {
    hit.mat.emissive = vec3(0.0);
    hit.mat.albedo = vec3(0.7, 0.1, 0.1);
    hit.mat.specular_proba = 0.0;
  }
  // Right
  if (intersectPlane(ray, vec3(1.25, 0, 0), vec3(-1, 0, 0), hit)) {
    hit.mat.emissive = vec3(0.0);
    hit.mat.albedo = vec3(0.1, 0.7, 0.1);
    hit.mat.specular_proba = 0.0;
  }
  // Floor
  if (intersectPlane(ray, vec3(0, -1.25, 0), vec3(0, 1, 0), hit)) {
    hit.mat.emissive = vec3(0.0);
    hit.mat.albedo = vec3(0.7, 0.7, 0.7);
    hit.mat.specular_proba = 0.0;
  }
  // Rear
  if (intersectPlane(ray, vec3(0, 0, -1.6), vec3(0, 0, 1), hit)) {
    hit.mat.emissive = vec3(0.0);
    hit.mat.albedo = vec3(0.0, 0.0, 0.0);
    hit.mat.specular_proba = 0.0;
  }

  if (scene == 1) {
    if (intersectSphere(ray, vec4(-0.9, -0.95 * sin(time), 2.0, 0.3), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.9, 0.9, 0.5); // yellow
      hit.mat.specular_proba = 0.0;
    }
    if (intersectSphere(ray, vec4(0.0, -0.95, 2.0, 0.3), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.9, 0.5, 0.9); // pink
      hit.mat.specular_proba = 0.0;
    }
    if (intersectSphere(ray, vec4(0.9, -0.95, 2.0, 0.3), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.5, 0.9, 0.9); // light blue
      hit.mat.specular_proba = 0.0;
    }
  } else if (scene == 2) {
    if (intersectSphere(ray, vec4(-0.9, -0.95 * sin(time), 2.0, 0.3), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.9, 0.9, 0.5); // yellow
      hit.mat.specular_proba = 0.1;
      hit.mat.roughness = 0.2;
      hit.mat.specular_color = vec3(0.9, 0.9, 0.9); // light gray
    }
    if (intersectSphere(ray, vec4(0.0, -0.95, 2.0, 0.3), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.9, 0.5, 0.9); // pink
      hit.mat.specular_proba = 0.3;
      hit.mat.roughness = 0.2;
      hit.mat.specular_color = vec3(0.9, 0.9, 0.9); // light gray
    }
    if (intersectSphere(ray, vec4(0.9, -0.95, 2.0, 0.3), hit)) {
      hit.mat.emissive = vec3(0.0);
      hit.mat.albedo = vec3(0.0, 0.0, 1.0); // pure blue
      hit.mat.specular_proba = 0.5;
      hit.mat.roughness = 0.4;
      hit.mat.specular_color = vec3(1.0, 0.0, 0.0); // pure red
    }

    float x = -1.0;
    float roughness = 0.0;
    for (int col = 0; col < 5; ++col, x += 0.5, roughness += 0.25) {
      if (intersectSphere(ray, vec4(x, 0.0, 2.3, 0.175), hit)) {
        hit.mat.emissive = vec3(0.0);
        hit.mat.albedo = vec3(1.0, 1.0, 1.0); // pure white
        hit.mat.specular_proba = 1.0;
        hit.mat.roughness = roughness;
        hit.mat.specular_color = vec3(0.3, 1.0, 0.3); // green
      }
    }
  } else if (scene == 3) {
    float prob = 0.0;
    float y = 1.0;
    float z = 2.3;
    for (int row = 0; row < 5; ++row, y -= 0.5, z -= 0.2, prob += 0.25) {
      float x = -1.0;
      float roughness = 0.0;
      for (int col = 0; col < 5; ++col, x += 0.5, roughness += 0.25) {
        if (intersectSphere(ray, vec4(x, y, z, 0.175), hit)) {
          hit.mat.emissive = vec3(0.0);
          hit.mat.albedo = vec3(1.0, 1.0, 1.0); // pure white
          hit.mat.specular_proba = prob;
          hit.mat.roughness = roughness;
          hit.mat.specular_color = vec3(0.3, 1.0, 0.3); // green
        }
      }
    }
  }

  return hit;
}

vec3 shootRayAtWorld(in Ray ray)
{
  for (int i = 0; i < max_bounces; ++i) {
    Hit hit = getHit(ray);
    if (hit.t < inf) {
      ray.color += hit.mat.emissive * ray.throughput;
      ray.orig = hit.point + 0.001 * hit.normal;

      vec3 diffuse_dir = normalize(RandomUnitVector(rngState) + hit.normal);

      // // Branchless version (not faster...)
      // vec3 specular_dir = reflect(ray.dir, hit.normal);
      // bool is_specular = RandomFloat01(rngState) < hit.mat.specular_proba;
      // ray.dir = mix(diffuse_dir, specular_dir, (1 - pow(hit.mat.roughness, 2)) * float(is_specular));
      // ray.throughput *= mix(hit.mat.albedo, hit.mat.specular_color, float(is_specular));

      if (hit.mat.specular_proba > 0 && RandomFloat01(rngState) < hit.mat.specular_proba)
      {
        vec3 specular_dir = reflect(ray.dir, hit.normal);
        ray.dir = mix(specular_dir, diffuse_dir, pow(hit.mat.roughness, 2));
        ray.throughput *= hit.mat.specular_color;
      } else {
        ray.dir = diffuse_dir;
        ray.throughput *= hit.mat.albedo;
      }

      if (max(max(ray.throughput.r, ray.throughput.g), ray.throughput.b) == 0) {
        break; // break if throughput is zero
      }
    } else {
      break; // break if no hit found
    }
  }
  return ray.color;
}

vec3 LessThan(vec3 f, float value)
{
  return vec3(
    (f.x < value) ? 1.0f : 0.0f,
    (f.y < value) ? 1.0f : 0.0f,
    (f.z < value) ? 1.0f : 0.0f);
}

vec3 LinearToSRGB(vec3 rgb)
{
  rgb = clamp(rgb, 0.0f, 1.0f);
  return mix(pow(rgb, vec3(1.0f / 2.4f)) * 1.055f - 0.055f, rgb * 12.92f, LessThan(rgb, 0.0031308f));
}

vec3 SRGBToLinear(vec3 rgb)
{
  rgb = clamp(rgb, 0.0f, 1.0f);
  return mix(pow(((rgb + 0.055f) / 1.055f), vec3(2.4f)), rgb / 12.92f, LessThan(rgb, 0.04045f));
}

// ACES tone mapping curve fit to go from HDR to LDR
//https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 ACESFilm(vec3 x)
{
  const float a = 2.51f;
  const float b = 0.03f;
  const float c = 2.43f;
  const float d = 0.59f;
  const float e = 0.14f;
  return clamp((x*(a*x + b)) / (x*(c*x + d) + e), 0.0f, 1.0f);
}

void main()
{
  const float fov = radians(60.0);
  float focal = vp_width / 2.0 / tan(fov / 2.0);

  float u0 = (vp_width - 1.0) / 2.0;
  float v0 = (vp_height - 1.0) / 2.0;

  float u = gl_FragCoord.x;
  float v = gl_FragCoord.y;

  Ray ray;
  ray.orig = camera_pos;
  ray.dir = normalize(vec3(u - u0, v - v0, focal));
  ray.color = vec3(0.0);
  ray.throughput = vec3(1.0);

  vec3 rgb_avg;
  for (int i = 0; i < num_passes; ++i) {
    // calculate subpixel camera jitter for anti aliasing
    vec2 jitter = randomUnitVectorOnCircle(rngState) / 2;
    vec3 dir = normalize(vec3(u - u0, v - v0, focal) + vec3(jitter, 0.0));
    ray.dir = dir;
    rgb_avg += shootRayAtWorld(ray);
  }

  vec3 rgb = rgb_avg / float(num_passes);

  // Post-processing pass
  if (post_processing) {
    rgb *= 0.5;              // exposure
    rgb = ACESFilm(rgb);     // tone mapping
    rgb = LinearToSRGB(rgb); // sRGB
  }

  // Set pixel color
  FragColor = vec4(rgb, 1.0);
}
