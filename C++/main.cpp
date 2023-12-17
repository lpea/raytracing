#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <iostream>
#include <math.h>
#include <random>
#include <utility>

using Point = cv::Point3f;
using Vec = cv::Point3f;
// using Color = cv::Scalar;
using Color = cv::Vec3f;

Color::value_type max(const Color &c)
{
    return MAX(MAX(c[0], c[1]), c[2]);
}

class RandomGenerator
{
public:
    RandomGenerator() : rd(), gen(rd()), dis_uniform(), dis_normal()
    {
    }
    float getUniform()
    {
        // return a random value in [0.0, 1.0).
        return dis_uniform(gen);
    }
    float getNormal()
    {
        return dis_normal(gen);
    }

private:
    std::random_device rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen;
    std::uniform_real_distribution<float> dis_uniform;
    std::normal_distribution<float> dis_normal;
};

float getRandomValueUniform()
{
    static RandomGenerator gen;
    return gen.getUniform();
}

float getRandomValueNormal()
{
    static RandomGenerator gen;
    return gen.getNormal();
}

Vec randomUnitVectorOnCircle()
{
    const auto theta = getRandomValueUniform() * 2.0f * float(M_PI);
    return Vec(std::cos(theta), std::sin(theta), 0);
}

Vec randomUnitVectorOnSphere()
{
    // Random unit vector on the unit sphere
    // https://mathworld.wolfram.com/SpherePointPicking.html
    const auto x = getRandomValueNormal();
    const auto y = getRandomValueNormal();
    const auto z = getRandomValueNormal();
    const auto invNorm = 1.0f / std::hypot(x, y, z);
    return Vec(x * invNorm, y * invNorm, z * invNorm);
}

Vec normalize(const Vec &v)
{
    return v / cv::norm(v);
}

Vec reflect(const Vec &v, const Vec &normal)
{
    return v - 2 * v.dot(normal) * normal;
}

Vec lerp(const Vec &v0, const Vec &v1, float t)
{
    return v0 * (1 - t) + v1 * t;
}

// BGR
static const Color BLACK(0.0, 0.0, 0.0);
static const Color WHITE(1.0, 1.0, 1.0);

struct Material
{
    Material(const Color &emission_color,
             float emission_intensity,
             const Color &albedo,
             float specular_prob,
             float roughness,
             const Color &specular_color)
        : emission_color(emission_color),
          emission_intensity(emission_intensity),
          albedo(albedo),
          specular_prob(specular_prob),
          roughness(roughness),
          specular_color(specular_color)
    {
    }

    Color emission_color;
    float emission_intensity;
    Color albedo;
    float specular_prob;
    float roughness;
    Color specular_color;
};

Material makePassiveMaterial(const Color &albedo,
                             float specular_prob = 0.0,
                             float roughness = 1.0,
                             const Color &specular_color = WHITE)
{
    return Material(BLACK, 0.0, albedo, specular_prob, roughness, specular_color);
}

struct Ray
{
    Ray(const Point origin, const Vec direction) : orig(origin),
                                                   dir(direction / cv::norm(direction)),
                                                   color(0, 0, 0),
                                                   throughput(1.0, 1.0, 1.0)
    {
    }

    Point orig;
    Vec dir;
    Color color;
    Color throughput;
};

struct Hit
{
    bool found{false};
    float distance{std::numeric_limits<float>::infinity()};
    Point hit_point{};
    Vec normal{};
};

struct Object
{
    virtual Hit intersect(const Ray &) const = 0;
    virtual const Material &getMaterial() const = 0;
};

struct Sphere : Object
{
    Point pos;
    float radius;

    Material mat;

    Sphere(const Point &pos, float radius, const Material &material) : pos(pos), radius(radius), mat(material) {}

    Hit intersect(const Ray &ray) const override
    {
        const auto v = ray.orig - pos;
        const auto b = ray.dir.dot(v);
        const auto c = v.dot(v) - radius * radius;
        const auto delta = b * b - c;

        if (delta > 0)
        {
            const auto dist = -b - std::sqrt(delta);
            if (dist >= 0)
            {
                const auto hit_point = ray.orig + dist * ray.dir;
                // Todo handle case where the ray came from inside the sphere?
                return {true, dist, hit_point, (hit_point - pos) / radius};
            }
        }
        return {};
    }

    const Material &getMaterial() const override
    {
        return mat;
    }
};

struct Plane : Object
{
    Point pos;
    Vec normal;

    Material mat;

    Plane(const Point &position, const Vec &normal, const Material &material)
        : pos(position), normal(normal), mat(material)
    {
    }

    Hit intersect(const Ray &ray) const override
    {
        const auto v = pos - ray.orig;
        const auto p = ray.dir.dot(normal);
        if (p != 0.0f)
        {
            const auto dist = v.dot(normal) / p;
            if (dist >= 0.0f)
            {
                const auto hit_point = ray.orig + dist * ray.dir;
                return {true, dist, hit_point, p < 0.0f ? normal : -normal};
            }
        }
        return {};
    }

    const Material &getMaterial() const override
    {
        return mat;
    }
};

struct Triangle : Object
{
    Point a;
    Point b;
    Point c;

    Material mat;

private:
    Vec ab;
    Vec ac;
    Vec normal;

public:
    Triangle(Point a, Point b, Point c, Material mat)
        : a(a), b(b), c(c), mat(mat), ab(b - a), ac(c - a), normal(ab.cross(ac))
    {
    }

    // https://stackoverflow.com/a/42752998
    Hit intersect(const Ray &ray) const override
    {
        const auto det = -ray.dir.dot(normal);
        const auto invdet = 1.0f / det;
        const auto AO = ray.orig - a;
        const auto DAO = AO.cross(ray.dir);
        const auto u = ac.dot(DAO) * invdet;
        const auto v = -ab.dot(DAO) * invdet;
        const auto t = AO.dot(normal) * invdet;
        // std::abs(det) to detect intersection no matter the normal orientation.
        if (std::abs(det) > 0.0f and t > 0.0f and u >= 0.0f and v >= 0.0f and (u + v) <= 1.0f)
        {
            return {true, t, ray.orig + t * ray.dir, normalize(det < 0.0f ? -normal : normal)};
        }
        return {};
    }

    const Material &getMaterial() const override
    {
        return mat;
    }
};

// Made of two triangles
// TODO: what about a plain rectangle implementation? Can it be more efficient?
struct Quad : Object
{
    Material mat;

private:
    Triangle abc;
    Triangle cda;

public:
    Quad(Point a, Point b, Point c, Point d, Material mat)
        : mat(mat), abc(a, b, c, mat), cda(c, d, a, mat)
    {
        // TODO: check all points are on a plane?
        // Maybe check that abc.getNormal().dot(cda.getNormal()) == 1.0?
    }

    Hit intersect(const Ray &ray) const override
    {
        const auto hit = abc.intersect(ray);
        if (hit.found and hit.distance >= 0.0f)
        {
            return hit;
        }
        return cda.intersect(ray);
    }

    const Material &getMaterial() const override
    {
        return mat;
    }
};

using ObjectPtr = std::shared_ptr<Object>;
using Scene = std::vector<ObjectPtr>;

static const Color LIGHT_YELLOW0(0.7, 0.9, 1.0);
static const Color GRAY0(0.7, 0.7, 0.7);
static const Color RED0(0.1, 0.1, 0.7);
static const Color GREEN0(0.1, 0.7, 0.1);

Scene buildSceneCornellBox(bool quad_lighting = true)
{
    std::vector<ObjectPtr> scene;
    if (quad_lighting)
    {
        // Light (quad) + ceiling
        scene.push_back(std::make_shared<Quad>(
            Point(-0.5, -1.24, 2.25),
            Point(0.5, -1.24, 2.25),
            Point(0.5, -1.24, 1.75),
            Point(-0.5, -1.24, 1.75),
            Material(LIGHT_YELLOW0, 20.0, BLACK, 0.0, 1.0, LIGHT_YELLOW0)));                                    // Light
        scene.push_back(std::make_shared<Plane>(Point(0, -1.25, 0), Vec(0, 1, 0), makePassiveMaterial(GRAY0))); // Ceiling
    }
    else
    {
        // Light + ceiling (cheaper alternative)
        scene.push_back(std::make_shared<Plane>(Point(0, -1.25, 0), Vec(0, 1, 0), Material(LIGHT_YELLOW0, 2.0, BLACK, 0.0, 1.0, LIGHT_YELLOW0)));
    }

    // Walls
    scene.push_back(std::make_shared<Plane>(Point(0, 0, 2.5), Vec(0, 0, -1), makePassiveMaterial(GRAY0)));   // Back
    scene.push_back(std::make_shared<Plane>(Point(-1.25, 0, 0), Vec(1, 0, 0), makePassiveMaterial(RED0)));   // Left
    scene.push_back(std::make_shared<Plane>(Point(1.25, 0, 0), Vec(-1, 0, 0), makePassiveMaterial(GREEN0))); // Right
    scene.push_back(std::make_shared<Plane>(Point(0, 1.25, 0), Vec(0, -1, 0), makePassiveMaterial(GRAY0)));  // Floor
    scene.push_back(std::make_shared<Plane>(Point(0, 0, -1.6), Vec(0, 0, 1), makePassiveMaterial(BLACK)));   // Rear

    return scene;
}

Scene buildSceneCornellBoxQuads()
{
    const Point left_top_near(-1.25, -1.25, 1.5);
    const Point left_top_far(-1.25, -1.25, 2.5);
    const Point left_bottom_near(-1.25, 1.25, 1.5);
    const Point left_bottom_far(-1.25, 1.25, 2.5);
    const Point right_top_near(1.25, -1.25, 1.5);
    const Point right_top_far(1.25, -1.25, 2.5);
    const Point right_bottom_near(1.25, 1.25, 1.5);
    const Point right_bottom_far(1.25, 1.25, 2.5);

    std::vector<ObjectPtr> scene;
    // Light
    scene.push_back(std::make_shared<Quad>(
        Point(-0.5, -1.24, 2.25),
        Point(0.5, -1.24, 2.25),
        Point(0.5, -1.24, 1.75),
        Point(-0.5, -1.24, 1.75),
        Material(LIGHT_YELLOW0, 20.0, BLACK, 0.0, 1.0, LIGHT_YELLOW0)));

    // Walls
    scene.push_back(std::make_shared<Quad>(
        left_top_near, left_top_far, right_top_far, right_top_near, makePassiveMaterial(GRAY0))); // Ceiling
    scene.push_back(std::make_shared<Quad>(
        left_top_far, right_top_far, right_bottom_far, left_bottom_far, makePassiveMaterial(GRAY0))); // Back
    scene.push_back(std::make_shared<Quad>(
        left_top_near, left_top_far, left_bottom_far, left_bottom_near, makePassiveMaterial(RED0))); // Left
    // scene.push_back(std::make_shared<Triangle>(
    //     left_top_near, left_top_far, left_bottom_far, makePassiveMaterial(RED0))); // Left (up)
    // scene.push_back(std::make_shared<Triangle>(
    //     left_bottom_far, left_bottom_near, left_top_near, makePassiveMaterial(RED0))); // Left (down)
    scene.push_back(std::make_shared<Quad>(
        right_top_near, right_top_far, right_bottom_far, right_bottom_near, makePassiveMaterial(GREEN0))); // Right
    scene.push_back(std::make_shared<Quad>(
        left_bottom_near, left_bottom_far, right_bottom_far, right_bottom_near, makePassiveMaterial(GRAY0))); // Floor

    // scene.push_back(std::make_shared<Quad>(
    //     Point(-1.25, -1.25, -1.6),
    //     Point(1.25, -1.25, -1.6),
    //     Point(1.25, 1.25, -1.6),
    //     Point(-1.25, 1.25, -1.6),
    //     makePassiveMaterial(GRAY0))); // Rear

    return scene;
}

Scene buildSceneCornellBox1()
{
    auto scene = buildSceneCornellBox();

    // Matte Spheres
    // static const Color YELLOW1(0.75, 0.9, 0.9);
    // static const Color PINK1(0.9, 0.75, 0.9);
    // static const Color LIGHT_BLUE1(0.9, 0.9, 0.75);
    // scene.push_back(std::make_shared<Sphere>(Point(-0.9, 0.95, 2.0), 0.3, makePassiveMaterial(YELLOW1)));
    // scene.push_back(std::make_shared<Sphere>(Point(0, 0.95, 2.0), 0.3, makePassiveMaterial(PINK1)));
    // scene.push_back(std::make_shared<Sphere>(Point(0.9, 0.95, 2.0), 0.3, makePassiveMaterial(LIGHT_BLUE1)));

    // with an albedo boost to compensate for sRGB conversion
    static const Color YELLOW1b(0.5, 0.9, 0.9);
    static const Color PINK1b(0.9, 0.5, 0.9);
    static const Color LIGHT_BLUE1b(0.9, 0.9, 0.5);
    scene.push_back(std::make_shared<Sphere>(Point(-0.9, 0.95, 2.0), 0.3, makePassiveMaterial(YELLOW1b)));
    scene.push_back(std::make_shared<Sphere>(Point(0, 0.95, 2.0), 0.3, makePassiveMaterial(PINK1b)));
    scene.push_back(std::make_shared<Sphere>(Point(0.9, 0.95, 2.0), 0.3, makePassiveMaterial(LIGHT_BLUE1b)));

    return scene;
}

static const Color YELLOW2(0.5, 0.9, 0.9);
static const Color LIGHT_GRAY2(0.9, 0.9, 0.9);
static const Color PINK2(0.9, 0.5, 0.9);
static const Color PURE_BLUE(1.0, 0.0, 0.0);
static const Color PURE_RED(0.0, 0.0, 1.0);

static const Color GREEN2(0.3, 1.0, 0.3);

Scene buildSceneCornellBox2()
{
    auto scene = buildSceneCornellBox();

    // Shiny Spheres
    scene.push_back(std::make_shared<Sphere>(Point(-0.9, 0.95, 2.0), 0.3, makePassiveMaterial(YELLOW2, 0.1, 0.2, LIGHT_GRAY2)));
    scene.push_back(std::make_shared<Sphere>(Point(0, 0.95, 2.0), 0.3, makePassiveMaterial(PINK2, 0.3, 0.2, LIGHT_GRAY2)));
    scene.push_back(std::make_shared<Sphere>(Point(0.9, 0.95, 2.0), 0.3, makePassiveMaterial(PURE_BLUE, 0.5, 0.4, PURE_RED)));

    // Green Spheres
    scene.push_back(std::make_shared<Sphere>(Point(-1.0, 0.0, 2.3), 0.175, makePassiveMaterial(WHITE, 1.0, 0, GREEN2)));
    scene.push_back(std::make_shared<Sphere>(Point(-0.5, 0.0, 2.3), 0.175, makePassiveMaterial(WHITE, 1.0, 0.25, GREEN2)));
    scene.push_back(std::make_shared<Sphere>(Point(0.0, 0.0, 2.3), 0.175, makePassiveMaterial(WHITE, 1.0, 0.5, GREEN2)));
    scene.push_back(std::make_shared<Sphere>(Point(0.5, 0.0, 2.3), 0.175, makePassiveMaterial(WHITE, 1.0, 0.75, GREEN2)));
    scene.push_back(std::make_shared<Sphere>(Point(1.0, 0.0, 2.3), 0.175, makePassiveMaterial(WHITE, 1.0, 1.0, GREEN2)));

    return scene;
}

Scene buildSceneSpecularSpheres()
{
    auto scene = buildSceneCornellBox(false);

    // Make a 5x5 grid of white spheres with green specular reflections with varying values
    // of specular reflection probability and roughness.
    auto prob = 0.0;
    auto y = -1.0;
    auto z = 2.3;
    for (auto row = 0; row < 5; ++row, y += 0.5, z -= 0.2, prob += 0.25)
    {
        auto roughness = 0.0;
        auto x = -1.0;
        for (auto col = 0; col < 5; ++col, x += 0.5, roughness += 0.25)
        {
            scene.push_back(std::make_shared<Sphere>(Point(x, y, z), 0.175, makePassiveMaterial(WHITE, prob, roughness, GREEN2)));
        }
    }

    return scene;
}

Color shootRayAtScene(Ray ray, const Scene &scene, int max_bounces)
{
    for (auto i = 0; i < max_bounces; ++i)
    {
        Hit best_hit{};
        ObjectPtr hit_obj;
        for (const auto &obj : scene)
        {
            const auto hit = obj->intersect(ray);

            if (hit.found and 0.0f < hit.distance and hit.distance < best_hit.distance)
            {
                best_hit = hit;
                hit_obj = obj;
            }
        }
        if (hit_obj)
        {
            const auto &normal = best_hit.normal;
            ray.orig = best_hit.hit_point + 0.001f * normal;

            const auto &mat = hit_obj->getMaterial();
            ray.color += (mat.emission_color * mat.emission_intensity).mul(ray.throughput);

            const auto diffuse_dir = normalize(normal + randomUnitVectorOnSphere());

            if (getRandomValueUniform() < mat.specular_prob) // specular reflection
            {
                const auto specular_dir = reflect(ray.dir, normal);
                ray.dir = lerp(specular_dir, diffuse_dir, mat.roughness * mat.roughness);
                ray.throughput = ray.throughput.mul(mat.specular_color);
            }
            else // diffuse reflection
            {
                ray.dir = diffuse_dir;
                ray.throughput = ray.throughput.mul(mat.albedo);
            }

            // PERFORMANCE
            if (max(ray.throughput) == 0.0f) // Stop bouncing around if the throughput is zero
            {
                break;
            }
            // TODO: break early if throughput < threshold?
        }
    }

    return ray.color;
}

template <typename T>
T clip(T x, T low = 0.0, T high = 1.0)
{
    return MAX(MIN(x, high), low);
}

Color ACESFilm(const Color &color)
{
    // Tone mapping curve
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    static const auto a = 2.51f;
    static const auto b = 0.03f;
    static const auto c = 2.43f;
    static const auto d = 0.59f;
    static const auto e = 0.14f;
    const auto lambda = [&](auto x)
    {
        x = (x * (a * x + b)) / (x * (c * x + d) + e);
        return clip(x);
    };
    return Color(lambda(color[0]), lambda(color[1]), lambda(color[2]));
}

Color linearTosRGB(const Color &color)
{
    const auto lambda = [&](auto x)
    {
        x = clip(x);
        return x < 0.0031308f ? x * 12.92f : std::pow(x, 1.0f / 2.4f) * 1.055f - 0.055f;
    };
    return Color(lambda(color[0]), lambda(color[1]), lambda(color[2]));
}

Color sRGBToLinear(const Color &color)
{
    const auto lambda = [&](auto x)
    {
        x = clip(x);
        return x < 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f);
    };
    return Color(lambda(color[0]), lambda(color[1]), lambda(color[2]));
}

// Input image uses floating point representation, pixel value get multiplied
// by 255 and converted to uint8.
// Optionally apply post processing: exposure scaling, tone-mapping and sRGB conversion.
cv::Mat convertForDisplay(const cv::Mat &image_fp, bool apply_post_processing)
{
    static cv::Mat im_disp(image_fp.size(), CV_8UC3);

    if (apply_post_processing)
    {
        for (auto v = 0; v < image_fp.rows; ++v)
        {
            for (auto u = 0; u < image_fp.cols; ++u)
            {
                auto color = image_fp.at<Color>(v, u);
                // Apply exposure
                color *= 0.5f;
                // Apply tone mapping: convert unbounded HDR color range to SDR color range
                // Unbound floating point color value is mapped to [0.0, 1.0]
                color = ACESFilm(color);
                // Convert from linear RGB to sRGB
                color = linearTosRGB(color);
                // Rescale final image for display [0.0, 1.0] → [0, 255]
                im_disp.at<cv::Vec3b>(v, u) = color * 255;
            }
        }
    }
    else
    {
        image_fp.convertTo(im_disp, im_disp.type(), 255.0f);
    }

    return im_disp;
}

cv::Mat renderFrame(const Scene &scene)
{
    // Algorithm parameters
    static const auto nb_frames = 5000;
    static const auto parallel_execution = true;
    static const auto max_bounces = 8;
    static const auto apply_post_processing = true;

    const auto t0 = std::chrono::steady_clock::now();
    // Camera description
    const auto width = 320;
    const auto height = 240;
    const auto origin = Point(0, 0, -1.5);
    const float fov = 60.0f * M_PI / 180.0f;
    const auto focal = (width / 2.0f) / std::tan(fov / 2.0f);

    const auto u0 = float(width - 1) / 2.0f;
    const auto v0 = float(height - 1) / 2.0f;

    cv::Mat avg_im = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat new_im = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat im_disp;
    for (auto i = 0; i < nb_frames; ++i)
    {
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "Rendering frame " << i << std::endl;

        const auto update_pixel_value = [&](int u, int v)
        {
            Vec dir(u - u0, v - v0, focal);

            // Antialiasing: add jitter on ray direction
            const auto jitter = randomUnitVectorOnCircle() / 2.0f;
            dir += jitter;

            const Ray ray(origin, dir + jitter);
            new_im.at<Color>(v, u) = shootRayAtScene(ray, scene, max_bounces);
        };

        if (parallel_execution)
        {
            // Split work in parallel execution
            cv::parallel_for_(
                cv::Range(0, height * width), [&](const cv::Range &range)
                {
                    for (auto r = range.start; r < range.end; r++)
                    {
                        const auto v = r / width;
                        const auto u = r % width;
                        update_pixel_value(u, v);
                    } });
        }
        else
        {
            for (auto v = 0; v < height; ++v)
            {
                for (auto u = 0; u < width; ++u)
                {
                    update_pixel_value(u, v);
                }
            }
        }

        // Average with previous images
        const auto weight = 1.0f / (i + 1);
        avg_im = avg_im * (1 - weight) + new_im * weight;

        const auto t2 = std::chrono::steady_clock::now();
        const auto overall_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() / 1000.0;
        const auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;
        std::cout << "[" << overall_time << "] "
                  << "Frame " << i << " rendered in " << frame_time << " s" << std::endl;

        // Post-process and conversion to 3x8 bits
        im_disp = convertForDisplay(avg_im, apply_post_processing);

        const auto t3 = std::chrono::steady_clock::now();
        const auto post_proc_time = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() / 1000.0;
        std::cout << "Post-processing: " << post_proc_time << " s" << std::endl;

        if (i % 100 == 0)
        {
            cv::imwrite("frame_" + std::to_string(i) + ".png", im_disp);
        }
        cv::imshow("render", im_disp);
        cv::waitKey(1);
    }

    return im_disp;
}

int main()
{
    const auto scene = buildSceneCornellBox2();
    const auto final_image = renderFrame(scene);
    cv::imwrite("final_render.png", final_image);
    cv::imshow("final render", final_image);
    cv::waitKey();
    return 0;
}
