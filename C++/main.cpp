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

class RandomGenerator
{
public:
    RandomGenerator() : rd(), gen(rd()), dis()
    {
    }
    double get()
    {
        // return a random value in [0.0, 1.0).
        return dis(gen);
    }

private:
    std::random_device rd; // Will be used to obtain a seed for the random number engine
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;
};

float getRandomValue()
{
    static RandomGenerator gen;
    return gen.get();
}

Vec randomUnitVector()
{
    const auto theta = getRandomValue() * M_PI;
    const auto phi = getRandomValue() * 2 * M_PI;
    const float x = std::sin(theta) * std::cos(phi);
    const float y = std::sin(theta) * std::sin(phi);
    const float z = std::cos(theta);
    return {x, y, z};
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
static const Color RED(0.0, 0.0, 0.75);
static const Color PINK(0.75, 0.5, 0.75);
static const Color LIGHT_PINK(1.0, 0.75, 1.0);
static const Color LIGHTER_PINK(1.0, 0.875, 1.0);
static const Color GREEN(0.0, 0.75, 0.0);
static const Color LIGHT_GREEN(0.5, 0.75, 0.5);
static const Color LIGHTER_GREEN(0.75, 1.0, 0.75);
static const Color BLUE(0.75, 0.0, 0.0);
static const Color LIGHT_BLUE(0.75, 0.5, 0.5);
static const Color YELLOW(0.0, 0.75, 0.75);
static const Color LIGHT_YELLOW(0.5, 0.75, 0.75);
static const Color LIGHTER_YELLOW(0.75, 1.0, 1.0);
static const Color PURPLE(0.75, 0.0, 0.75);
static const Color ORANGE(0.0, 0.5, 0.75);

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

struct Object
{
    virtual double intersect(const Ray &) const = 0;
    virtual Vec getNormal(const Point &) const = 0;
    virtual const Material &getMaterial() const = 0;
};

struct Sphere : Object
{
    Point pos;
    float radius;
    Material mat;

    Sphere(const Point &pos, float radius, const Material &material) : pos(pos), radius(radius), mat(material) {}

    double intersect(const Ray &ray) const override
    {
        const auto v = ray.orig - pos;
        const auto d = ray.dir;
        const auto r2 = radius * radius;

        const auto b = d.dot(v);
        const auto c = v.dot(v) - r2;
        const auto delta = b * b - c;
        if (delta > 0)
        {
            return -b - std::sqrt(delta);
        }
        else
        {
            return std::numeric_limits<double>::infinity();
        }
    }

    Vec getNormal(const Point &pt) const override
    {
        return (pt - pos) / radius;
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

    double intersect(const Ray &ray) const override
    {
        const auto v = pos - ray.orig;
        const auto p = ray.dir.dot(normal);
        if (p != 0)
        {
            return v.dot(normal) / p;
        }
        else
        {
            return std::numeric_limits<double>::infinity();
        }
    }

    Vec getNormal(const Point &) const override
    {
        return normal;
    }

    const Material &getMaterial() const override
    {
        return mat;
    }
};

using ObjectPtr = std::shared_ptr<Object>;
using Scene = std::vector<ObjectPtr>;

Scene buildSceneCornellBox1()
{
    std::vector<ObjectPtr> scene;
    // Ceiling
    scene.push_back(std::make_shared<Plane>(Point(0, -1.8, 0), Vec(0, 1, 0), Material(WHITE, 1.0, WHITE, 0.0, 1.0, WHITE)));
    // Walls
    scene.push_back(std::make_shared<Plane>(Point(0, 0, 5), Vec(0, 0, -1), makePassiveMaterial(WHITE)));
    scene.push_back(std::make_shared<Plane>(Point(-2, 0, 0), Vec(1, 0, 0), makePassiveMaterial(RED)));
    scene.push_back(std::make_shared<Plane>(Point(2, 0, 0), Vec(-1, 0, 0), makePassiveMaterial(GREEN)));
    scene.push_back(std::make_shared<Plane>(Point(0, 1, 0), Vec(0, -1, 0), makePassiveMaterial(WHITE)));
    scene.push_back(std::make_shared<Plane>(Point(0, 0, -0.2), Vec(0, 0, 1), makePassiveMaterial(WHITE)));
    // Spheres
    scene.push_back(std::make_shared<Sphere>(Point(-1, 0.7, 3), 0.3, makePassiveMaterial(LIGHTER_YELLOW)));
    scene.push_back(std::make_shared<Sphere>(Point(0, 0.7, 3), 0.3, makePassiveMaterial(LIGHTER_PINK)));
    scene.push_back(std::make_shared<Sphere>(Point(1, 0.7, 3), 0.3, makePassiveMaterial(LIGHTER_GREEN)));

    return scene;
}

Scene buildSceneCornellBox2()
{
    std::vector<ObjectPtr> scene;
    // Ceiling
    scene.push_back(std::make_shared<Plane>(Point(0, -1.5, 0), Vec(0, 1, 0), Material(WHITE, 1.0, WHITE, 0.0, 1.0, WHITE)));
    // Walls
    scene.push_back(std::make_shared<Plane>(Point(0, 0, 3.7), Vec(0, 0, -1), makePassiveMaterial(WHITE))); // Front
    scene.push_back(std::make_shared<Plane>(Point(-1.5, 0, 0), Vec(1, 0, 0), makePassiveMaterial(RED)));   // Left
    scene.push_back(std::make_shared<Plane>(Point(1.5, 0, 0), Vec(-1, 0, 0), makePassiveMaterial(GREEN))); // Right
    scene.push_back(std::make_shared<Plane>(Point(0, 1, 0), Vec(0, -1, 0), makePassiveMaterial(WHITE)));   // Floor
    scene.push_back(std::make_shared<Plane>(Point(0, 0, -0.2), Vec(0, 0, 1), makePassiveMaterial(WHITE))); // Rear
    // Shiny Spheres
    scene.push_back(std::make_shared<Sphere>(Point(-1.1, 0.6, 3), 0.4, makePassiveMaterial(LIGHT_YELLOW, 0.1, 0.2, Color(0.9, 0.9, 0.9))));
    scene.push_back(std::make_shared<Sphere>(Point(0, 0.6, 3), 0.4, makePassiveMaterial(PINK, 0.3, 0.2, Color(0.9, 0.9, 0.9))));
    scene.push_back(std::make_shared<Sphere>(Point(1.1, 0.6, 3), 0.4, makePassiveMaterial(BLUE, 0.5, 0.5, RED)));
    // Green Spheres
    scene.push_back(std::make_shared<Sphere>(Point(-1.25, -0.5, 3.5), 0.2, makePassiveMaterial(GREEN, 1.0, 0, Color(0.3, 1.0, 0.3))));
    scene.push_back(std::make_shared<Sphere>(Point(-0.625, -0.5, 3.5), 0.2, makePassiveMaterial(GREEN, 1.0, 0.25, Color(0.3, 1.0, 0.3))));
    scene.push_back(std::make_shared<Sphere>(Point(0, -0.5, 3.5), 0.2, makePassiveMaterial(GREEN, 1.0, 0.5, Color(0.3, 1.0, 0.3))));
    scene.push_back(std::make_shared<Sphere>(Point(0.625, -0.5, 3.5), 0.2, makePassiveMaterial(GREEN, 1.0, 0.75, Color(0.3, 1.0, 0.3))));
    scene.push_back(std::make_shared<Sphere>(Point(1.25, -0.5, 3.5), 0.2, makePassiveMaterial(GREEN, 1.0, 1.0, Color(0.3, 1.0, 0.3))));

    return scene;
}

Color shootRayAtScene(Ray ray, const Scene &scene)
{
    static const auto MAX_BOUNCES = 6;
    for (auto i = 0; i < MAX_BOUNCES; ++i)
    {
        auto min_distance = std::numeric_limits<float>::infinity();
        ObjectPtr hit_obj;
        for (const auto &obj : scene)
        {
            const auto distance = obj->intersect(ray);
            if (0 < distance and distance < min_distance)
            {
                min_distance = distance;
                hit_obj = obj;
            }
        }
        if (hit_obj)
        {
            const auto hit_point = ray.orig + min_distance * ray.dir;
            const auto normal = hit_obj->getNormal(hit_point);
            ray.orig = hit_point + 0.001 * normal;

            const auto &mat = hit_obj->getMaterial();
            ray.color += (mat.emission_color * mat.emission_intensity).mul(ray.throughput);

            const auto is_specular = getRandomValue() < mat.specular_prob;
            const auto diffuse_dir = normalize(normal + randomUnitVector());

            if (is_specular)
            {
                const auto specular_dir = reflect(ray.dir, normal);
                ray.dir = lerp(specular_dir, diffuse_dir, mat.roughness * mat.roughness);
                ray.throughput = ray.throughput.mul(mat.specular_color);
            }
            else
            {
                ray.dir = diffuse_dir;
                ray.throughput = ray.throughput.mul(mat.albedo);
            }
        }
    }

    return ray.color;
}

void convertAndSaveFrame(const std::string &name, const cv::Mat &im)
{
    cv::Mat im8b;
    im.convertTo(im8b, CV_8U, 255.0);
    cv::imwrite(name, im8b);
}

cv::Mat renderFrame(const Scene &scene)
{
    static const auto nb_frames = 1000;
    static const auto parallel_execution = true;

    const auto t0 = std::chrono::steady_clock::now();
    const auto width = 640;
    const auto height = 480;
    const auto origin = Point(0, 0, 0);
    const auto fov = 60 * M_PI / 180;
    const auto focal = (width / 2) / std::tan(fov / 2);

    const auto u0 = (width - 1) / 2;
    const auto v0 = (height - 1) / 2;

    cv::Mat avg_im(height, width, CV_32FC3, Color(0, 0, 0));
    cv::Mat new_im(height, width, CV_32FC3, Color(0, 0, 0));
    for (auto i = 0; i < nb_frames; ++i)
    {
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "Rendering frame " << i << std::endl;

        const auto update_pixel_value = [&](int u, int v)
        {
            const Vec dir(u - u0 + getRandomValue() - 0.5, v - v0 + getRandomValue() - 0.5, focal);
            Ray ray(origin, dir);
            auto color = shootRayAtScene(ray, scene);
            new_im.at<Color>(v, u) = color;
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

        const auto weight = 1.0 / (i + 1);
        avg_im = avg_im * (1 - weight) + new_im * weight;

        const auto t2 = std::chrono::steady_clock::now();
        const auto overall_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() / 1000.0;
        const auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0;

        std::cout << "[" << overall_time << "] "
                  << "Frame " << i << " rendered in " << frame_time << " s" << std::endl;
        if (i % 100 == 0)
        {
            convertAndSaveFrame("frame_" + std::to_string(i) + ".png", avg_im);
        }
        cv::imshow("render", avg_im);
        cv::waitKey(1);
    }

    return avg_im;
}

int main()
{
    const auto scene = buildSceneCornellBox2();
    const auto t1 = std::chrono::steady_clock::now();
    const auto im = renderFrame(scene);
    const auto t2 = std::chrono::steady_clock::now();
    convertAndSaveFrame("final_render.png", im);
    cv::imshow("final render", im);
    cv::waitKey();
    return 0;
}
