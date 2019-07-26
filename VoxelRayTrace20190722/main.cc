#include "voxel_octree.h"
#include <iostream>
#include "camera.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float Res = .01f;

using Color = jql::Mat<std::uint8_t, 3, 1>;
using FColor = jql::Vec3;

void addi_write_pixel(std::uint8_t* img, int w, int x, int y, Color color)
{
        auto* p = &img[3 * (y * w + x)];
        p[0] += color.x;
        p[1] += color.y;
        p[2] += color.z;
}

void addi_write_pixel(std::uint8_t* img, int w, int x, int y, FColor color)
{
        auto* p = &img[3 * (y * w + x)];

        color = jql::clamp<float>(color, 0, .99f);

        p[0] += color.x * 255;
        p[1] += color.y * 255;
        p[2] += color.z * 255;
}

Color read_pixel(std::uint8_t* img, int w, int x, int y)
{
        auto* p = &img[3 * (y * w + x)];
        return { p[0], p[1], p[2] };
}

jql::Vec3 trace(vo::VoxelOctree& root, const jql::Ray& ray, int depth)
{
        auto voxel = vo::ray_march(root, ray);
        if (!voxel) {
                float t = 0.5 * (ray.d.y + 1.0);
                return jql::lerp(jql::Vec3{ 1.0f, 1.0f, 1.0f },
                                 jql::Vec3{ 0.5f, 0.7f, 1.0f }, t);
        }
        //if (voxel->type == vo::VoxelType::LightSource)
        //        return voxel->albedo;

        //if (voxel->sg.sharpness > 0)
        //        return voxel->sg.amplitude;
        //return { 0, 0, 0 };

        jql::ISect isect{};
        voxel->aabb.isect(ray, &isect);
        auto litness = vo::compute_litness(root, isect, Res);
        return litness;

        jql::Ray sray;
        jql::Vec3 att;
        if (depth > 0 && voxel->scatter(ray, isect, &att, &sray)) {
                sray.tmin += jql::eps;
                return att * trace(root, sray, depth - 1);
        }
        else {
                return { 0, 0, 0 };
        }
}

int main()
{
        int W = 1024;
        int H = 1024;
        float SW = 1.f;
        float SH = 1.f;

        auto voxels = vo::obj2voxel(
                "D:\\jiangqilei\\Documents\\Asset\\lionc\\export\\lionc.obj", Res);
        auto root = vo::build_voxel_octree(voxels);

        Sampler sampler{};

        jql::Vec3 LightDir{ 1, 1, 1 };
        Film shadow_film(1, 1, 1024, 1024);
        Camera shadow_camera{ { 1.5f, 3.5f, 0 },
                              { 0, 0, 0 },
                              { 0, -1, 0 },
                              jql::to_radian(60.f) };
        auto shadow_samples =
                shadow_camera.GenerateSamples(shadow_film, sampler);

        for (auto& sample : shadow_samples) {
                auto& ray = sample.rays.front();
                auto voxel = vo::ray_march(root, ray);
                if (!voxel)
                        continue;
                jql::ISect isect{};
                voxel->aabb.isect(ray, &isect);
                voxel->sg =
                        vo::SG{ isect.reflect(-ray.d), 5.f, voxel->albedo };
        }

        vo::voxel_filter(&root);

        Film film(SW, SH, W, H);
        Camera camera{
                { 3.5f, 0, 0 }, { 0, 0, 0 }, { 0, -1, 0 }, jql::to_radian(60.f)
        };
        auto samples = camera.GenerateSamples(film, sampler);
        for (auto& sample : samples) {

                jql::Vec3 color{};
                for (const auto& ray : sample.rays) {
                        auto c = trace(root, ray, 5);
                        color += c;
                }
                color *= (1.f / sample.rays.size());
                film.at(sample.x, sample.y) = color;
        }

        auto d = film.to_byte_array();
        stbi_write_bmp("./test.bmp", W, H, 3, d.data());

        return 0;
}
