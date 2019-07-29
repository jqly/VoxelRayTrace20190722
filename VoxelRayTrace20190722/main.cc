#include "voxel_octree.h"
#include <iostream>
#include "camera.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float Res = .01f;

jql::Vec3 trace(vo::VoxelOctree& root, const jql::Ray& ray, int depth)
{
        auto voxel = vo::ray_march(root, ray);
        if (!voxel) {
                float t = 0.5 * (ray.d.y + 1.0);
                return jql::lerp(jql::Vec3{ 1.0f, 1.0f, 1.0f },
                                 jql::Vec3{ 0.6f, 0.8f, 1.0f }, t);
        }
        //return voxel->albedo;
        //if (voxel->type == vo::VoxelType::LightSource)
        //        return voxel->albedo;

        if (voxel->sg.sharpness > 0)
                return voxel->sg.amplitude;
        return { 0, 0, 0 };

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
        int W = 256;
        int H = 256;
        float SW = 1.f;
        float SH = 1.f;
        float FoVy = jql::to_radian(60.f);

        auto voxels = vo::obj2voxel(
                "D:\\jiangqilei\\Documents\\Asset\\sponza\\sponza.obj",
                Res);
        //auto voxels = vo::obj2voxel(
        //        "D:\\jiangqilei\\Documents\\Asset\\lionc\\export\\lionc.obj",
        //        Res);

        jql::print("voxelizer...\n");
        auto root = vo::build_voxel_octree(voxels);

        jql::print("light map...\n");
        Film sfilm(1, 1, 2048, 2048);
        Camera scam{ jql::to_radian(60),
                     { 1,10,1},
                     { 0, 0, 0 },
                     { 0, 1, 0 } };
        for (int x = 0; x < sfilm.nx; ++x) {
                for (int y = 0; y < sfilm.ny; ++y) {
                        for (const auto& ray : scam.gen_rays4(sfilm, x, y)) {
                                auto voxel = vo::ray_march(root, ray);
                                if (!voxel)
                                        continue;
                                jql::ISect isect{};
                                voxel->aabb.isect(ray, &isect);
                                if (voxel->sg.sharpness > 0)
                                        voxel->sg = dot(
                                                voxel->sg,
                                                vo::SG{ isect.reflect(-ray.d),
                                                        5.f, voxel->albedo });
                                else
                                        voxel->sg =
                                                vo::SG{ isect.reflect(-ray.d),
                                                        5.f, voxel->albedo };
                        }
                }
        }
        jql::print("filtering...\n");
        vo::voxel_filter(&root);

        jql::print("cone tracing...\n");
        Camera cam{ FoVy, { .2f,.1f,.2f }, { .1f, .2f, 0 }, { 0, 1, 0 } };
        Film film(SW, SH, W, H);
        for (int x = 0; x < film.nx; ++x) {
                for (int y = 0; y < film.ny; ++y) {
                        for (const auto& ray : cam.gen_rays4(film, x, y)) {
                                auto c = trace(root, ray, 5);
                                film.add(x, y, c*.25f);
                        }
                }
        }

        auto d = film.to_byte_array();
        stbi_write_bmp("./test.bmp", W, H, 3, d.data());

        return 0;
}
