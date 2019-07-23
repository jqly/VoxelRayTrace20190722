#include "voxel_octree.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

        color = jql::clamp<float>(color, 0, 1);

        p[0] += color.x * 255;
        p[1] += color.y * 255;
        p[2] += color.z * 255;
}

Color read_pixel(std::uint8_t* img, int w, int x, int y)
{
        auto* p = &img[3 * (y * w + x)];
        return { p[0], p[1], p[2] };
}

jql::Vec3 trace(const vo::VoxelOctree& root, const jql::Ray& ray, int depth)
{
        auto voxel = vo::ray_march(root, ray);
        if (!voxel) {
                return { 0,0,0 };
        }
        if (voxel->type == vo::VoxelType::LightSource)
                return { 1,1,1 };
        jql::ISect isect{};
        voxel->aabb.isect(ray, &isect);
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

        auto voxels = vo::obj2voxel(
                "D:\\jiangqilei\\Documents\\Asset\\lionc\\lionc.obj", .1f);

        //auto voxels = vo::obj2voxel(
        //        "D:\\jiangqilei\\Documents\\Asset\\sphere\\sphere.obj", .1f);

        //for (auto& vox : voxels)
        //        jql::print("{},{},{}\n", vox.aabb.center().x,
        //                   vox.aabb.center().y, vox.aabb.center().z);

        auto root = vo::build_voxel_octree(voxels);

        int W = 1024;
        int H = 1024;
        float SW = 1.f;
        float SH = 1.f;
        std::vector<std::uint8_t> d(W * H * 3);
        jql::Vec3 rayo = root.aabb.center();
        rayo.x += 3.5f;
        jql::PCG pcg{ 0xc01dbeef };
        std::uniform_real_distribution<float> distr(0.2f, .8f);
        for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                        jql::Vec3 pos = root.aabb.center();
                        for (int s = 0; s < 4; ++s) {
                                pos.x += 3.f;
                                pos.y += ((x + distr(pcg)) / W - .5f) * SW;
                                pos.z +=
                                        ((H - (y + distr(pcg)) - 1) / H - .5f) *
                                        SH;
                                jql::Ray ray{ rayo, pos - rayo };
                                auto color = trace(root, ray, 5);
                                addi_write_pixel(d.data(), W, x, y,
                                                 color*(1.f/4.f));
                        }
                }
        }

        //stbi_flip_vertically_on_write(true);
        stbi_write_bmp("./test.bmp", W, H, 3, d.data());

        return 0;
}
