#include "voxel_octree.h"
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using Color = jql::Mat<unsigned char, 3, 1>;

bool Trace(const vo::VoxelOctree& root, const jql::Ray& ray,
           Color* color, int trace_level)
{
        if (trace_level == 0)
                return false;
        jql::Vec3 env{ 0, .5f, .5f };
        auto voxel = vo::ray_march(root, ray);
        if (!voxel) {
                *color = jql::cast<Color>(env * 255);
                return false;
        }
        *color = jql::cast<Color>(voxel->color*255);
        return true;
        jql::Vec3 fcolor{ 1, 1, 1 };
        jql::ISect isect{};
        if (!voxel->aabb.isect(ray, &isect)) {
                *color = jql::cast<Color>(env * 255);
                return false;
        }

        jql::Vec3 rdir = isect.reflect(-ray.d);
        jql::Ray rray{ isect.hit, rdir, 1e-3f, 1000.f };

        Color rcolor{};
        auto diffuse = jql::dot(isect.normal, -ray.d);
        if (diffuse <= 0) {
                *color = { 0, 0, 0 };
                return true;
        }
        if (Trace(root, rray, &rcolor, trace_level - 1)) {
                *color = jql::cast<Color>((rcolor / 256.f) *
                                          (jql::cast<jql::Vec3>(voxel->color) / 256.f) * diffuse *
                                          255.f);
                return true;
        }
        else {
                *color = jql::cast<Color>(
                        1.f * env *
                        (jql::cast<jql::Vec3>(voxel->color) / 256.f) *
                                          diffuse * 255.f);
                return true;
        }

        //*color = { 255, 255, 255 };
        return true;
}

int main()
{

        auto voxels = vo::obj2voxel(
                "D:\\jiangqilei\\Documents\\Asset\\color_lion\\color_lion.obj", .01f);

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
        rayo.x += 4.f;
        for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                        jql::Vec3 pos = root.aabb.center();
                        pos.x += 3.f;
                        pos.y += (1.f * x / W - .5f) * SW;
                        pos.z += (1.f * (H - y - 1) / H - .5f) * SH;
                        jql::Ray ray{ rayo, pos - rayo };
                        Color color{};
                        //if (x == 98 && y == 109)
                        //        jql::print("j\n");
                        Trace(root, ray, &color, 5);

                        auto idx = (y * W + x) * 3;
                        d[idx + 0] = static_cast<std::uint8_t>(color.x);
                        d[idx + 1] = static_cast<std::uint8_t>(color.y);
                        d[idx + 2] = static_cast<std::uint8_t>(color.z);
                }
        }

        //stbi_flip_vertically_on_write(true);
        stbi_write_bmp("./test.bmp", W, H, 3, d.data());

        return 0;
}
