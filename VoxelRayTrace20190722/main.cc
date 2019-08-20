#include "voxel_octree.h"
#include <iostream>
#include "camera.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float Res = .01f;

jql::Vec3 trace(gi::VoxelOctree& root, const jql::Ray& ray, int depth,
                bool even_invisible = false)
{
        gi::VoxelOctree* leaf_ptr{};
        gi::VoxelBase* voxel_ptr{};
        ISect isect{};
        if (!gi::ray_march(&root, ray, &leaf_ptr, &voxel_ptr, &isect,
                           even_invisible)) {
                float t = 0.5 * (ray.d.y + 1.0);
                return jql::lerp(jql::Vec3{ 1.0f, 1.0f, 1.0f },
                                 jql::Vec3{ 0.6f, 0.8f, 1.0f }, t);
        }
        auto indirect_light = gi::cone_trace(root, isect, Res);
        //return litness;
        Vec3 direct_light = leaf_ptr->compute_illum(-ray.d);
        if (voxel_ptr->is_visible())
                return voxel_ptr->get_albedo(isect) *
                       (indirect_light + direct_light);
        else
                return voxel_ptr->get_albedo(isect);
}

int main()
{
        int W = 1024;
        int H = 1024;
        float SW = 1.f;
        float SH = 1.f;
        float FoVy = jql::to_radian(60.f);

        jql::print("voxelizer...\n");

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        auto voxels = gi::obj2voxel(
                "C:\\Users\\jiangqilei\\source\\repos\\VoxelRayTrace20190722\\Asset\\sponza\\sponza.obj",
                &attrib, &shapes, &materials);
        jql::print("#tris={}\n", voxels.size());
        jql::print("octree...\n");

        std::vector<gi::VoxelBase*> voxel_ptrs;
        for (auto& voxel : voxels)
                voxel_ptrs.push_back(&voxel);

        std::vector<gi::LightProbe> probes;
        jql::PCG pcg{ 0xc01dbeefdeadbead };
        for (int i = 0; i < 100; ++i) {
                Vec3 rp = jql::random_point_in_unit_sphere(pcg);
                probes.push_back({ Vec3{ .5f + rp.x, .4f + rp.y, rp.z }, .1f });
        }

        for (auto& voxel : probes)
                voxel_ptrs.push_back(&voxel);

        gi::VoxelOctree root;
        gi::ray_march_init(&root, voxel_ptrs, 6);

        auto res = root.aabb.size() / std::powf(2.f, 6.f);
        Res = *std::min_element(jql::begin(res), jql::end(res));

        Vec3 light_dir = jql::normalize(Vec3{ 1, 10, 1 });

        jql::print("light map...\n");
        Film sfilm(1, 1, 2048, 2048);
        Camera scam{
                jql::to_radian(60), { 1, 10, 1 }, { 0, 0, 0 }, { 0, 1, 0 }
        };

        render_mt(&sfilm, [&scam, &root](Film* film, int px, int py) {
                auto rays = scam.gen_rays4(*film, px, py);
                for (const auto& ray : rays) {
                        gi::VoxelOctree* leaf_ptr{};
                        jql::ISect isect{};
                        gi::VoxelBase* voxel_ptr{};
                        if (!gi::ray_march(&root, ray, &leaf_ptr, &voxel_ptr,
                                           &isect))
                                continue;
                        auto illum = voxel_ptr->get_diffuse(isect, ray,
                                                            Vec3{ 1, 1, 1 });
                        for (int i = 0; i < 6; ++i) {
                                float coeff = jql::dot(leaf_ptr->illum_d[i], isect.normal);
                                coeff = jql::clamp(coeff, 0.f, 1.f);
                                leaf_ptr->illum[i] += coeff * illum;
                        }
                }
        });

        jql::print("filtering...\n");
        gi::cone_trace_init_filter(&root);

        jql::print("light probing...\n");

        for (auto& probe : probes)
                probe.gather_light(&root, Res, light_dir, root.compute_illum(light_dir));

        std::vector<gi::LightProbe*> probe_ptrs;
        for (auto& p : probes)
                probe_ptrs.push_back(&p);

        jql::print("cone tracing...\n");
        Camera cam{ jql::to_radian(90),
                    { 1.f, 1.3f, -.2f },
                    { .0f, .4f, 0 },
                    { 0, 1, 0 } };
        Film film(SW, SH, W, H);

        render_mt(&film, [&cam, &root](Film* film, int px, int py) {
                for (const auto& ray : cam.gen_rays4(*film, px, py)) {
                        auto c = trace(root, ray, 5, true);
                        film->add(px, py, c * .25f);
                }
        });

        auto d = film.to_float_array();
        stbi_write_hdr("./test2.hdr", W, H, 3, d.data());
        jql::print("success.\n");
        return 0;
}
