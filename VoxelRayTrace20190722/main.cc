#include "voxel_octree.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include <future>
#include "thread_pool_cpp/thread_pool.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float Res = .01f;

//static std::vector<vo::LightProbe> probes;

jql::Vec3 trace(gi::VoxelOctree& root, const jql::Ray& ray, int depth)
{
        //// Drawback: probe disregard physical occlusions.
        //for (auto& probe : probes) {
        //        float t{};
        //        if (probe.isectt(ray, &t)) {
        //                auto dir = jql::normalize(ray.o + t * ray.d - probe.o);
        //                return probe.eval(dir);
        //        }
        //}

        gi::VoxelOctree* leaf_ptr{};
        gi::VoxelBase* voxel_ptr{};
        ISect isect{};
        if (!gi::ray_march(&root, ray, &leaf_ptr, &voxel_ptr, &isect)) {
                float t = 0.5 * (ray.d.y + 1.0);
                return jql::lerp(jql::Vec3{ 1.0f, 1.0f, 1.0f },
                                 jql::Vec3{ 0.6f, 0.8f, 1.0f }, t);
        }

        //return leaf_ptr->diffuse;

        //if (voxel->type == vo::VoxelType::LightProbe) {

        //        jql::Ray probe_ray{ voxel->aabb.center(), voxel->normal,
        //                            jql::length(voxel->aabb.size()) };

        //        auto voxel2 = vo::ray_march(root, probe_ray);
        //        if (!voxel2)
        //                return {};
        //        jql::ISect isect{};
        //        voxel2->aabb.isect(probe_ray, &isect);
        //        auto litness = vo::compute_litness(root, isect, Res);
        //        return litness + voxel2->litness;
        //}

        //return voxel->albedo;
        //if (voxel->type == vo::VoxelType::LightSource)
        //        return voxel->albedo;

        //return voxel_ptr->get_diffuse();
        ////return voxel_ptr->get_albedo(isect);
        ////return .5f*(1+isect.normal);
        auto litness = gi::cone_trace(root, isect, Res);
        //return litness;
        return voxel_ptr->get_albedo(isect) * (litness + leaf_ptr->diffuse);
        //return voxel_ptr->get_diffuse() +
        //       voxel_ptr->get_albedo(isect) * litness;
        /*
        jql::Ray sray;
        jql::Vec3 att;
        if (depth > 0 && voxel->scatter(ray, isect, &att, &sray)) {
                sray.tmin += jql::eps;
                return att * trace(root, sray, depth - 1);
        }
        else {
                return { 0, 0, 0 };
        }*/
}

void test_thread_pool()
{
        const unsigned NumWaiters = 1600;
        std::vector<std::promise<float>> waiters{ NumWaiters };
        tp::ThreadPool pool;
        auto begin_count = std::chrono::high_resolution_clock::now()
                                   .time_since_epoch()
                                   .count();
        for (int i = 0; i < waiters.size(); ++i)
                pool.post([&waiters, i = i]() {
                        float a = i;
                        auto v = jql::Vec3{ a, a, a };
                        auto q = jql::Quat{ (float)i, v };
                        auto m = jql::Quat2Mat3(q);
                        for (int i = 0; i < 100000; ++i) {
                                m = jql::dot(m, m);
                                m[0] = jql::normalize(m[0]);
                                m[1] = jql::normalize(m[1]);
                                m[2] = jql::normalize(m[2]);
                        }
                        waiters[i].set_value(jql::value_sum(m));
                });

        for (auto& waiter : waiters) {
                auto future = waiter.get_future();
                future.wait();
        }

        long long int end_count = std::chrono::high_resolution_clock::now()
                                          .time_since_epoch()
                                          .count();
        std::cout << (double)(end_count - begin_count) / (double)1000000
                  << " ms" << std::endl;
}

int main()
{
        //test_thread_pool();
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

        jql::print("octree...\n");

        std::vector<gi::VoxelBase*> voxel_ptrs;
        for (auto& voxel : voxels)
                voxel_ptrs.push_back(&voxel);

        gi::VoxelOctree root;
        gi::ray_march_init(&root, voxel_ptrs, 6);

        auto res = root.aabb.size() / std::powf(2.f, 6.f);
        Res = *std::min_element(jql::begin(res), jql::end(res));
        jql::print("Res={}\n", Res);

        Vec3 light_dir = jql::normalize(Vec3{ 1, 10, 1 });
        {
                jql::print("light map...\n");
                const int W = 2048;
                const int H = 2048;
                const int partw = 8;
                const int parth = 8;
                const int subimgw = W / partw;
                const int subimgh = H / parth;
                Film sfilm(1, 1, W, H);
                Camera scam{ jql::to_radian(60),
                             { 1, 10, 1 },
                             { 0, 0, 0 },
                             { 0, 1, 0 } };
                std::vector<std::promise<void>> waiters{ partw * parth };
                tp::ThreadPool pool;

                for (int ph = 0; ph < parth; ++ph) {
                        for (int pw = 0; pw < partw; ++pw) {
                                pool.post([&root, &scam, &waiters, pw, ph,
                                           partw, parth, &sfilm, subimgw,
                                           subimgh]() {
                                        for (int x = pw * subimgw;
                                             x < pw * subimgw + subimgw; ++x) {
                                                for (int y = ph * subimgh;
                                                     y < ph * subimgh + subimgh;
                                                     ++y) {
                                                        for (const auto& ray :
                                                             scam.gen_rays4(
                                                                     sfilm, x,
                                                                     y)) {
                                                                gi::VoxelOctree*
                                                                        leaf_ptr{};
                                                                jql::ISect
                                                                        isect{};
                                                                gi::VoxelBase*
                                                                        voxel_ptr{};
                                                                if (!gi::ray_march(
                                                                            &root,
                                                                            ray,
                                                                            &leaf_ptr,
                                                                            &voxel_ptr,
                                                                            &isect))
                                                                        continue;
                                                                leaf_ptr->diffuse +=
                                                                        voxel_ptr
                                                                                ->get_diffuse(
                                                                                        isect,
                                                                                        ray,
                                                                                        Vec3{ .5,
                                                                                              .5,
                                                                                              .5 });
                                                        }
                                                }
                                        }
                                        waiters[ph * partw + pw].set_value();
                                });
                        }
                }
                for (auto& waiter : waiters)
                        waiter.get_future().wait();
        }

        jql::print("filtering...\n");
        gi::cone_trace_init_filter(&root);

        //jql::print("light probing...\n");

        //for (int i = 0; i < 3; ++i) {
        //        probes.push_back({ Vec3{ .5f - .6f * i, .4f, -.15f }, .2f });
        //        probes.back().gather_light(root, Res, light_dir);
        //}

        jql::print("cone tracing...\n");
        Camera cam{ jql::to_radian(90),
                    { 1.f, 1.3f, -.2f },
                    { .0f, .4f, 0 },
                    { 0, 1, 0 } };
        Film film(SW, SH, W, H);

        {
                const int partw = 8;
                const int parth = 8;
                const int subimgw = W / partw;
                const int subimgh = H / parth;
                std::vector<std::promise<void>> waiters{ partw * parth };
                tp::ThreadPool pool;

                for (int ph = 0; ph < parth; ++ph) {
                        for (int pw = 0; pw < partw; ++pw) {
                                pool.post([&root, &cam, &waiters, pw, ph, partw,
                                           parth, &film, subimgw, subimgh]() {
                                        for (int x = pw * subimgw;
                                             x < pw * subimgw + subimgw; ++x) {
                                                for (int y = ph * subimgh;
                                                     y < ph * subimgh + subimgh;
                                                     ++y) {
                                                        for (const auto& ray :
                                                             cam.gen_rays4(film,
                                                                           x,
                                                                           y)) {
                                                                auto c = trace(
                                                                        root,
                                                                        ray, 5);
                                                                film.add(
                                                                        x, y,
                                                                        c * .25f);
                                                        }
                                                }
                                        }
                                        waiters[ph * partw + pw].set_value();
                                });
                        }
                }
                for (auto& waiter : waiters)
                        waiter.get_future().wait();
        }

        //for (int x = 0; x < film.nx; ++x) {
        //        for (int y = 0; y < film.ny; ++y) {
        //                for (const auto& ray : cam.gen_rays1(film, x, y)) {
        //                        auto c = trace(root, ray, 5);
        //                        film.add(x, y, c * .25f);
        //                }
        //        }
        //}

        auto d = film.to_float_array();
        stbi_write_hdr("./test2.hdr", W, H, 3, d.data());
        jql::print("success.\n");
        return 0;
}
