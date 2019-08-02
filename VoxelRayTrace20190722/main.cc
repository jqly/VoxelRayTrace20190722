#include "voxel_octree.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include <future>
#include "thread_pool_cpp/thread_pool.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static float Res = .01f;

jql::Vec3 trace(const vo::VoxelOctree& root, const jql::Ray& ray, int depth)
{
        auto voxel = vo::ray_march(root, ray);
        if (!voxel) {
                float t = 0.5 * (ray.d.y + 1.0);
                return jql::lerp(jql::Vec3{ 1.0f, 1.0f, 1.0f },
                                 jql::Vec3{ 0.6f, 0.8f, 1.0f }, t);
        }

        if (voxel->type == vo::VoxelType::LightProbe) {

                jql::Ray probe_ray{ voxel->aabb.center(), voxel->normal,
                                    jql::length(voxel->aabb.size()) };

                auto voxel2 = vo::ray_march(root, probe_ray);
                if (!voxel2)
                        return {};
                jql::ISect isect{};
                voxel2->aabb.isect(probe_ray, &isect);
                auto litness = vo::compute_litness(root, isect, Res);
                return litness + voxel2->litness;
        }

        //return voxel->albedo;
        //if (voxel->type == vo::VoxelType::LightSource)
        //        return voxel->albedo;

        //return voxel->litness;

        jql::ISect isect{};
        voxel->aabb.isect(ray, &isect);
        auto litness = vo::compute_litness(root, isect, Res);
        //return litness;
        return voxel->litness + voxel->albedo * litness;

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
        auto voxels = vo::obj2voxel(
                "C:\\Users\\jiangqilei\\source\\repos\\VoxelRayTrace20190722\\Asset\\sponza\\sponza.obj",
                Res);
        //auto voxels = vo::obj2voxel(
        //        "D:\\jiangqilei\\Documents\\Asset\\lionc\\export\\lionc.obj",
        //        Res);

        jql::print("octree...\n");
        auto root = vo::build_voxel_octree(voxels);

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
                                                                auto voxel = vo::ray_march(
                                                                        root,
                                                                        ray);
                                                                if (!voxel)
                                                                        continue;
                                                                jql::ISect
                                                                        isect{};
                                                                voxel->aabb.isect(
                                                                        ray,
                                                                        &isect);
                                                                voxel->litness +=
                                                                        voxel->albedo *
                                                                        (jql::dot(
                                                                                voxel->normal,
                                                                                -ray.d));
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
        vo::voxel_filter(&root);

        jql::print("cone tracing...\n");
        Camera cam{ jql::to_radian(90),
                    { .1f, .1f, 0.f },
                    { .0f, .1f, 0 },
                    { 0, 1, 0 } };
        Film film(SW, SH, W, H);

        {
                const int partw = 4;
                const int parth = 4;
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
                                                             cam.gen_rays1(film,
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
        jql::print("{},{},{}=={}\n", W, H, d.size(), W * H * 3);
        stbi_write_hdr("./test.hdr", W, H, 3, d.data());

        return 0;
}
