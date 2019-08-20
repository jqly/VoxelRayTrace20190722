
#ifndef JIANGQILEI_CAMERA_H
#define JIANGQILEI_CAMERA_H

#include <vector>
#include <random>
#include "graphics_math.h"
#include <thread>
#include <future>
#include "thread_pool_cpp/thread_pool.hpp"

using jql::Vec2;
using jql::iVec2;
using jql::Vec3;
using jql::iVec3;
using jql::Vec4;
using jql::iVec4;
using jql::Mat4;
using jql::iMat4;
using jql::Mat3;
using jql::iMat3;
using jql::Ray;

class Film {
public:
        const float w, h;
        const int nx, ny;

        Film(float w, float h, int nx, int ny);

        Vec3 get(int x, int y) const;
        void set(int x, int y, const Vec3& color);
        void add(int x, int y, const Vec3& color);
        std::vector<std::uint8_t> to_byte_array() const;
        std::vector<float> to_float_array() const;

private:
        std::vector<Vec3> data_;
};

// fn(Film*, int px, int py);
template <typename FN>
void render_mt(Film* film, FN fn)
{
        const iVec2 nt{ 8, 8 };
        const iVec2 pt{ film->nx / nt.x, film->ny / nt.y };

        std::vector<std::promise<void>> waiters(nt.x * nt.y);
        tp::ThreadPool pool;

        for (int tx = 0; tx < nt.x; ++tx) {
                for (int ty = 0; ty < nt.y; ++ty) {
                        auto& waiter = waiters[tx + ty * nt.x];
                        const iVec2 p0 = pt * iVec2{ tx, ty };
                        const iVec2 p1 = p0 + pt;
                        pool.post([film, &waiter, &fn, p0, p1]() {
                                for (int py = p0.y; py < p1.y; ++py) {
                                        for (int px = p0.x; px < p1.x; ++px) {
                                                fn(film, px, py);
                                        }
                                }
                                waiter.set_value();
                        });
                }
        }
        for (auto& waiter : waiters)
                waiter.get_future().wait();
}

class Camera {
public:
        const float fov;
        const float near;
        const float far;

        Camera(float fov, Vec3 eye, Vec3 spot, Vec3 up, float near = 0,
               float far = std::numeric_limits<float>::max());
        std::vector<Ray> gen_rays1(const Film& film, int px, int py);
        std::vector<Ray> gen_rays4(const Film& film, int px, int py);

private:
        Mat4 C_;
};

#endif  // JIANGQILEI_CAMERA_H
