
#ifndef JIANGQILEI_CAMERA_H
#define JIANGQILEI_CAMERA_H

#include <vector>
#include <random>
#include "graphics_math.h"

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
