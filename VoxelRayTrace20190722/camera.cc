#include "camera.h"

Film::Film(float w, float h, int nx, int ny)
        : w{ w }
        , h{ h }
        , nx{ nx }
        , ny{ ny }
{
        std::vector<jql::Vec3>(nx * ny).swap(data_);
}

void Film::set(int x, int y, const Vec3& color)
{
        data_[y * ny + x] = color;
}

void Film::add(int x, int y, const Vec3& color)
{
        data_[y * ny + x] += color;
}

Vec3 Film::get(int x, int y) const
{
        return data_[y * ny + x];
}

std::vector<std::uint8_t> Film::to_byte_array() const
{
        std::vector<std::uint8_t> d(nx * ny * 3);
        for (int y = 0; y < ny; ++y) {
                for (int x = 0; x < nx; ++x) {
                        auto v = get(x, y) * 255.9f;
                        auto idx = (y * nx + x) * 3;
                        d[idx + 0] = static_cast<std::uint8_t>(v.x);
                        d[idx + 1] = static_cast<std::uint8_t>(v.y);
                        d[idx + 2] = static_cast<std::uint8_t>(v.z);
                }
        }
        return d;
}

Camera::Camera(float fov, Vec3 eye, Vec3 spot, Vec3 up, float near, float far)
        : fov{ fov }
        , near{ near }
        , far{ far }
{
        const Vec3 forward_ = normalize(spot - eye);
        const Vec3 s = normalize(cross(forward_, up));
        const Vec3 up_ = normalize(cross(s, forward_));

        C_ = jql::affine_transform(Mat3{ s, up_, -forward_ }, eye);
}

std::vector<Ray> Camera::gen_rays1(const Film& film, int px, int py)
{
        assert(px >= 0 && px < film.nx && py >= 0 && py < film.ny);
        const float x = px - film.nx / 2;
        const float y = (film.ny - 1 - py) - film.ny / 2;
        const float z = -(film.h / (2 * std::tanf(fov / 2)));
        std::vector<Ray> rays;
        std::vector<Vec2> samples{ Vec2{ 4, 4 } / 8.f };
        for (auto sample : samples) {
                auto x_ = (x + sample.x) / film.nx;
                auto y_ = (y + sample.y) / film.ny;
                Vec3 rayo = jql::point_transform(C_, {});
                Vec3 rayd = jql::vector_transform(C_, { x_, y_, z });
                rays.emplace_back(rayo, rayd, near, far);
        }
        return rays;
}

std::vector<Ray> Camera::gen_rays4(const Film& film, int px, int py)
{
        assert(px >= 0 && px < film.nx && py >= 0 && py < film.ny);
        const float x = px - film.nx / 2;
        const float y = (film.ny - 1 - py) - film.ny / 2;
        const float z = -(film.h / (2 * std::tanf(fov / 2)));
        std::vector<Ray> rays;
        std::vector<Vec2> samples{ Vec2{ 1, 5 } / 8.f, Vec2{ 3, 1 } / 8.f,
                                   Vec2{ 7, 3 } / 8.f, Vec2{ 5, 7 } / 8.f };
        for (auto sample : samples) {
                auto x_ = (x + sample.x) / film.nx;
                auto y_ = (y + sample.y) / film.ny;
                Vec3 rayo = jql::point_transform(C_, {});
                Vec3 rayd = jql::vector_transform(C_, { x_, y_, z });
                rays.emplace_back(rayo, rayd, near, far);
        }
        return rays;
}
