
#ifndef JIANGQILEI_CAMERA_H
#define JIANGQILEI_CAMERA_H

#include <vector>
#include <random>
#include "graphics_math.h"

class Film {
public:
        float w, h;
        int nx, ny;
        std::vector<jql::Vec3> data;

        Film(float w, float h, int nx, int ny)
                : w{ w }
                , h{ h }
                , nx{ nx }
                , ny{ ny }
        {
                std::vector<jql::Vec3>(nx * ny).swap(data);
        }

        jql::Vec3 at(int x, int y) const
        {
                return data[y * ny + x];
        }

        jql::Vec3& at(int x, int y)
        {
                return data[y * ny + x];
        }

        std::vector<std::uint8_t> to_byte_array()
        {
                std::vector<std::uint8_t> d(nx * ny * 3);
                for (int y = 0; y < ny; ++y) {
                        for (int x = 0; x < nx; ++x) {
                                auto v = at(x, y) * 255.9f;
                                auto idx = (y * nx + x) * 3;
                                d[idx + 0] = static_cast<std::uint8_t>(v.x);
                                d[idx + 1] = static_cast<std::uint8_t>(v.y);
                                d[idx + 2] = static_cast<std::uint8_t>(v.z);
                        }
                }
                return d;
        }
};

class FilmSample {
public:
        int x, y;
        std::vector<jql::Ray> rays;
};

class Sampler {
public:
        jql::PCG re{ 0xc01dbeef };
        std::uniform_real_distribution<float> d{ 0, 1 };
        std::vector<jql::Vec2> samples;

        Sampler()
        {
                int N = 3;
                for (int i = 0; i < N - 1; ++i) {
                        for (int j = 0; j < N - 1; ++j) {
                                samples.push_back(
                                        { (i + d(re)) / N, (j + d(re)) / N });
                        }
                }
        }

        std::vector<jql::Vec2> Sample()
        {
                return samples;
        }
};

class Camera {
public:
        Camera(jql::Vec3 pos, jql::Vec3 target, jql::Vec3 up, float FoVy)
        {
                pos_ = pos;
                dir_ = jql::normalize(target - pos);
                up_ = up;
                FoVy_ = FoVy;
        }

        std::vector<FilmSample> GenerateSamples(const Film& film,
                                                Sampler& sampler) const
        {
                auto view = jql::transpose(jql::lookat(pos_, dir_, up_));

                auto offset = jql::Vec3{ -film.w / 2.f, -film.h / 2.f,
                                         -film.h / (2.f * tan(FoVy_ / 2.f)) };
                std::vector<FilmSample> film_samples;
                jql::Vec2 d{ film.w / film.nx, film.h / film.ny };
                for (int y = 0; y < film.ny; ++y) {
                        for (int x = 0; x < film.nx; ++x) {
                                std::vector<jql::Ray> rays;
                                for (jql::Vec2& sample : sampler.Sample()) {
                                        auto film_pos =
                                                jql::cast<jql::Vec3>(
                                                        (jql::Vec2{ (float)x, (float)y } +
                                                         sample) *
                                                        d) +
                                                offset;
                                        film_pos = jql::vector_transform(
                                                view, jql::normalize(film_pos));

                                        rays.emplace_back(pos_, film_pos);
                                }
                                //if (x == 478 && y == 473)
                                //        jql::print("stum\n");
                                film_samples.push_back({ x, y, rays });
                        }
                }
                return film_samples;
        }

private:
        jql::Vec3 pos_, dir_, up_;
        float FoVy_;
};

#endif
