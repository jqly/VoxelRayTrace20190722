
#ifndef JIANGQILEI_OCTREE_H
#define JIANGQILEI_OCTREE_H

#include <list>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <stack>
#include "./graphics_math.h"
#include "./util.h"

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
using jql::AABB3D;
using jql::Quat;

namespace vo
{

class SG {
public:
        Vec3 a;
        Vec3 d;
        float s;

        SG() = default;
        SG(Vec3 a, Vec3 d, float s);
        Vec3 eval(Vec3 dir) const;
        Vec3 integral() const;
};

SG prod(const SG& lhs, const SG& rhs);
Vec3 inner_prod(const SG& lhs, const SG& rhs);

enum class VoxelType { Unknown, Object, LightSource, LightProbe };

class Voxel {
public:
        VoxelType type;
        AABB3D aabb;
        Vec3 albedo;
        Vec3 normal;
        Vec3 litness;

        bool scatter(const jql::Ray& iray, const jql::ISect& isect,
                     jql::Vec3* att, jql::Ray* sray) const;
        static jql::PCG pcg;
};

std::vector<Voxel> obj2voxel(const std::string& filepath, float voxel_size);

class VoxelOctree {
public:
        jql::AABB3D aabb;
        std::vector<Voxel*> voxels;
        std::unique_ptr<VoxelOctree> children[8];
        float opacity = -1.f;
        Vec3 D; // normal. sigma^2=(1-D)/D.
        Vec3 litness;
        Vec3 albedo;

        VoxelOctree(const jql::AABB3D& aabb);

        bool is_leaf() const;
};

VoxelOctree build_voxel_octree(std::vector<Voxel>& voxels);
float voxel_filter(VoxelOctree* root);

Voxel* ray_march(const VoxelOctree& root, const jql::Ray& ray);

jql::Vec3 compute_litness(const VoxelOctree& root, const jql::ISect& isect,
                          float res);

class LightProbe {
public:
        const Vec3 o;
        const float r;

        LightProbe(Vec3 o, float r)
                : o{ o }
                , r{ r }
        {
                ;
        }

        void gather_light(const vo::VoxelOctree& root, float res,
                          Vec3 light_dir)
        {
                for (int i = 0; i < 14; ++i) {

                        auto dir = dirs_[i];

                        // for each directions, we sample 4 rays around.
                        Vec3 litness_avg{};

                        const Quat qs[4]{ Quat::angle_axis(jql::to_radian(5),
                                                           Vec3{ 1, 0, 0 }),
                                          Quat::angle_axis(jql::to_radian(-4),
                                                           Vec3{ 0, 1, 0 }),
                                          Quat::angle_axis(jql::to_radian(2),
                                                           Vec3{ 0, 0, 1 }),
                                          Quat::angle_axis(jql::to_radian(-4),
                                                           Vec3{ 1, 1, 1 }) };

                        for (int s = 0; s < 4; ++s) {
                                auto& q = qs[s];
                                auto r_ = jql::rotate(q, dir);

                                Ray ray{ o, r_ };
                                Voxel* voxel = ray_march(root, ray);
                                if (!voxel) {
                                        auto litness =
                                                jql::dot(light_dir, ray.d);
                                        litness = jql::clamp(litness, 0.f, 1.f);
                                        litness_avg += .25f * litness;
                                        continue;
                                }
                                jql::ISect isect{};
                                voxel->aabb.isect(ray, &isect);
                                auto litness =
                                        compute_litness(root, isect, res);
                                litness_avg += .25f * litness;
                        }

                        sgs_[i] = SG{ litness_avg, dir, 10 };
                        
                }
        }

        bool isectt(const Ray& ray, float* t)
        {
                if (jql::sphere_ray_isect({ o, r }, ray, t)) {
                        return true;
                }
                return false;
        }

        bool isect(const Ray& ray, jql::ISect* isect)
        {
                float t{};
                if (jql::sphere_ray_isect({ o, r }, ray, &t)) {
                        *isect =
                                jql::ISect{ ray.o + t * ray.d, isect->hit - o };
                        return true;
                }
                return false;
        }

        Vec3 eval(Vec3 dir)
        {
                Vec3 litness{};
                for (auto& sg : sgs_)
                        litness += sg.eval(dir);
                return litness;
        }

private:
        SG sgs_[14];

        const Vec3 dirs_[14] = { { 1, 0, 0 },   { 0, 1, 0 },   { 0, 0, 1 },
                                 { -1, 0, 0 },  { 0, -1, 0 },  { 0, 0, -1 },
                                 { 1, 1, 1 },   { 1, 1, -1 },  { 1, -1, 1 },
                                 { 1, -1, -1 }, { -1, 1, 1 },  { -1, 1, -1 },
                                 { -1, -1, 1 }, { -1, -1, -1 } };
};
}

#endif
