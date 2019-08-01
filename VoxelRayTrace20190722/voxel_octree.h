
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
}

#endif
