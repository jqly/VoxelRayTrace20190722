
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

namespace vo
{

class SG {
public:
        jql::Vec3 axis;
        float sharpness = -1;
        jql::Vec3 amplitude;

        SG()
                : axis{ 0, 0, 0 }
                , sharpness{ -1 }
                , amplitude{ 0, 0, 0 }
        {
        }

        SG(jql::Vec3 axis, float sharpness, jql::Vec3 amplitude)
                : axis{ jql::normalize(axis) }
                , sharpness{ sharpness }
                , amplitude{ amplitude }
        {
        }

        jql::Vec3 eval(jql::Vec3 d) const
        {
                if (sharpness < 0)
                        return {};
                float tmp = jql::dot(d, axis);
                return amplitude * std::expf(sharpness * (tmp - 1.f));
        }
};

SG dot(const SG& lhs, const SG& rhs);

enum class VoxelType { Unknown, Object, LightSource };

class Voxel {
public:
        VoxelType type;
        jql::AABB3D aabb;
        jql::Vec3 albedo;
        jql::Vec3 normal;
        mutable SG sg{};

        bool scatter(const jql::Ray& iray, const jql::ISect& isect,
                     jql::Vec3* att, jql::Ray* sray) const;
        static jql::PCG pcg;
};

std::vector<Voxel> obj2voxel(const std::string& filepath, float voxel_size);

class VoxelOctree {
public:
        jql::AABB3D aabb;
        std::vector<const Voxel*> voxels;
        std::unique_ptr<VoxelOctree> children[8];
        float opacity = -1.f;
        SG sg{};

        VoxelOctree(const jql::AABB3D& aabb);

        bool is_leaf() const;
};

VoxelOctree build_voxel_octree(const std::vector<Voxel>& voxels);
float voxel_filter(VoxelOctree* root);

const Voxel* ray_march(const VoxelOctree& root, const jql::Ray& ray);

jql::Vec3 compute_litness(const VoxelOctree& root, const jql::ISect& isect,
                          float res);
}

#endif
