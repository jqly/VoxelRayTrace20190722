
#ifndef JIANGQILEI_OCTREE_H
#define JIANGQILEI_OCTREE_H

#include <list>
#include <vector>
#include <cmath>
#include <memory>
#include <stack>
#include "./graphics_math.h"
#include "./util.h"

namespace vo
{

class Voxel {
public:
        jql::AABB3D aabb;
        jql::Vec3 albedo;
        jql::Vec3 normal;
        bool scatter(const jql::Ray& iray, const jql::ISect& isect,
                     jql::Vec3* att, jql::Ray* sray) const
        {
                auto tmp = jql::dot(isect.normal, -iray.d);
                if (tmp <= 0)
                        return false;
                auto d = 2 * tmp * isect.normal +
                         iray.d;
                *sray = jql::Ray{ isect.hit, d };
                *att = albedo;
                return true;
        }
};

std::vector<Voxel> obj2voxel(const std::string& filepath, float voxel_size);

class VoxelOctree {
public:
        jql::AABB3D aabb;
        std::vector<const Voxel*> voxels;
        std::unique_ptr<VoxelOctree> children[8];

        VoxelOctree(const jql::AABB3D& aabb);

        bool is_leaf() const;
};

VoxelOctree build_voxel_octree(const std::vector<Voxel>& voxels);

const Voxel* ray_march(const VoxelOctree& root, const jql::Ray& ray);
}

#endif
