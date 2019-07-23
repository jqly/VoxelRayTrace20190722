
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
        jql::Vec4 color;
        jql::Vec3 normal;
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
