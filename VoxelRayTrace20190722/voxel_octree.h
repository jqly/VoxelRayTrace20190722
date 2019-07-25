
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

enum class VoxelType { Object, LightSource };

class Voxel {
public:
        VoxelType type;
        jql::AABB3D aabb;
        jql::Vec3 albedo;
        jql::Vec3 normal;

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

        VoxelOctree(const jql::AABB3D& aabb);

        bool is_leaf() const;
};

VoxelOctree build_voxel_octree(const std::vector<Voxel>& voxels);

const Voxel* ray_march(const VoxelOctree& root, const jql::Ray& ray);

float compute_ao(const VoxelOctree& root, const jql::ISect& isect, float res);
}

#endif
