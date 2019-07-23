
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

        jql::Vec3 random_in_unit_sphere() const
        {
                std::uniform_real_distribution<float> distr{ -1, 1 };
                jql::Vec3 p;
                do {
                        p = 2.f * jql::Vec3{ distr(pcg), distr(pcg),
                                             distr(pcg) } -
                            jql::Vec3{ 1, 1, 1 };
                } while (jql::dot(p, p) >= 1.f);
                return p;
        }

        bool scatter(const jql::Ray& iray, const jql::ISect& isect,
                     jql::Vec3* att, jql::Ray* sray) const
        {

                auto tmp = jql::dot(isect.normal, -iray.d);
                if (tmp <= 0)
                        return false;

                *sray = jql::Ray{ isect.hit, isect.normal+random_in_unit_sphere() };
                *att = albedo;
                return true;
        }
        static jql::PCG pcg;
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
