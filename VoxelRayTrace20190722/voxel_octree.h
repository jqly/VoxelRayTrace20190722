
#ifndef JIANGQILEI_OCTREE_H
#define JIANGQILEI_OCTREE_H

#include <list>
#include <vector>
#include <cmath>
#include <memory>
#include <random>
#include <unordered_map>
#include <stack>
#include "./tiny_obj_loader.h"
#include "./graphics_math.h"
#include "./util.h"
#include "./tribox2.h"
#include "./raytri.h"

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
using jql::ISect;
using jql::AABB3D;
using jql::Quat;

namespace gi
{

class VoxelBase {
public:
        virtual AABB3D get_aabb() const = 0;
        virtual bool isect(const Ray& ray, jql::ISect* isect) const = 0;
        virtual Vec3 get_diffuse(const ISect& isect, const Ray& ray,
                                 const Vec3& color) const = 0;
        virtual Vec3 get_albedo(const jql::ISect& isect) const = 0;
        virtual bool is_overlap(const AABB3D& aabb) const = 0;
        virtual bool is_visible() const = 0;

public:
        struct Tex {
                std::vector<unsigned char> data;
                int width;
                int height;
                int channels;
        };

protected:
        static Vec4 texel_fetch(std::string name, Vec2 coord);

private:
        static std::unordered_map<std::string, Tex> texs_;
};

class VoxelOctree {
public:
        // For ray march.
        AABB3D aabb{};
        std::vector<VoxelBase*> voxels;
        std::unique_ptr<VoxelOctree> children[8];
        // For cone trace.
        float coverage{};
        Vec3 diffuse{};
        int depth{};
};

void ray_march_init(VoxelOctree* root, std::vector<VoxelBase*>& voxels,
                    int max_depth);
bool ray_march(VoxelOctree* root, const Ray& ray, VoxelOctree** leaf_ptr,
               VoxelBase** voxel_ptr, ISect* isect, bool even_invisible = false);
void cone_trace_init_filter(VoxelOctree* root);
Vec3 cone_trace(const VoxelOctree& root, const ISect& isect,
                float min_voxel_size);

class Triangle : public VoxelBase {
public:
        Triangle(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 n0, Vec3 n1, Vec3 n2, Vec2 t0,
                 Vec2 t1, Vec2 t2, tinyobj::material_t* mtl);

        AABB3D get_aabb() const override;
        bool isect(const Ray& ray, jql::ISect* isect) const override;
        Vec3 get_diffuse(const ISect& isect, const Ray& light,
                         const Vec3& color) const override;
        Vec3 get_albedo(const jql::ISect& isect) const;
        bool is_overlap(const AABB3D& aabb) const override;
        bool is_visible() const override;

private:
        Vec3 p_[3];
        Vec3 n_[3];
        Vec2 t_[3];
        tinyobj::material_t* mtl_;
        AABB3D aabb_;
};

std::vector<Triangle> obj2voxel(const std::string& filepath,
                                tinyobj::attrib_t* attrib,
                                std::vector<tinyobj::shape_t>* shapes,
                                std::vector<tinyobj::material_t>* materials);

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

class LightProbe : public VoxelBase {
public:
        LightProbe(Vec3 o, float r);

        AABB3D get_aabb() const override;
        bool isect(const Ray& ray, jql::ISect* isect) const override;
        Vec3 get_diffuse(const ISect& isect, const Ray& light,
                         const Vec3& color) const override;
        Vec3 get_albedo(const jql::ISect& isect) const override;
        bool is_overlap(const AABB3D& aabb) const override;
        bool is_visible() const override;
        void gather_light(VoxelOctree* root, float res, Vec3 light_dir);
        Vec3 eval(Vec3 dir) const;

private:
        Vec3 o_;
        float r_;

        SG sgs_[14];

        const Vec3 dirs_[14] = { { 1, 0, 0 },   { 0, 1, 0 },   { 0, 0, 1 },
                                 { -1, 0, 0 },  { 0, -1, 0 },  { 0, 0, -1 },
                                 { 1, 1, 1 },   { 1, 1, -1 },  { 1, -1, 1 },
                                 { 1, -1, -1 }, { -1, 1, 1 },  { -1, 1, -1 },
                                 { -1, -1, 1 }, { -1, -1, -1 } };
};
}

#endif
