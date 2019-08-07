
#include "voxel_octree.h"
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <thread>
#include <future>
#include "thread_pool_cpp/thread_pool.hpp"
#include "graphics_math.h"
#include "util.h"
#include "./tiny_obj_loader.h"
#include "./tribox2.h"
#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"

namespace gi
{

bool is_leaf(const VoxelOctree& root)
{
        return root.children[0] == nullptr;
}

void split(VoxelOctree* root)
{
        assert(is_leaf(*root));
        auto child_aabb_size = root->aabb.size() / 2;
        for (int i = 0; i < 8; ++i) {
                jql::AABB3D child_aabb{};
                jql::iVec3 mask{ i & 4 ? 1 : 0, i & 2 ? 1 : 0, i & 1 ? 1 : 0 };
                child_aabb.min = root->aabb.min + mask * child_aabb_size;
                child_aabb.max = child_aabb.min + child_aabb_size;
                root->children[i] = std::make_unique<VoxelOctree>();
                root->children[i]->aabb = child_aabb;
        }
}

void insert(VoxelOctree* root, VoxelBase* voxel, int cur_depth, int max_depth)
{
        if (!voxel->is_overlap(root->aabb))
                return;
        if (!is_leaf(*root)) {
                for (int i = 0; i < 8; ++i)
                        insert(root->children[i].get(), voxel, cur_depth + 1,
                               max_depth);
                return;
        }

        // Since we always try split when overlap, leaf is either empty or
        // reaches max depth.
        assert(root->voxels.empty() || (cur_depth == max_depth));

        if (cur_depth == max_depth) {
                root->voxels.push_back(voxel);
                return;
        }
        split(root);
        for (int i = 0; i < 8; ++i) {
                insert(root->children[i].get(), voxel, cur_depth + 1,
                       max_depth);
        }
}

void ray_march_init(VoxelOctree* root, std::vector<VoxelBase*>& voxels,
                    int max_depth)
{
        root->aabb = {};
        for (auto& voxel : voxels)
                root->aabb.merge(voxel->get_aabb());
        for (auto& voxel : voxels)
                insert(root, voxel, 1, max_depth);
}

void travorder(const VoxelOctree& root, const jql::Ray& ray, int ord[8])
{
        assert(!is_leaf(root));

        struct Item {
                int ci;
                float dist;
        } items[8];

        for (int ci = 0; ci < 8; ++ci) {
                auto child = root.children[ci].get();
                items[ci].ci = ci;
                items[ci].dist = jql::dot(ray.d, child->aabb.center() - ray.o);
        }
        std::sort(items, items + 8, [](const Item& lhs, const Item& rhs) {
                return lhs.dist < rhs.dist;
        });
        for (int i = 0; i < 8; ++i)
                ord[i] = items[i].ci;
        return;
}

bool ray_march_isect(const std::vector<VoxelBase*>& voxels, const Ray& ray,
                     VoxelBase** voxel_ptr, ISect* isect, bool even_invisible)
{
        struct Record {
                ISect isect;
                float depth;
                int i;
        };

        std::vector<Record> records;

        for (int i = 0; i < voxels.size(); ++i) {
                auto& voxel = voxels[i];
                if ((even_invisible || voxel->is_visible()) &&
                    voxel->isect(ray, isect)) {
                        float t = jql::length(isect->hit - ray.o);
                        records.push_back({ *isect, t, i });
                }
        }

        if (records.empty())
                return false;

        auto p = std::min_element(records.begin(), records.end(),
                                  [](const Record& lhs, const Record& rhs) {
                                          return lhs.depth < rhs.depth;
                                  });
        *voxel_ptr = voxels[p->i];
        *isect = p->isect;
        return true;
}

bool ray_march(VoxelOctree* root, const Ray& ray, VoxelOctree** leaf_ptr,
               VoxelBase** voxel_ptr, ISect* isect, bool even_invisible)
{
        if (!root->aabb.isect(ray, nullptr))
                return false;

        if (is_leaf(*root)) {
                if (ray_march_isect(root->voxels, ray, voxel_ptr, isect,
                                    even_invisible)) {
                        *leaf_ptr = root;
                        return true;
                }
                return false;
        }

        struct TravRec {
                const VoxelOctree* root;
                int ord[8];
                int cur;
        };

        std::stack<TravRec> trav;

        {
                TravRec rec{};
                rec.root = root;
                rec.cur = 0;
                travorder(*root, ray, rec.ord);

                trav.push(rec);
        }

        while (!trav.empty()) {
                TravRec& r0 = trav.top();
                auto* child = r0.root->children[r0.ord[r0.cur++]].get();
                if (r0.cur == 8)
                        trav.pop();

                if (!child->aabb.isect(ray, nullptr))
                        continue;

                if (!is_leaf(*child)) {
                        TravRec r1{};
                        r1.root = child;
                        r1.cur = 0;
                        travorder(*child, ray, r1.ord);
                        trav.push(r1);
                        continue;
                }

                if (ray_march_isect(child->voxels, ray, voxel_ptr, isect,
                                    even_invisible)) {
                        *leaf_ptr = child;
                        return true;
                }
        }
        return false;
}

void cone_trace_init_filter(VoxelOctree* root)
{
        if (is_leaf(*root)) {
                if (root->voxels.empty()) {
                        root->coverage = 0;
                        root->diffuse = {};
                        return;
                }
                root->coverage = 1.f;
                return;
        }
        root->coverage = 0.f;
        root->diffuse = Vec3{};
        for (int i = 0; i < 8; ++i) {
                cone_trace_init_filter(root->children[i].get());
                root->coverage += root->children[i]->coverage;
                root->diffuse += root->children[i]->diffuse;
        }
        root->coverage /= 8.f;
        root->diffuse /= 8.f;
}

class Cone {
public:
        jql::Vec3 o;
        jql::Vec3 d;
        // tan half aperture.
        // tan(pi/6) = sqrt(3)/3
        static constexpr float aperture = 0.577350269f;
        static constexpr float step = .1f;
        static constexpr float litness_decay = 1.f;
};

static constexpr jql::Vec4 HemiCones[] = {
        { 0.000000f, 0.000000f, 1.0f, 0.25f },
        { 0.000000f, 0.866025f, 0.5f, 0.15f },
        { 0.823639f, 0.267617f, 0.5f, 0.15f },
        { 0.509037f, -0.700629f, 0.5f, 0.15f },
        { -0.509037f, -0.700629f, 0.5f, 0.15f },
        { -0.823639f, 0.267617f, 0.5f, 0.15f },
};

jql::Mat3 orthonormal_basis(jql::Vec3 n)
{
        float s = (0.0f > n.z) ? -1.0f : 1.0f;
        float a0 = -1.0f / (s + n.z);
        float a1 = n.x * n.y * a0;

        jql::Vec3 t = { 1.0f + s * n.x * n.x * a0, s * a1, -s * n.x };
        jql::Vec3 b = { a1, s + n.y * n.y * a0, -n.y };

        return { t, b, n };
}

Vec3 cone_trace(const VoxelOctree& root, const Cone& cone, float min_voxel_size)
{
        float mindist = 1.414f * min_voxel_size;
        float maxdist = jql::length(root.aabb.size());

        float dist = mindist;
        float opacity = 0.f;
        Vec3 diffuse{};
        while (dist < maxdist && opacity < 1.f) {
                auto p = cone.o + cone.d * dist;
                auto diam = std::max(mindist, cone.aperture * 2.f * dist);
                if (maxdist < diam)
                        break;
                int split_level = std::log2f(maxdist / diam);
                const VoxelOctree* tree = &root;
                while (!is_leaf(*tree) && split_level) {
                        auto center = tree->aabb.center();
                        int i = 0;
                        i += (p.x > center.x ? 4 : 0);
                        i += (p.y > center.y ? 2 : 0);
                        i += (p.z > center.z ? 1 : 0);
                        tree = tree->children[i].get();
                        split_level--;
                }
                if (split_level == 0) {
                        auto transparency = jql::clamp(1.f - opacity, 0.f, 1.f);
                        float a = tree->coverage * cone.step;
                        diffuse += (1.f / (1 + cone.litness_decay * dist)) *
                                   transparency * tree->coverage *
                                   tree->diffuse;
                        opacity += transparency * a;
                }
                dist += cone.step * diam;
        }

        return diffuse;
}

Vec3 cone_trace(const VoxelOctree& root, const ISect& isect,
                float min_voxel_size)
{
        auto tangent_to_uvw = orthonormal_basis(isect.normal);

        Cone cone;
        cone.o = isect.hit;
        Vec3 diffuse{};

        for (int i = 0; i < 6; ++i) {
                auto d = jql::cast<Vec3>(HemiCones[i]);
                const float weight = HemiCones[i].w;
                cone.d = jql::normalize(jql::dot(tangent_to_uvw, d));
                diffuse += weight * cone_trace(root, cone, min_voxel_size);
        }

        return diffuse;
}

std::vector<Triangle> obj2voxel(const std::string& filepath,
                                tinyobj::attrib_t* attrib,
                                std::vector<tinyobj::shape_t>* shapes,
                                std::vector<tinyobj::material_t>* materials)
{

        std::string warn;
        std::string err;

        auto mtldir = jql::get_file_base_dir(filepath) + "/";

        bool ret = tinyobj::LoadObj(attrib, shapes, materials, &warn, &err,
                                    filepath.c_str(), mtldir.c_str(), true);

        if (!warn.empty()) {
                std::cout << warn << std::endl;
        }

        if (!err.empty()) {
                std::cerr << err << std::endl;
        }

        if (!ret) {
                std::cerr << "Error loading .obj.\n";
                exit(1);
        }

        std::vector<Triangle> tris;

        for (int s = 0; s < shapes->size(); ++s) {
                auto& mesh = (*shapes)[s].mesh;
                for (int f = 0; f < mesh.num_face_vertices.size(); ++f) {
                        auto fv = mesh.num_face_vertices[f];
                        assert(fv == 3);
                        Vec3 trivs[3], trins[3];
                        Vec2 trits[3];
                        for (int v = 0; v < fv; ++v) {
                                auto idx = mesh.indices[f * 3 + v];
                                std::copy_n(
                                        &attrib->vertices[3 * idx.vertex_index],
                                        3, jql::begin(trivs[v]));
                                assert(idx.normal_index != -1);
                                std::copy_n(
                                        &attrib->normals[3 * idx.normal_index],
                                        3, jql::begin(trins[v]));
                                if (idx.texcoord_index == -1)
                                        trits[v] = {};
                                else
                                        std::copy_n(
                                                &attrib->texcoords
                                                         [2 *
                                                          idx.texcoord_index],
                                                2, jql::begin(trits[v]));
                        }
                        auto* mtl = &(*materials)[mesh.material_ids[f]];
                        if (mtl->diffuse_texname.rfind(mtldir, 0) != 0) {
                                mtl->diffuse_texname =
                                        mtldir + mtl->diffuse_texname;
                        }
                        Triangle tri{ trivs[0], trivs[1], trivs[2], trins[0],
                                      trins[1], trins[2], trits[0], trits[1],
                                      trits[2], mtl };
                        tris.push_back(tri);
                }
        }
        return tris;
}

VoxelBase::Tex load_image(const std::string& filepath)
{
        VoxelBase::Tex tex{};
        //stbi_set_flip_vertically_on_load(1);
        unsigned char* data = stbi_load(filepath.c_str(), &tex.width,
                                        &tex.height, &tex.channels, 0);
        if (data == nullptr) {
                std::cerr << "Image load failed.\n";
                exit(1);
        }
        int size = tex.width * tex.height * tex.channels;
        tex.data.resize(size);
        std::copy_n(data, size, tex.data.begin());
        stbi_image_free(data);
        return tex;
}

std::unordered_map<std::string, VoxelBase::Tex> VoxelBase::texs_;

float unit_cycle(float s)
{
        while (s > 1.f)
                s -= 1.f;
        while (s < 0.f)
                s += 1.f;
        return s;
}

Vec4 VoxelBase::texel_fetch(std::string name, Vec2 coord)
{
        auto found = texs_.find(name);
        if (found == texs_.end()) {
                texs_.insert({ name, load_image(name) });
                found = texs_.find(name);
        }
        assert(found != texs_.end());

        auto& tex = found->second;
        int x = jql::clamp<int>(unit_cycle(coord.x) * tex.width, 0,
                                tex.width - 1);
        int y = jql::clamp<int>(unit_cycle(coord.y) * tex.height, 0,
                                tex.height - 1);
        y = tex.height - 1 - y;
        const unsigned char* p = &tex.data[(y * tex.width + x) * tex.channels];

        jql::Vec4 pixel{};

        std::copy_n(p, tex.channels, jql::begin(pixel));
        return pixel / 255.f;
}
Triangle::Triangle(Vec3 p0, Vec3 p1, Vec3 p2, Vec3 n0, Vec3 n1, Vec3 n2,
                   Vec2 t0, Vec2 t1, Vec2 t2, tinyobj::material_t* mtl)
        : p_{ p0, p1, p2 }
        , n_{ jql::normalize(n0), jql::normalize(n1), jql::normalize(n2) }
        , t_{ t0, t1, t2 }
        , mtl_{ mtl }
{
        aabb_ = AABB3D{ p_, p_ + 3 };
}

AABB3D Triangle::get_aabb() const
{
        return aabb_;
}

bool Triangle::isect(const Ray& ray, jql::ISect* isect) const
{
        double dt{}, du{}, dv{};
        double rayo[]{ ray.o.x, ray.o.y, ray.o.z };
        double rayd[]{ ray.d.x, ray.d.y, ray.d.z };
        double p0[]{ p_[0].x, p_[0].y, p_[0].z };
        double p1[]{ p_[1].x, p_[1].y, p_[1].z };
        double p2[]{ p_[2].x, p_[2].y, p_[2].z };
        int result = intersect_triangle3(rayo, rayd, p0, p1, p2, &dt, &du, &dv);
        if (result != 1)
                return false;
        float u = jql::clamp<float>(du, 0, 1);
        float v = jql::clamp<float>(dv, 0, 1);
        float w = jql::clamp<float>(1 - u - v, 0, 1);
        auto ntmp = n_[0] * w + n_[1] * u + n_[2] * v;
        isect->normal = jql::normalize(ntmp);
        isect->hit = ray.o + (float)dt * ray.d;
        //isect->hit = p_[0] * w + p_[1] * u + p_[2] * v;
        //auto hit = ray.o + (float)dt * ray.d;
        //if (jql::length(isect->hit - hit) > .001f)
        //        jql::print("bc hit: {}, hit: {}\n", isect->hit, hit);
        return true;
}

Vec3 Triangle::get_diffuse(const ISect& isect, const Ray& light,
                           const Vec3& color) const
{
        Vec3 albedo = get_albedo(isect);
        auto tmp = jql::dot(isect.normal, -light.d);
        tmp = jql::clamp(tmp, 0.f, 1.f);
        return albedo * tmp * color;
}

Vec3 Triangle::get_albedo(const jql::ISect& isect) const
{
        Vec3 albedo{};
        if (mtl_->diffuse_texname.empty()) {
                std::copy_n(mtl_->diffuse, 3, jql::begin(albedo));
                return albedo;
        }
        auto bc = jql::barycentric(isect.hit, p_[0], p_[1], p_[2]);
        bc = jql::clamp(bc, 0.f, 1.f);
        Vec2 texcoord = bc.x * t_[0] + bc.y * t_[1] + bc.z * t_[2];
        auto albedo4 = texel_fetch(mtl_->diffuse_texname, texcoord);
        albedo = jql::cast<Vec3>(albedo4);
        return albedo;
}

bool Triangle::is_overlap(const AABB3D& aabb) const
{
        auto center = aabb.center();
        auto half = aabb.size() / 2.f;
        return 1 == triBoxOverlap(jql::begin(center), jql::begin(half),
                                  (float(*)[3])p_);
}
bool Triangle::is_visible() const
{
        return true;
}
SG::SG(Vec3 a, Vec3 d, float s)
        : a{ a }
        , d{ jql::normalize(d) }
        , s{ s }
{
}
Vec3 SG::eval(Vec3 dir) const
{
        float tmp = jql::dot(dir, d);
        auto ret = a * std::expf(s * (tmp - 1));
        return ret;
}
Vec3 SG::integral() const
{
        float tmp = 1.0f - exp(-2.0f * s);
        return 2 * jql::pi * (a / s) * tmp;
}
SG prod(const SG& lhs, const SG& rhs)
{
        Vec3 d = (lhs.s * lhs.d + rhs.s * rhs.d) / (lhs.s + rhs.s);
        float dl = jql::length(d);
        float lm = lhs.s + rhs.s;

        return { lhs.a * rhs.a * std::expf(lm * (dl - 1)), d, lm * dl };
}
Vec3 inner_prod(const SG& lhs, const SG& rhs)
{
        float uml = jql::length(lhs.s * lhs.d + rhs.s * rhs.d);
        Vec3 expo = std::expf(uml - lhs.s - rhs.s) * lhs.a * rhs.a;
        float other = 1.0f - std::expf(-2.0f * uml);
        return (2.0f * jql::pi * expo * other) / uml;
}
LightProbe::LightProbe(Vec3 o, float r)
        : o_{ o }
        , r_{ r }
{
}

AABB3D LightProbe::get_aabb() const
{
        return AABB3D{ o_ - r_, o_ + r_ };
}

bool LightProbe::isect(const Ray& ray, jql::ISect* isect) const
{
        float t{};
        if (jql::sphere_ray_isect({ o_, r_ }, ray, &t)) {
                *isect = jql::ISect{ ray.o + t * ray.d, isect->hit - o_ };
                return true;
        }
        return false;
}

Vec3 LightProbe::get_diffuse(const ISect& isect, const Ray& light,
                             const Vec3& color) const
{
        return get_albedo(isect);
}

Vec3 LightProbe::get_albedo(const jql::ISect& isect) const
{
        Vec3 dir = jql::normalize(isect.hit - o_);
        return eval(dir);
}
bool LightProbe::is_overlap(const AABB3D& aabb) const
{
        auto c1 = aabb.min;
        auto c2 = aabb.max;
        auto squared = [](float v) { return v * v; };
        float dist_squared = squared(r_);
        if (o_.x < c1.x)
                dist_squared -= squared(o_.x - c1.x);
        else if (o_.x > c2.x)
                dist_squared -= squared(o_.x - c2.x);
        if (o_.y < c1.y)
                dist_squared -= squared(o_.y - c1.y);
        else if (o_.y > c2.y)
                dist_squared -= squared(o_.y - c2.y);
        if (o_.z < c1.z)
                dist_squared -= squared(o_.z - c1.z);
        else if (o_.z > c2.z)
                dist_squared -= squared(o_.z - c2.z);
        return dist_squared > 0;
}

bool LightProbe::is_visible() const
{
        return false;
}
void LightProbe::gather_light(VoxelOctree* root, float res, Vec3 light_dir)
{
        for (int i = 0; i < 14; ++i) {

                auto dir = dirs_[i];

                // for each directions, we sample 4 rays around.
                Vec3 litness_avg{};

                const Quat qs[4]{
                        Quat::angle_axis(jql::to_radian(5), Vec3{ 1, 0, 0 }),
                        Quat::angle_axis(jql::to_radian(-4), Vec3{ 0, 1, 0 }),
                        Quat::angle_axis(jql::to_radian(2), Vec3{ 0, 0, 1 }),
                        Quat::angle_axis(jql::to_radian(-4), Vec3{ 1, 1, 1 })
                };

                for (int s = 0; s < 4; ++s) {
                        auto& q = qs[s];
                        auto r_ = jql::rotate(q, dir);

                        Ray ray{ o_, r_ };
                        VoxelOctree* leaf_ptr;
                        VoxelBase* voxel_ptr;
                        ISect isect;
                        if (!ray_march(root, ray, &leaf_ptr, &voxel_ptr,
                                       &isect)) {
                                auto litness = jql::dot(light_dir, ray.d);
                                litness = jql::clamp(litness, 0.f, 1.f);
                                litness_avg += .25f * litness;
                                continue;
                        }
                        voxel_ptr->isect(ray, &isect);
                        auto litness = cone_trace(*root, isect, res);
                        litness_avg += .25f * litness;
                }

                sgs_[i] = SG{ litness_avg, dir, 10 };
        }
}
Vec3 LightProbe::eval(Vec3 dir) const
{
        Vec3 litness{};
        for (auto& sg : sgs_)
                litness += sg.eval(dir);
        return litness;
}
}
