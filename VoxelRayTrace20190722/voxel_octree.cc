
#include "voxel_octree.h"
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include "graphics_math.h"
#include "util.h"
#include "./tiny_obj_loader.h"
#include "./tribox2.h"
#define STB_IMAGE_IMPLEMENTATION
#include "./stb_image.h"

namespace vo
{

jql::PCG Voxel::pcg{ 0xc01dbeef };

class Image {
public:
        std::vector<unsigned char> data;
        int width;
        int height;
        int channels;
};

Image load_image(const std::string& filepath)
{
        Image image{};
        //stbi_set_flip_vertically_on_load(1);
        unsigned char* data = stbi_load(filepath.c_str(), &image.width,
                                        &image.height, &image.channels, 0);
        if (data == nullptr) {
                std::cerr << "Image load failed.\n";
                exit(1);
        }
        int size = image.width * image.height * image.channels;
        image.data.resize(size);
        std::copy_n(data, size, image.data.begin());
        stbi_image_free(data);
        return image;
}

jql::Vec4 get_pixel_from_texcoord(const Image& image, jql::Vec2 texcoord)
{
        int x = jql::clamp<int>(texcoord.x * image.width, 0, image.width - 1);
        int y = jql::clamp<int>(texcoord.y * image.height, 0, image.height - 1);
        y = image.height - 1 - y;
        const unsigned char* p =
                &image.data[(y * image.width + x) * image.channels];

        jql::Vec4 pixel{};

        std::copy_n(p, image.channels, jql::begin(pixel));

        return pixel / 255.f;
}

std::unordered_map<std::uint64_t, Voxel> trivox(std::vector<jql::Vec3>& tri,
                                                float grid_size)
{
        // Compute grid min max.
        jql::Vec3 bmin = jql::min(tri[0], jql::min(tri[1], tri[2])) / grid_size;
        std::transform(jql::begin(bmin), jql::end(bmin), jql::begin(bmin),
                       std::floorf);
        jql::iVec3 gmin = jql::cast<jql::iVec3>(bmin);

        jql::Vec3 bmax = jql::max(tri[0], jql::max(tri[1], tri[2])) / grid_size;
        std::transform(jql::begin(bmax), jql::end(bmax), jql::begin(bmax),
                       std::ceilf);
        jql::iVec3 gmax = jql::cast<jql::iVec3>(bmax);

        std::unordered_map<std::uint64_t, Voxel> voxmap;

        // Iterate each grid
        for (int gx = gmin.x; gx < gmax.x; ++gx) {
                for (int gy = gmin.y; gy < gmax.y; ++gy) {
                        for (int gz = gmin.z; gz < gmax.z; ++gz) {
                                // To aabb.
                                jql::Vec3 bmin{ gx, gy, gz };
                                bmin *= grid_size;
                                jql::AABB3D b{ bmin, bmin + grid_size };
                                auto c = b.center();
                                auto hs = b.size() / 2;
                                int overlap = triBoxOverlap(
                                        jql::begin(c), jql::begin(hs),
                                        (float(*)[3])tri.data());
                                if (!overlap)
                                        continue;

                                std::uint64_t uid;
                                uid = gx & ((1 << 20) - 1);
                                uid <<= 20;
                                uid |= gy & ((1 << 20) - 1);
                                uid <<= 20;
                                uid |= gz & ((1 << 20) - 1);

                                Voxel vox{};
                                vox.aabb = b;

                                voxmap.insert({ uid, vox });
                        }
                }
        }
        return voxmap;
}

std::vector<Voxel> vo::obj2voxel(const std::string& filepath, float voxel_size)
{
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string warn;
        std::string err;

        auto mtldir = jql::get_file_base_dir(filepath);

        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
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

        std::unordered_map<std::uint64_t, Voxel> voxmap;
        std::unordered_map<std::string, Image> imgmap;

        for (size_t s = 0; s < shapes.size(); ++s) {
                size_t index_offset = 0;
                auto& mesh = shapes[s].mesh;
                for (size_t f = 0; f < mesh.num_face_vertices.size(); ++f) {
                        auto fv = mesh.num_face_vertices[f];
                        assert(fv == 3);
                        std::vector<jql::Vec3> tri{ fv };
                        for (size_t v = 0; v < fv; ++v) {
                                auto idx = mesh.indices[index_offset + v];
                                std::copy_n(
                                        &attrib.vertices[3 * idx.vertex_index],
                                        3, jql::begin(tri[v]));
                        }

                        // Tinyobj has per-face material.
                        jql::Vec4 color{};
                        jql::Vec3 normal{};
                        VoxelType type;

                        const auto& mtl = materials[mesh.material_ids[f]];
                        if (mtl.unknown_parameter.find("lightsource") !=
                            mtl.unknown_parameter.end()) {
                                type = VoxelType::LightSource;
                                std::copy_n(mtl.diffuse, 3, jql::begin(color));
                        }
                        else if (!mtl.diffuse_texname.empty()) {
                                type = VoxelType::Object;
                                auto found = imgmap.find(mtl.diffuse_texname);
                                if (found == imgmap.end()) {
                                        auto image =
                                                load_image(mtldir + "\\" +
                                                           mtl.diffuse_texname);
                                        imgmap.insert(
                                                { mtl.diffuse_texname, image });
                                }

                                // Get the diffuse color of first vertex.
                                // Assign later to all voxels related.
                                jql::Vec2 texcoord{};
                                auto idx = mesh.indices[index_offset + 0];
                                std::copy_n(
                                        &attrib.texcoords[2 *
                                                          idx.texcoord_index],
                                        2, jql::begin(texcoord));
                                color = get_pixel_from_texcoord(
                                        imgmap[mtl.diffuse_texname], texcoord);
                                std::copy_n(
                                        &attrib.normals[3 * idx.normal_index],
                                        3, jql::begin(normal));
                        }
                        else
                                ;
                        auto trivoxmap = trivox(tri, voxel_size);
                        for (auto& trivox : trivoxmap) {
                                trivox.second.albedo =
                                        jql::cast<jql::Vec3>(color);
                                trivox.second.normal = normal;
                                trivox.second.type = type;
                                auto p = voxmap.find(trivox.first);
                                if (p == voxmap.end()) {
                                        voxmap.insert(trivox);
                                }
                                else {
                                        p->second.albedo =
                                                (p->second.albedo +
                                                 jql::cast<jql::Vec3>(color)) *
                                                .5f;
                                        p->second.normal = jql::normalize(
                                                p->second.normal + normal);
                                }
                        }
                        index_offset += fv;
                }
        }

        std::vector<Voxel> res;
        for (auto& vox : voxmap)
                res.push_back(vox.second);
        return res;
}

VoxelOctree::VoxelOctree(const jql::AABB3D& aabb)
        : aabb{ aabb }
{
}

bool VoxelOctree::is_leaf() const
{
        return children[0].get() == nullptr;
}

void split(VoxelOctree* root)
{
        assert(root->is_leaf());
        auto root_aabb_size = root->aabb.size();
        auto child_aabb_size = root_aabb_size / 2;
        for (int i = 0; i < 8; ++i) {
                jql::AABB3D child_aabb{};
                jql::iVec3 mask{ i & 4 ? 1 : 0, i & 2 ? 1 : 0, i & 1 ? 1 : 0 };
                child_aabb.min = root->aabb.min + mask * child_aabb_size;
                child_aabb.max = child_aabb.min + child_aabb_size;
                root->children[i] =
                        std::move(std::make_unique<VoxelOctree>(child_aabb));
        }
}

void insert(VoxelOctree* root, const Voxel* voxel)
{
        if (!jql::aabb_is_intersect(root->aabb, voxel->aabb))
                return;
        if (!root->is_leaf()) {
                for (int i = 0; i < 8; ++i)
                        insert(root->children[i].get(), voxel);
                return;
        }
        if (root->voxels.empty() ||
            !jql::compare(root->aabb.size(), 2 * voxel->aabb.size(),
                          std::greater<float>())) {
                root->voxels.push_back(voxel);
                return;
        }

        split(root);
        assert(!root->is_leaf());
        assert(root->voxels.size() == 1);
        const Voxel* old_voxel = root->voxels.front();
        root->voxels.clear();
        for (int i = 0; i < 8; ++i) {
                insert(root->children[i].get(), old_voxel);
                insert(root->children[i].get(), voxel);
        }
}

VoxelOctree build_voxel_octree(const std::vector<Voxel>& voxels)
{
        jql::AABB3D aabb{};
        for (auto& vox : voxels)
                aabb.merge(vox.aabb);
        VoxelOctree root{ aabb };
        for (auto& vox : voxels)
                insert(&root, &vox);
        return root;
}

void travorder(const VoxelOctree& root, const jql::Ray& ray, int ord[8])
{
        assert(!root.is_leaf());

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

const Voxel* ray_march(const VoxelOctree& root, const jql::Ray& ray)
{
        if (!root.aabb.isect(ray, nullptr))
                return nullptr;

        if (root.is_leaf()) {
                if (root.voxels.empty())
                        return nullptr;
                return root.voxels.front();
        }

        struct TravRec {
                const VoxelOctree* root;
                int ord[8];
                int cur;
        };

        std::stack<TravRec> trav;

        {
                TravRec rec{};
                rec.root = &root;
                rec.cur = 0;
                travorder(root, ray, rec.ord);

                trav.push(rec);
        }

        while (!trav.empty()) {
                TravRec& r0 = trav.top();
                auto* child = r0.root->children[r0.ord[r0.cur++]].get();
                if (r0.cur == 8)
                        trav.pop();
                if (!child->aabb.isect(ray, nullptr))
                        continue;
                if (child->is_leaf()) {
                        struct Record {
                                float depth;
                                int i;
                        };

                        std::vector<Record> records;

                        for (int i = 0; i < child->voxels.size(); ++i) {
                                float t{};
                                if (child->voxels[i]->aabb.isectt(ray, &t))
                                        records.push_back({ t, i });
                        }

                        if (records.empty())
                                continue;

                        auto& p = std::min_element(
                                          records.begin(), records.end(),
                                          [](const Record& lhs,
                                             const Record& rhs) {
                                                  return lhs.depth < rhs.depth;
                                          })
                                          ->i;
                        return child->voxels[p];
                }

                TravRec r1{};
                r1.root = child;
                r1.cur = 0;
                travorder(*child, ray, r1.ord);
                trav.push(r1);
        }
        return nullptr;
}
}
