#pragma once

#include "reycode/reycode.h"
#include "reycode/mesh/hexcore.h"
#include <Kokkos_Core.hpp>

namespace reycode {
    template<class Exec, class Mem>
    class HexcoreAMR {
        Hexcore<Exec, Mem> &hexcore;

        using Hexcore = reycode::Hexcore<Exec, Mem>;
        using Morton_Code = typename Hexcore::Morton_Code;
        using Refinement_Mask = typename Hexcore::Refinement_Mask;
        using Device = typename Hexcore::Device;
        using Boundary_Patch = typename Hexcore::Update_Desc::Boundary_Patch;
        using Cell = typename Hexcore::Cell;
        using Face = typename Hexcore::Face;
    public:
        explicit HexcoreAMR(Hexcore &hexcore) : hexcore(hexcore) {}

        HexcoreAMR &uniform(vec3 extent, uvec3 dims) {
            uint64_t cell_count = dims.x * dims.y * dims.z;

            Kokkos::View<Morton_Code *, Device> morton_codes0("UNIFORM MORTON", cell_count);
            Kokkos::View<Refinement_Mask *, Device> refinement_mask0("UNIFORM MASK", cell_count);

            uint64_t max_refinement = Hexcore::MAX_REFINEMENT;

            uvec3 ghost = uvec3(1);
            vec3 dx = extent / vec3(dims);

            Kokkos::parallel_for("Uniform morton ",
                                 Kokkos::RangePolicy<Exec>(0, cell_count),
                                 KOKKOS_LAMBDA(uint64_t idx) {

                                     uvec3 pos;
                                     pos.x = (idx % dims.x);
                                     pos.y = (idx / dims.x) % dims.y;
                                     pos.z = (idx / dims.x / dims.y);

                                     pos = pos + ghost;
                                     pos = pos << max_refinement;

                                     typename Hexcore::Morton_Code code = Hexcore::morton_encode(pos);
                                     typename Hexcore::Refinement_Mask refinement = {};
                                     refinement.ghost = false;

                                     morton_codes0(idx) = code;
                                     refinement_mask0(idx) = refinement;
                                 });

            typename Hexcore::Init_Desc init_desc;
            init_desc.capacity = morton_codes0.size();
            init_desc.extent = extent;
            init_desc.dx = dx;

            typename Hexcore::Update_Desc update_desc;
            update_desc.count = morton_codes0.size();
            update_desc.morton_codes = morton_codes0;
            update_desc.refinement_mask = refinement_mask0;

            hexcore.init(init_desc);
            hexcore.update(update_desc);

            update_ghost();

            return *this;
        }

        void adapt(uint32_t levels) {
            assert(levels <= Hexcore::MAX_REFINEMENT);
            vec3 extent = hexcore.extent();

            for (uint32_t i = 0; i < levels; i++) {
                uint32_t count = hexcore.cell_count();

                enum class Refine_Flag : char {
                    NotACell,
                    Untouched,
                    Refine
                };

                Kokkos::View<Refine_Flag *, Mem> refine_cell_mask0("refine mask", count);
                Kokkos::View<Refine_Flag *, Mem> refine_cell_mask1("refinement mask", count);

                hexcore.for_each_cell("Refine",
                                      KOKKOS_LAMBDA(Cell &cell) {
                                          uint32_t idx = cell.id();
                                          Refinement_Mask mask = cell.refinement_mask();
                                          uint32_t level = mask.level;

                                          vec3 center = extent * vec3(0.3,0.5,0.5);
                                          real radius = 0.2;

                                          auto sdf = [=](vec3 pos) {
                                              return length(center - pos) - radius;
                                          };

                                          vec3 p = cell.center();
                                          vec3 dx = cell.dx();

                                          bool inside = false;
                                          bool refine = false;

                                          real d[6];
                                          for (uint32_t j = 0; j < CUBE_FACE_COUNT; j++) {
                                              d[j] = sdf(p + dx * vec3(cube_normals[j]));
                                          }

                                          bool sign_change = false;
                                          for (uint32_t j = 1; j < CUBE_FACE_COUNT; j++) {
                                              sign_change = sign_change || (d[j] > 0 != d[0] > 0);
                                          }

                                          inside = !sign_change && d[0] < 0;

                                          bool refine_intersect = sign_change && mask.level < 4;
                                          bool refine_box = center - radius*vec3(5,5,5) < p
                                                            && p < (center + radius*vec3(8,5,5));

                                          refine = (mask.level<2 && refine_box) || (mask.level<4 && refine_intersect);

                                          Refine_Flag flag = Refine_Flag::Untouched;
                                          if (inside) flag = Refine_Flag::NotACell;
                                          else if (refine) flag = Refine_Flag::Refine;

                                          refine_cell_mask0(idx) = flag;
                                      });

                printf("=====\n");

                uint32_t balance_it = 10;
                for (uint32_t i = 0; i < balance_it; i++) {
                    hexcore.for_each_cell("Balance",
                                          KOKKOS_LAMBDA(Cell &cell) {
                                              uint32_t idx = cell.id();
                                              Refinement_Mask mask = cell.refinement_mask();
                                              uint32_t level = mask.level;

                                              Refine_Flag flag = refine_cell_mask0(idx);
                                              if (flag == Refine_Flag::NotACell) {
                                                  refine_cell_mask1(idx) = Refine_Flag::NotACell;
                                                  return;
                                              }

                                              bool refine = flag==Refine_Flag::Refine;
                                              bool unbalanced = false;

                                              for (Face face: cell.faces()) {
                                                  Cell neigh = face.neigh();
                                                  if (neigh.id() == UINT64_MAX) continue;
                                                  Refinement_Mask neigh_mask = neigh.refinement_mask();
                                                  Refine_Flag flag = refine_cell_mask0(neigh.id());
                                                  if (flag==Refine_Flag::NotACell) continue;

                                                  int level_diff = int(neigh_mask.level + (flag==Refine_Flag::Refine)) -int(mask.level + refine);
                                                  if (level_diff > 1) unbalanced = true;
                                              }

                                              vec3 center = cell.center() + 0.5_R * cell.dx();

                                              refine_cell_mask1(idx) = refine||unbalanced ? Refine_Flag::Refine
                                                                              : Refine_Flag::Untouched;
                                          });

                    std::swap(refine_cell_mask0, refine_cell_mask1);
                }

                auto &refine_cell_mask = refine_cell_mask0;

                uint64_t count_refined;
                Kokkos::View<uint64_t *, Mem> cell_index("cell index", count);
                Kokkos::parallel_scan("Scan",
                                      Kokkos::RangePolicy<Exec>(0, count),
                                      KOKKOS_LAMBDA(uint64_t i, uint64_t &partial_sum, bool is_final) {
                                          if (is_final) cell_index(i) = partial_sum;

                                          Refine_Flag flag = refine_cell_mask(i);
                                          if (flag == Refine_Flag::Untouched) partial_sum++;
                                          if (flag == Refine_Flag::Refine) partial_sum += 8;
                                      }, count_refined);

                Kokkos::View<Morton_Code *, Device> morton_codes1("morton codes1", count_refined);
                Kokkos::View<Refinement_Mask *, Device> refinement_mask1("refinement mask", count_refined);

                hexcore.for_each_cell("Refine",
                                      KOKKOS_LAMBDA(Cell &cell) {
                                          uint32_t i = cell.id();
                                          Refinement_Mask mask = cell.refinement_mask();
                                          Refine_Flag flag = refine_cell_mask(i);
                                          if (flag == Refine_Flag::NotACell) return;
                                          bool refined = flag == Refine_Flag::Refine;
                                          uint64_t offset = cell_index(i);
                                          Morton_Code code = cell.morton_code();

                                          mask.up = 0;
                                          mask.down = 0;
                                          int diff[6][4] = {};

                                          for (Face &face: cell.faces()) {
                                              Cell neigh = face.neigh();
                                              if (neigh.id()==UINT64_MAX || neigh.is_ghost()) continue;
                                              Refinement_Mask neigh_mask = neigh.refinement_mask();
                                              Refine_Flag flag = refine_cell_mask(neigh.id());
                                              if (flag == Refine_Flag::NotACell) continue;

                                              uint32_t neigh_level = neigh_mask.level + (flag== Refine_Flag::Refine);
                                              int d = int(neigh_level) - int(mask.level + refined);

                                              if (abs(d) > 1) {
                                                  vec3 center = face.center();
                                                  printf("Unbalanced fatal: %i, diff: %i\n", i, d);
                                              }
                                              diff[face.cube_face()][face.sub_cell()] = d;
                                          }

                                          if (refined) {
                                              uvec3 cell_pos = Hexcore::morton_decode(code);
                                              for (uint32_t j = 0; j < 8; j++) {
                                                  uvec3 o = uvec3(j % 2, (j / 2) % 2, j / 4);
                                                  uvec3 sub_cell_offset = o << (Hexcore::MAX_REFINEMENT - (mask.level
                                                                                                           + 1));

                                                  Refinement_Mask submask = mask;
                                                  assert(!submask.ghost);
                                                  submask.level = mask.level + 1;

                                                  submask.up = 0;
                                                  submask.down = 0;
                                                  for (int j = 0; j < CUBE_FACE_COUNT; j++) {
                                                      uint32_t axis = cube_face_axis[j];
                                                      uint32_t axis1 = (axis + 1) % 3;
                                                      uint32_t axis2 = (axis + 2) % 3;
                                                      bool outside = cube_normals[j][axis] > 0 == o[axis] > 0;
                                                      int d = 0;
                                                      if (outside) {
                                                          if (mask.up & (1 << j)) {
                                                              uint32_t subcell = o[axis1] + o[axis2] * 2;
                                                              d = diff[j][subcell];
                                                          } else {
                                                              d = diff[j][0];
                                                          }
                                                      }
                                                      if (d > 0) submask.up |= 1 << j;
                                                      if (d < 0) submask.down |= 1 << j;
                                                  }

                                                  assert(submask.level <= Hexcore::MAX_REFINEMENT);

                                                  uvec3 pos = cell_pos + sub_cell_offset;
                                                  morton_codes1(offset + j) = Hexcore::morton_encode(pos);
                                                  refinement_mask1(offset + j) = submask;
                                                  morton_codes1(offset + j);
                                              }
                                          } else {
                                              mask.up = 0;
                                              mask.down = 0;
                                              for (int j = 0; j < CUBE_FACE_COUNT; j++) {
                                                  if (diff[j][0] > 0) mask.up |= 1 << j;
                                                  if (diff[j][0] < 0) mask.down |= 1 << j;
                                              }

                                              morton_codes1(offset) = code;
                                              refinement_mask1(offset) = mask;
                                          }
                                      });

                typename Hexcore::Update_Desc desc;
                desc.count = morton_codes1.size();
                desc.morton_codes = morton_codes1;
                desc.refinement_mask = refinement_mask1;
                //desc.patches = std::move(patches1);
                hexcore.update(desc);
            }

            update_ghost();
        }

        void update_ghost() {
            vec3 extent = hexcore.extent();
            Kokkos::View<uint32_t *, Mem> cell_offset;

            Kokkos::vector<uint64_t, Mem> patch_base(CUBE_FACE_COUNT + 1);
            Kokkos::vector<uint64_t, Mem> patch_counts(CUBE_FACE_COUNT + 1);

            auto patch_id = [extent](Cube_Face face, vec3 center) {
                bool inside = vec3(0.1) < center && center < 0.9*extent;
                uint32_t patch_id = inside ? CUBE_FACE_COUNT : face;
                return patch_id;
            };

            hexcore.for_each_cell("Refine",
                                  KOKKOS_LAMBDA(Cell &cell) {
                                      for (Face &face: cell.faces()) {
                                          if (face.neigh().id() == UINT64_MAX || face.neigh().is_ghost()) {
                                              Kokkos::atomic_inc(&patch_counts(patch_id(face.cube_face(), cell.center())));
                                          }
                                      }
                                  });


            std::vector<Boundary_Patch> patches;
            Kokkos::vector<Boundary_Patch, Mem> patches_view;

            uint64_t count_with_ghost = hexcore.cell_count();
            for (uint32_t i = 0; i < patch_counts.size(); i++) {
                uint64_t count = patch_counts(i);

                Boundary_Patch patch = {};
                patch.morton_codes = Kokkos::View<Morton_Code *, Mem>("codes", count);
                patch.faces = Kokkos::View<Cube_Face *, Mem>("mask", count);

                patches_view.push_back(patch);
                patches.push_back(patch);

                patch_base(i) = count_with_ghost;
                patch_counts(i) = 0;
                count_with_ghost += count;
            }

            Kokkos::View<Morton_Code *, Device> morton_codes1("morton codes1", count_with_ghost);
            Kokkos::View<Refinement_Mask *, Device> refinement_mask1("refinement mask", count_with_ghost);

            hexcore.for_each_cell("Refine",
                                  KOKKOS_LAMBDA(Cell &cell) {
                                      uint32_t idx = cell.id();
                                      Morton_Code code = cell.morton_code();
                                      Refinement_Mask mask = cell.refinement_mask();
                                      uint32_t level = mask.level;
                                      for (Face &face: cell.faces()) {
                                          Cell neigh = face.neigh();
                                          if (neigh.id() == UINT64_MAX || neigh.is_ghost()) {
                                              uint32_t patch = patch_id(face.cube_face(), cell.center());
                                              uint64_t offset = Kokkos::atomic_fetch_inc(&patch_counts(patch));
                                              uint64_t base = patch_base(patch);

                                              Morton_Code code = neigh.morton_code();
                                              Refinement_Mask refinement = cell.refinement_mask();
                                              refinement.ghost = true;
                                              refinement.up = 0;
                                              refinement.down = 0;

                                              morton_codes1(base + offset) = code;
                                              refinement_mask1(base + offset) = refinement;

                                              patches_view(patch).morton_codes(offset) = code;
                                              patches_view(patch).faces(offset) = Cube_Face(cube_opposite_faces[face.cube_face()]);
                                          }
                                      }

                                      morton_codes1(idx) = code;
                                      refinement_mask1(idx) = mask;
                                  });

            typename Hexcore::Update_Desc desc;
            desc.count = count_with_ghost;
            desc.morton_codes = morton_codes1;
            desc.refinement_mask = refinement_mask1;
            desc.patches = patches;
            hexcore.update(desc);
        }
    };
}