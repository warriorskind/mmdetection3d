#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void dynamic_voxelize_kernel(
    const T* points, T_int* coors, const float voxel_x, const float voxel_y,
    const float voxel_z, const float coors_x_min, const float coors_y_min,
    const float coors_z_min, const float coors_x_max, const float coors_y_max,
    const float coors_z_max, const int grid_x, const int grid_y,
    const int grid_z, const int num_points, const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
    } else {
      coors_offset[0] = c_x;
      coors_offset[1] = c_y;
      coors_offset[2] = c_z;
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_point_to_voxel(const int nthreads, const T* points,
                                      T_int* point_to_voxelidx,
                                      T_int* coor_to_voxelidx, T* voxels,
                                      const int max_points,
                                      const int num_features,
                                      const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    int index = thread_idx / num_features;

    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num > -1 && voxelidx > -1) {
      auto voxels_offset =
          voxels + voxelidx * max_points * num_features + num * num_features;

      int k = thread_idx % num_features;
      voxels_offset[k] = points[thread_idx];
    }
  }
}

template <typename T, typename T_int>
__global__ void assign_voxel_coors(const int nthreads, T_int* coor,
                                   T_int* point_to_voxelidx,
                                   T_int* coor_to_voxelidx, T_int* voxel_coors,
                                   const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    // if (index >= num_points) return;
    int index = thread_idx / NDim;
    int num = point_to_voxelidx[index];
    int voxelidx = coor_to_voxelidx[index];
    if (num == 0 && voxelidx > -1) {
      auto coors_offset = voxel_coors + voxelidx * NDim;
      int k = thread_idx % NDim;
      coors_offset[k] = coor[thread_idx];
    }
  }
}

template <typename T_int>
__global__ void point_to_voxelidx_kernel(const T_int* coor,
                                         T_int* point_to_voxelidx,
                                         T_int* point_to_pointidx,
                                         const int max_points,
                                         const int max_voxels,
                                         const int num_points, const int NDim) {
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    // skip invalid points
    if ((index >= num_points) || (coor_offset[0] == -1)) return;

    int num = 0;
    int coor_x = coor_offset[0];
    int coor_y = coor_offset[1];
    int coor_z = coor_offset[2];
    // only calculate the coors before this coor[index]
    for (int i = 0; i < index; ++i) {
      auto prev_coor = coor + i * NDim;
      if (prev_coor[0] == -1) continue;

      // Find all previous points that have the same coors
      // if find the same coor, record it
      if ((prev_coor[0] == coor_x) && (prev_coor[1] == coor_y) &&
          (prev_coor[2] == coor_z)) {
        num++;
        if (num == 1) {
          // point to the same coor that first show up
          point_to_pointidx[index] = i;
        } else if (num >= max_points) {
          // out of boundary
          return;
        }
      }
    }
    if (num == 0) {
      point_to_pointidx[index] = index;
    }
    if (num < max_points) {
      point_to_voxelidx[index] = num;
    }
  }
}

// Build voxel sort key: valid points get packed key, invalid get INT64_MAX
__global__ void build_voxel_key_kernel(const int* coor, int64_t* keys,
                                       const int grid_y, const int grid_z,
                                       const int num_points, const int NDim) {
  const int64_t INVALID_KEY = 0x7FFFFFFFFFFFFFFFL;
  const int64_t grid_yz = static_cast<int64_t>(grid_y) * grid_z;

  CUDA_1D_KERNEL_LOOP(index, num_points) {
    auto coor_offset = coor + index * NDim;
    if (coor_offset[0] == -1) {
      keys[index] = INVALID_KEY;
    } else {
      int64_t c_x = coor_offset[0];
      int64_t c_y = coor_offset[1];
      int64_t c_z = coor_offset[2];
      keys[index] =
          c_x * grid_yz + c_y * static_cast<int64_t>(grid_z) + c_z;
    }
  }
}

// Fill point_to_voxelidx and point_to_pointidx from sorted order.
// Parallel version: each thread handles one point. For each pos, find how many
// preceding points share the same voxel_key (backward scan, O(max_points) per
// thread since same-voxel points typically <= max_points).
// sorted_indices[pos] = original point index with pos-th smallest key.
template <typename T_int>
__global__ void fill_point_to_voxelidx_from_sorted_kernel(
    const int64_t* keys, const int* sorted_indices, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, const int max_points, const int num_points) {
  const int64_t INVALID_KEY = 0x7FFFFFFFFFFFFFFFL;

  CUDA_1D_KERNEL_LOOP(pos, num_points) {
    int orig_idx = sorted_indices[pos];
    int64_t key = keys[pos];

    if (key == INVALID_KEY) continue;

    // Count preceding points with same voxel_key (segment start at segment_start_pos)
    int segment_start_pos = pos;
    while (segment_start_pos > 0 && keys[segment_start_pos - 1] == key) {
      segment_start_pos--;
    }
    int point_pos_in_voxel = pos - segment_start_pos;  // 0 for first in voxel
    int segment_start_orig_idx = sorted_indices[segment_start_pos];

    point_to_pointidx[orig_idx] = segment_start_orig_idx;
    if (point_pos_in_voxel < max_points) {
      point_to_voxelidx[orig_idx] = point_pos_in_voxel;
    }
  }
}

// flag[i]=1 if point i is first in voxel (point_to_voxelidx==0), else 0
__global__ void determin_voxel_flag_kernel(const int* point_to_voxelidx,
                                          int* flag, const int num_points) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    flag[i] = (point_to_voxelidx[i] == 0) ? 1 : 0;
  }
}

// Assign voxel idx to segment starts from exclusive scan result
__global__ void assign_voxel_idx_from_scan_kernel(
    const int* point_to_voxelidx, const int* scan_result,
    int* coor_to_voxelidx, int* voxel_num, const int max_voxels,
    const int num_points) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int last_flag = (num_points > 0 && point_to_voxelidx[num_points - 1] == 0)
                        ? 1
                        : 0;
    int total_segments =
        (num_points > 0) ? (scan_result[num_points - 1] + last_flag) : 0;
    voxel_num[0] = total_segments < max_voxels ? total_segments : max_voxels;
  }
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    if (point_to_voxelidx[i] == 0) {
      int voxelidx = scan_result[i];
      coor_to_voxelidx[i] = (voxelidx < max_voxels) ? voxelidx : -1;
    }
  }
}

// Propagate voxel idx from first point to others, then count points per voxel
__global__ void propagate_and_count_kernel(const int* point_to_voxelidx,
                                           const int* point_to_pointidx,
                                           int* coor_to_voxelidx,
                                           int* num_points_per_voxel,
                                           const int num_points) {
  CUDA_1D_KERNEL_LOOP(i, num_points) {
    if (point_to_voxelidx[i] > 0) {
      coor_to_voxelidx[i] = coor_to_voxelidx[point_to_pointidx[i]];
    }
    int voxelidx = coor_to_voxelidx[i];
    if (voxelidx >= 0) {
      atomicAdd(&num_points_per_voxel[voxelidx], 1);
    }
  }
}

template <typename T_int>
__global__ void determin_voxel_num(
    // const T_int* coor,
    T_int* num_points_per_voxel, T_int* point_to_voxelidx,
    T_int* point_to_pointidx, T_int* coor_to_voxelidx, T_int* voxel_num,
    const int max_points, const int max_voxels, const int num_points) {
  // only calculate the coors before this coor[index]
  for (int i = 0; i < num_points; ++i) {
    // if (coor[i][0] == -1)
    //    continue;
    int point_pos_in_voxel = point_to_voxelidx[i];
    // record voxel
    if (point_pos_in_voxel == -1) {
      // out of max_points or invalid point
      continue;
    } else if (point_pos_in_voxel == 0) {
      // record new voxel
      int voxelidx = voxel_num[0];
      if (voxel_num[0] >= max_voxels) continue;
      voxel_num[0] += 1;
      coor_to_voxelidx[i] = voxelidx;
      num_points_per_voxel[voxelidx] = 1;
    } else {
      int point_idx = point_to_pointidx[i];
      int voxelidx = coor_to_voxelidx[point_idx];
      if (voxelidx != -1) {
        coor_to_voxelidx[i] = voxelidx;
        num_points_per_voxel[voxelidx] += 1;
      }
    }
  }
}

__global__ void nondisterministic_get_assign_pos(
    const int nthreads, const int32_t *coors_map, int32_t *pts_id,
    int32_t *coors_count, int32_t *reduce_count, int32_t *coors_order) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    if (coors_idx > -1) {
      int32_t coors_pts_pos = atomicAdd(&reduce_count[coors_idx], 1);
      pts_id[thread_idx] = coors_pts_pos;
      if (coors_pts_pos == 0) {
        coors_order[coors_idx] = atomicAdd(coors_count, 1);
      }
    }
  }
}

template<typename T>
__global__ void nondisterministic_assign_point_voxel(
    const int nthreads, const T *points, const int32_t *coors_map,
    const int32_t *pts_id, const int32_t *coors_in,
    const int32_t *reduce_count, const int32_t *coors_order,
    T *voxels, int32_t *coors, int32_t *pts_count, const int max_voxels,
    const int max_points, const int num_features, const int NDim) {
  CUDA_1D_KERNEL_LOOP(thread_idx, nthreads) {
    int coors_idx = coors_map[thread_idx];
    int coors_pts_pos = pts_id[thread_idx];
    if (coors_idx > -1) {
      int coors_pos = coors_order[coors_idx];
      if (coors_pos < max_voxels && coors_pts_pos < max_points) {
        auto voxels_offset =
            voxels + (coors_pos * max_points + coors_pts_pos) * num_features;
        auto points_offset = points + thread_idx * num_features;
        for (int k = 0; k < num_features; k++) {
          voxels_offset[k] = points_offset[k];
        }
        if (coors_pts_pos == 0) {
          pts_count[coors_pos] = min(reduce_count[coors_idx], max_points);
          auto coors_offset = coors + coors_pos * NDim;
          auto coors_in_offset = coors_in + coors_idx * NDim;
          for (int k = 0; k < NDim; k++) {
            coors_offset[k] = coors_in_offset[k];
          }
        }
      }
    }
  }
}

namespace voxelization {

int hard_voxelize_gpu(const at::Tensor& points, at::Tensor& voxels,
                      at::Tensor& coors, at::Tensor& num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);

  at::cuda::CUDAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(at::kInt));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
        dynamic_voxelize_kernel<scalar_t, int>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                points.contiguous().data_ptr<scalar_t>(),
                temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y,
                voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
                coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
                num_features, NDim);
      }));
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // 2. map point to idx of corresponding voxel (optimized: sort + O(n) instead
  // of O(n^2))
  auto point_to_pointidx = -at::ones(
      {num_points}, points.options().dtype(at::kInt));
  auto point_to_voxelidx = -at::ones(
      {num_points}, points.options().dtype(at::kInt));

  auto voxel_keys =
      at::empty({num_points}, points.options().dtype(at::kLong));
  auto sorted_indices = at::arange(
      num_points,
      at::TensorOptions().dtype(at::kInt).device(points.device()));

  build_voxel_key_kernel<<<grid, block, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      temp_coors.contiguous().data_ptr<int>(),
      voxel_keys.contiguous().data_ptr<int64_t>(), grid_y, grid_z, num_points,
      NDim);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    thrust::device_ptr<int64_t> keys_ptr(
        voxel_keys.contiguous().data_ptr<int64_t>());
    thrust::device_ptr<int> indices_ptr(
        sorted_indices.contiguous().data_ptr<int>());
    thrust::stable_sort_by_key(thrust::cuda::par.on(stream), keys_ptr,
                               keys_ptr + num_points, indices_ptr);
  }
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  dim3 fill_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 fill_block(512);
  fill_point_to_voxelidx_from_sorted_kernel<int>
      <<<fill_grid, fill_block, 0, at::cuda::getCurrentCUDAStream()>>>(
          voxel_keys.contiguous().data_ptr<int64_t>(),
          sorted_indices.contiguous().data_ptr<int>(),
          point_to_voxelidx.contiguous().data_ptr<int>(),
          point_to_pointidx.contiguous().data_ptr<int>(), max_points,
          num_points);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // 3. Determine voxel num and coor index (4 steps: flag, scan, assign,
  // propagate+count). Use flag+exclusive_scan for Thrust compatibility (no
  // transform_exclusive_scan on older/custom Thrust).
  auto coor_to_voxelidx =
      -at::ones({num_points}, points.options().dtype(at::kInt));
  auto voxel_num = at::zeros({1}, points.options().dtype(at::kInt));
  auto flag = at::empty({num_points}, points.options().dtype(at::kInt));
  auto scan_result = at::empty({num_points}, points.options().dtype(at::kInt));

  dim3 map_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 map_block(512);

  //  确定哪些点是体素首点
  determin_voxel_flag_kernel<<<map_grid, map_block, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      point_to_voxelidx.contiguous().data_ptr<int>(),
      flag.contiguous().data_ptr<int>(), num_points);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // 结果can_result[i] = 在点 i 之前有多少个体素首点 
  {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    thrust::device_ptr<int> flag_ptr(flag.contiguous().data_ptr<int>());
    thrust::device_ptr<int> scan_ptr(scan_result.contiguous().data_ptr<int>());
    thrust::exclusive_scan(thrust::cuda::par.on(stream), flag_ptr,
                          flag_ptr + num_points, scan_ptr);
  }
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // 只给体素首点写上 coor_to_voxelidx，并写出 voxel_num。
  assign_voxel_idx_from_scan_kernel<<<map_grid, map_block, 0,
                                      at::cuda::getCurrentCUDAStream()>>>(
      point_to_voxelidx.contiguous().data_ptr<int>(),
      scan_result.contiguous().data_ptr<int>(),
      coor_to_voxelidx.contiguous().data_ptr<int>(),
      voxel_num.contiguous().data_ptr<int>(), max_voxels, num_points);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // 给非体素首点写上 coor_to_voxelidx，并统计每个体素有多少点。
  propagate_and_count_kernel<<<map_grid, map_block, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      point_to_voxelidx.contiguous().data_ptr<int>(),
      point_to_pointidx.contiguous().data_ptr<int>(),
      coor_to_voxelidx.contiguous().data_ptr<int>(),
      num_points_per_voxel.contiguous().data_ptr<int>(), num_points);
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  // 4. copy point features to voxels
  // Step 4 & 5 could be parallel
  auto pts_output_size = num_points * num_features;
  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(pts_output_size, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_point_to_voxel<float, int>
            <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                pts_output_size, points.contiguous().data_ptr<float>(),
                point_to_voxelidx.contiguous().data_ptr<int>(),
                coor_to_voxelidx.contiguous().data_ptr<int>(),
                voxels.contiguous().data_ptr<float>(), max_points, num_features,
                num_points, NDim);
      }));
  //   cudaDeviceSynchronize();
  //   AT_CUDA_CHECK(cudaGetLastError());

  // 5. copy coors of each voxels
  auto coors_output_size = num_points * NDim;
  dim3 coors_cp_grid(
      std::min(at::cuda::ATenCeilDiv(coors_output_size, 512), 4096));
  dim3 coors_cp_block(512);
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
        assign_voxel_coors<float, int><<<coors_cp_grid, coors_cp_block, 0,
                                         at::cuda::getCurrentCUDAStream()>>>(
            coors_output_size, temp_coors.contiguous().data_ptr<int>(),
            point_to_voxelidx.contiguous().data_ptr<int>(),
            coor_to_voxelidx.contiguous().data_ptr<int>(),
            coors.contiguous().data_ptr<int>(), num_points, NDim);
      }));
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  auto voxel_num_cpu = voxel_num.to(at::kCPU);
  int voxel_num_int = voxel_num_cpu.data_ptr<int>()[0];

  return voxel_num_int;
}

int nondisterministic_hard_voxelize_gpu(
    const at::Tensor &points, at::Tensor &voxels,
    at::Tensor &coors, at::Tensor &num_points_per_voxel,
    const std::vector<float> voxel_size,
    const std::vector<float> coors_range,
    const int max_points, const int max_voxels,
    const int NDim = 3) {

  CHECK_INPUT(points);

  at::cuda::CUDAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  if (num_points == 0)
    return 0;

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // map points to voxel coors
  at::Tensor temp_coors =
      at::zeros({num_points, NDim}, points.options().dtype(torch::kInt32));

  dim3 grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 block(512);

  // 1. link point to corresponding voxel coors
  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "hard_voxelize_kernel", ([&] {
    dynamic_voxelize_kernel<scalar_t, int>
    <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        points.contiguous().data_ptr<scalar_t>(),
        temp_coors.contiguous().data_ptr<int>(), voxel_x, voxel_y,
        voxel_z, coors_x_min, coors_y_min, coors_z_min, coors_x_max,
        coors_y_max, coors_z_max, grid_x, grid_y, grid_z, num_points,
        num_features, NDim);
  }));

  at::Tensor coors_map;
  at::Tensor coors_count;
  at::Tensor coors_order;
  at::Tensor reduce_count;
  at::Tensor pts_id;

  auto coors_clean = temp_coors.masked_fill(temp_coors.lt(0).any(-1, true), -1);

  std::tie(temp_coors, coors_map, reduce_count) =
      at::unique_dim(coors_clean, 0, true, true, false);

  if (temp_coors.index({0, 0}).lt(0).item<bool>()) {
    // the first element of temp_coors is (-1,-1,-1) and should be removed
    temp_coors = temp_coors.slice(0, 1);
    coors_map = coors_map - 1;
  }

  int num_coors = temp_coors.size(0);
  temp_coors = temp_coors.to(torch::kInt32);
  coors_map = coors_map.to(torch::kInt32);

  coors_count = coors_map.new_zeros(1);
  coors_order = coors_map.new_empty(num_coors);
  reduce_count = coors_map.new_zeros(num_coors);
  pts_id = coors_map.new_zeros(num_points);

  dim3 cp_grid(std::min(at::cuda::ATenCeilDiv(num_points, 512), 4096));
  dim3 cp_block(512);
  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "get_assign_pos", ([&] {
    nondisterministic_get_assign_pos<<<cp_grid, cp_block, 0,
    at::cuda::getCurrentCUDAStream()>>>(
        num_points,
        coors_map.contiguous().data_ptr<int32_t>(),
        pts_id.contiguous().data_ptr<int32_t>(),
        coors_count.contiguous().data_ptr<int32_t>(),
        reduce_count.contiguous().data_ptr<int32_t>(),
        coors_order.contiguous().data_ptr<int32_t>());
  }));

  AT_DISPATCH_ALL_TYPES(
      points.scalar_type(), "assign_point_to_voxel", ([&] {
    nondisterministic_assign_point_voxel<scalar_t>
    <<<cp_grid, cp_block, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_points, points.contiguous().data_ptr<scalar_t>(),
        coors_map.contiguous().data_ptr<int32_t>(),
        pts_id.contiguous().data_ptr<int32_t>(),
        temp_coors.contiguous().data_ptr<int32_t>(),
        reduce_count.contiguous().data_ptr<int32_t>(),
        coors_order.contiguous().data_ptr<int32_t>(),
        voxels.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int32_t>(),
        num_points_per_voxel.contiguous().data_ptr<int32_t>(),
        max_voxels, max_points,
        num_features, NDim);
  }));
  AT_CUDA_CHECK(cudaGetLastError());
  return max_voxels < num_coors ? max_voxels : num_coors;
}

void dynamic_voxelize_gpu(const at::Tensor& points, at::Tensor& coors,
                          const std::vector<float> voxel_size,
                          const std::vector<float> coors_range,
                          const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);

  at::cuda::CUDAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  const int col_blocks = at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "dynamic_voxelize_kernel", [&] {
    dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(), voxel_x, voxel_y, voxel_z,
        coors_x_min, coors_y_min, coors_z_min, coors_x_max, coors_y_max,
        coors_z_max, grid_x, grid_y, grid_z, num_points, num_features, NDim);
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  return;
}

}  // namespace voxelization
