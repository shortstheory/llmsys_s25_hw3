#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>

#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

namespace lightseq {
namespace cuda {
/**
@brief: softmax_kernel
Softmax forward kernel for
  enc-self-attn, dec-self-attn, encdec-attn

@thread
gridDim.x = dynamic
gridDim.y = batch_size
gridDim.z = nhead
blockDim.x = from_len

@param
inp: [batch_size, nhead, from_len, to_len], softmax input.
attn_mask: [batch_size, to_len], padding tokens are -inf,
  non padding tokens are 0.
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
// why are from_len and to_len different here? Isn't the input NxN where n is seq_len
template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(T *inp, const T *attn_mask, int from_len,
                                      int to_len, bool mask_future) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  // local to each and every block
  T mval[ele_per_thread];
  if (attn_mask) {
    // think this means that every blockIdx.y gets a different part of the 
    // attention mask to read
    attn_mask += batch_id * to_len;
    // loading a part of the attn mask to mval i think
    // assign REDUCE_FLOAT_INF_NEG to Out of bound vals
    // how do all threads load data in parallel if they all use the same ts_load??
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  // flattens the input to a scalar
  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    // read values into inp_val array
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      // is this an element by element load according to blockidx?
      // so we can load a row of tokens this way, but how does it work if all the threads
      // are loading the same data in parallel here? does each thread have its own copy of
      // the data then?
      // In blockload, each thread is loading num_elems_per_thread values and there are blockDim.x threads which load data at once
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    // Hint: use fmaxf() to compute max
    // BEGIN ASSIGN3_1
    // val[tok][ele] is a buffer for storing the elements under consideration for each thread; later used
    // in the exp and subtracting the max value of each row of tokens
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        float temp_val;
        // seems like we force the future values to be -inf when the mask_future flag is on here
        if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
          temp_val = REDUCE_FLOAT_INF_NEG;
        } else {
          temp_val = (float)inp_val[i][j];
          if (attn_mask) {
            temp_val += (float)mval[j];
          }
        }
        val[i][j] = temp_val;
        l_max[i] = fmaxf(l_max[i], temp_val);
      }
    }
    // END ASSIGN3_1
    // warp reduce max
    // seems like this if finding the max among <= 32 tokens
    warpReduce<ReduceType::kMax, token_per_reduce>(l_max);

    /* step 2. compute sum */
    // thread local sum
    // BEGIN ASSIGN3_1
    // Hint: use __expf() to compute exp
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - l_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // END ASSIGN3_1
    // warp reduce sum
    warpReduce<ReduceType::kSum, token_per_reduce>(l_sum);

    /* step 3. compute final result */
    // BEGIN ASSIGN3_1
    // Hint: use __fdividef() to compute division
    // Hint: use BlockStore to store the result
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      l_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * l_sum[i]);
      }
      // in place softmax so we write results here
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
    // END ASSIGN3_1
  }  // blockIdx.x
}

template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax(T *inp, const T *attn_mask, int from_len,
                                 int to_len, bool mask_future) {
  
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;

  T mval[ele_per_thread];
  if (attn_mask) {
    attn_mask += batch_id * to_len;
    BlockLoad(ts_load).Load(attn_mask, mval, to_len, REDUCE_FLOAT_INF_NEG);
  }

  inp += flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    // BEGIN ASSIGN3_1
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        float temp_val;
        // seems like we force the future values to be -inf when the mask_future flag is on here
        if (mask_future && ele_per_thread * threadIdx.x + j > token_id + i) {
          temp_val = REDUCE_FLOAT_INF_NEG;
        } else {
          temp_val = (float)inp_val[i][j];
          if (attn_mask) {
            temp_val += (float)mval[j];
          }
        }
        val[i][j] = temp_val;
        l_max[i] = fmaxf(l_max[i], temp_val);
      }
    }

    // END ASSIGN3_1
    // block reduce max
    blockReduce<ReduceType::kMax, token_per_reduce>(l_max);
    __syncthreads();
    // write shared
    __shared__ float s_max[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_max[i] = l_max[i];
      }
    }
    __syncthreads();

    /* step 2. compute sum */
    // thread local sum
    // BEGIN ASSIGN3_1
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - s_max[i]);
        l_sum[i] += val[i][j];
      }
    }

    // END ASSIGN3_1
    // block reduce sum
    blockReduce<ReduceType::kSum, token_per_reduce>(l_sum);
    __syncthreads();

    // write shared
    __shared__ float s_sum[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      }
    }
    __syncthreads();

    /* step 3. compute final result */
    // BEGIN ASSIGN3_1
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * s_sum[i]);
      }
      // in place softmax so we write results here
      BlockStore(ts_store).Store(inp + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }

    // END ASSIGN3_1
  }  // blockIdx.x
}

/*
  attn_mask!=nullptr for enc-self-attn and enc-dec-attn
  attn_mask=nullptr and mask_future=ture for dec-self-attn training
  attn_mask=nullptr and mask_future=false for dec-self-attn infer
*/
// template <>
extern "C" {
void launch_attn_softmax(float *inp, const float *attn_mask,
                                int batch_size, int nhead, int from_len,
                                int to_len, bool mask_future,
                                cudaStream_t stream) {

  int float_size = sizeof(float);
  int inp_size = batch_size * nhead * from_len * to_len * float_size;
  int attn_mask_size = batch_size * to_len * float_size;

  float *d_inp, *d_attn_mask;
  cudaMalloc((void **)&d_inp, inp_size);
  cudaMalloc((void **)&d_attn_mask, attn_mask_size);

  cudaMemcpy(d_inp, inp, inp_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_attn_mask, attn_mask, attn_mask_size, cudaMemcpyHostToDevice);

  dim3 grid_dim(1, batch_size, nhead);
  if (to_len <= 32) {
    ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(
        d_inp, d_attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 64) {
    ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(
        d_inp, d_attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 128) {
    grid_dim.x = 16;
    ker_attn_softmax<float, 64, 2><<<grid_dim, 64, 0, stream>>>(
        d_inp, d_attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 256) {
    grid_dim.x = 32;
    ker_attn_softmax<float, 128, 2><<<grid_dim, 128, 0, stream>>>(
        d_inp, d_attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 512) {
    grid_dim.x = 64;
    ker_attn_softmax<float, 256, 2><<<grid_dim, 256, 0, stream>>>(
        d_inp, d_attn_mask, from_len, to_len, mask_future);
  } else if (to_len <= 1024) {
    grid_dim.x = 128;
    ker_attn_softmax<float, 512, 2><<<grid_dim, 512, 0, stream>>>(
        d_inp, d_attn_mask, from_len, to_len, mask_future);
  } else {
    throw std::runtime_error(
        "Sequence length greater than 512 is currently not supported");
  }

  // Copy back to the host
  cudaMemcpy(inp, d_inp, inp_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_attn_softmax Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_inp);
  cudaFree(d_attn_mask);

}}


/**
@brief: ker_attn_softmax_bw
Softmax backward in self attention.

@thread
gridDim.x = batch_size * nhead * seq_len / warps_per_block
blockDim.x = WARP_SIZE
blockDim.y = warps_per_block

@param
grad: [batch_size, nhead, seq_len, seq_len], output grad.
output: [batch_size, nhead, seq_len, seq_len], output of softmax forward.
*/
template <typename T, int ITERATIONS>
__global__ void ker_attn_softmax_bw(T *grad, const T *inp, int softmax_length) {
  // nothing to see here, just basically reshaping the matrix
  // from BxHxNxN to BxHxN
  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  // stride over rows so each thread gets a different element of the sequence
  int offset = batch_idx * softmax_length + threadIdx.x;

  // seems like a sneaky way of changing the starting position of each thread
  grad += offset;
  inp += offset;

  // if softmax_len > thread size, then we would like each thread to do softmax_len/WARP_SIZE
  // elements instead
  T grad_reg[ITERATIONS];
  T inp_reg[ITERATIONS];
  float sum = 0.0;

  #pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    // mask out threads which have an index > softmax_length
    if (curr_idx < softmax_length) {
      // maybe possible to still use blockload over here?
      // actually no it wont work as blockload does strided accesses and not
      // with a gap of 32
      // here it's needed because we're breaking down the softmax_len into groups of 32
      grad_reg[i] = grad[i * WARP_SIZE];
      inp_reg[i] = inp[i * WARP_SIZE];
      sum += (float)grad_reg[i] * (float)inp_reg[i];
    }
  }

  // is this a more general form of warpReduce? Warp reduce seems to require 
  // explicit templates for each different size, i wonder why it's not unrolled
  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);
  #pragma unroll
  for (int i = 1; i < WARP_SIZE; i <<= 1) {
    // this uses shfl_xor instead of _shfl xor sync but it seems to be some thread block sort of thing going on here
    // sum is 0.0 for threads which are out of softmax_len
    sum += g.shfl_xor(sum, i);
  }

  #pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length)
    // basically kronecker delta going on here
      grad[i * WARP_SIZE] = (T)((float)inp_reg[i] * ((float)grad_reg[i] - sum));
  }
}

// template <typename T>
extern "C" {
void launch_attn_softmax_bw(float *out_grad,
                                const float *soft_inp, int rows,
                                int softmax_len,
                                cudaStream_t stream) {
  
  const int warps_per_block = 4;
  dim3 grid_dim((rows + warps_per_block - 1) / warps_per_block);
  dim3 block_dim(WARP_SIZE, warps_per_block);
  // BEGIN ASSIGN3_1
  float input_size = rows*softmax_len*sizeof(float);
  float *out_grad_gpu, *soft_inp_gpu;
  cudaMalloc(&out_grad_gpu, input_size);
  cudaMalloc(&soft_inp_gpu, input_size);

  cudaMemcpy(out_grad_gpu, out_grad, input_size,  cudaMemcpyHostToDevice);
  cudaMemcpy(soft_inp_gpu, soft_inp, input_size,  cudaMemcpyHostToDevice);
  
  // Launch kernel
  // Hint: use ker_attn_softmax_bw<float, ITERATIONS> depending on softmax_len
  if (softmax_len <= 32) {
    ker_attn_softmax_bw<float,  1><<<grid_dim, block_dim, 0, stream>>>(
        out_grad_gpu, soft_inp_gpu, softmax_len);
  } else if (softmax_len <= 64) {
    ker_attn_softmax_bw<float,  2><<<grid_dim, block_dim, 0, stream>>>(
      out_grad_gpu, soft_inp_gpu, softmax_len);
  } else if (softmax_len <= 128) {
    ker_attn_softmax_bw<float,  4><<<grid_dim, block_dim, 0, stream>>>(
      out_grad_gpu, soft_inp_gpu, softmax_len);
  } else if (softmax_len <= 256) {
    ker_attn_softmax_bw<float,  8><<<grid_dim, block_dim, 0, stream>>>(
      out_grad_gpu, soft_inp_gpu, softmax_len);
  } else if (softmax_len <= 512) {
    ker_attn_softmax_bw<float,  16><<<grid_dim, block_dim, 0, stream>>>(
      out_grad_gpu, soft_inp_gpu, softmax_len);
  } else if (softmax_len <= 1024) {
    ker_attn_softmax_bw<float,  32><<<grid_dim, block_dim, 0, stream>>>(
      out_grad_gpu, soft_inp_gpu, softmax_len);
  }else if (softmax_len <= 2048) {
    ker_attn_softmax_bw<float,  64><<<grid_dim, block_dim, 0, stream>>>(
      out_grad_gpu, soft_inp_gpu, softmax_len);
  } else {
    throw std::runtime_error(
        "Sequence length greater than 2048 is currently not supported");
  }

  // Copy back to the host
  cudaMemcpy(out_grad, out_grad_gpu, input_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_attn_softmax Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(out_grad_gpu);
  cudaFree(soft_inp_gpu);

  // Free memory on device
  // END ASSIGN3_1

}}

}  
} 
