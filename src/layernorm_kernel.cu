#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  float l_sum = 0.0;
  float l2_sum = 0.0;
  int offset = blockIdx.x * hidden_size; 
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + offset;  
  float4 *ln_resf4= reinterpret_cast< float4 *>(ln_res) + offset;  

  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    // float4 == 4 x float32 not fp4!
    l_sum += val.x + val.y + val.z + val.w;
    l2_sum += val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w;
  }
  // printf("threadIdx.x %d l_sum %f l2_sum %f\n", threadIdx.x, l_sum, l2_sum);
  // each thread loading 4 values here, so should it be hidden_size/4?
  // Step 2
  blockReduce<ReduceType::kSum, 1>(&l_sum);
  __syncthreads();

  blockReduce<ReduceType::kSum, 1>(&l2_sum);
  __syncthreads();
  __shared__ float s_var;
  __shared__ float s_mean;
  if (threadIdx.x == 0)
  {
    int real_hidden_dim = 4*hidden_size;
    s_mean = l_sum / real_hidden_dim;
    float l2_sum_mean = l2_sum / real_hidden_dim;
    s_var = l2_sum_mean - s_mean * s_mean + LN_EPSILON;
    // printf("lsum - %f l2sum - %f Svar - %f Smean %f l2summean\n", l_sum, l2_sum, s_var, s_mean, l2_sum_mean);

    if (means != nullptr)
    {
      means[blockIdx.x] = s_mean;
    }
    if (vars != nullptr)
    {
      vars[blockIdx.x] = s_var;
    }
    s_var = sqrt(s_var);
  }
  __syncthreads();

  // Step 3
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 bias_ =  *(reinterpret_cast<const float4 *>(bias)+idx);
    float4 scale_ = *(reinterpret_cast<const float4 *>(scale)+idx);
    float4 input_ = inp_f4[idx];
    float4 res;
    res.x = scale_.x*(input_.x - s_mean)/s_var + bias_.x;
    res.y = scale_.y*(input_.y - s_mean)/s_var + bias_.y;
    res.z = scale_.z*(input_.z - s_mean)/s_var + bias_.z;
    res.w = scale_.w*(input_.w - s_mean)/s_var + bias_.w;


    ln_resf4[idx] = res;
  }

  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  // __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ T gamma_buffer[TILE_DIM][TILE_DIM];
  __shared__ T betta_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
  
  int col = blockDim.x * blockIdx.x + threadIdx.x;

  T thread_grad_sum=0.0;
  T thread_grad_xhat_product_sum = 0.0;
  const bool use_gamma_beta = gamma&&betta;
  const bool use_mean_var = means && vars;

  if (col < width)
  {
    for (int i = threadIdx.y; i < rows; i+=blockDim.y)
    {
      int row = i;
      int idx = row*width + col;  
      T grad = out_grad[idx];
      T xhat = 0.0;
      const T input_or_output = inp[idx];
      if (use_gamma_beta)
      {
        const T gamma_val = gamma[col];
        const T beta_val = betta[col];
        xhat = (input_or_output-beta_val)/gamma_val;
        // printf("betta threadIdx.x %d threadIdx.y %d grad %f xhat %f\n", threadIdx.x, threadIdx.y, grad, xhat);
      } else if (use_mean_var)
      {
        const T mean = means[row];
        const T var = vars[row];
        xhat = (input_or_output-mean)/var;
      }
      thread_grad_sum += grad;
      thread_grad_xhat_product_sum += grad*xhat;
    }
  }
  // if (threadIdx.y < rows && col < width && blockIdx.x == 0)
  //   printf("%d %d %f\n", threadIdx.x, threadIdx.y, thread_grad_sum);

  betta_buffer[threadIdx.x][threadIdx.y] = thread_grad_sum;
  gamma_buffer[threadIdx.x][threadIdx.y] = thread_grad_xhat_product_sum;
  __syncthreads();
  int lane_id = g.thread_rank();
  T grad_sum = betta_buffer[threadIdx.y][threadIdx.x];
  T grad_xhat_product_sum = gamma_buffer[threadIdx.y][threadIdx.x];

  for (int offset = g.size()/2; offset > 0; offset/=2)
  {
    grad_sum += g.shfl_down(grad_sum, offset);
    grad_xhat_product_sum += g.shfl_down(grad_xhat_product_sum, offset);
  }
  int dest = blockDim.x * blockIdx.x + threadIdx.y;
  if (lane_id == 0 && dest < width)
  {
    betta_grad[dest] = grad_sum;
    gamma_grad[dest] = grad_xhat_product_sum;
    // printf("betta_grad %f gamma_grad %f dest %d\n", betta_grad[dest], gamma_grad[dest], dest);
  }

}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*w with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  
  // Step 1
  // Step 1
  int offset = blockIdx.x * hidden_dim;
  const int row = blockIdx.x;
  const int col = threadIdx.x;

  const float4* inp4_ptr =      reinterpret_cast<const float4*>(inp)+offset;
  const float4* gamma4_ptr =    reinterpret_cast<const float4*>(gamma);
  const float4* betta4_ptr =    reinterpret_cast<const float4*>(betta);
  const float4* out_grad4_ptr = reinterpret_cast<const float4*>(out_grad)+offset;
  float4* inp_grad4_ptr =       reinterpret_cast<float4*>(inp_grad)+offset;
  

  float4 xhat;
  float4 input_or_output4;
  float4 gamma4;
  float4 dxhat4;

  xhat.x = 0.0;
  xhat.y = 0.0;
  xhat.z = 0.0;
  xhat.w = 0.0;
  input_or_output4.x = 0.0;
  input_or_output4.y = 0.0;
  input_or_output4.z = 0.0;
  input_or_output4.w = 0.0;
  gamma4.x = 0.0;
  gamma4.y = 0.0;
  gamma4.z = 0.0;
  gamma4.w = 0.0;
  dxhat4.x = 0.0;
  dxhat4.y = 0.0;
  dxhat4.z = 0.0;
  dxhat4.w = 0.0;


  if (col < hidden_dim)
  {
    float4 input_or_output4 = inp4_ptr[col];
    float4 gamma4 = gamma4_ptr[col];
    if (betta4_ptr)
    {
      float4 betta4 = betta4_ptr[col];
      xhat.x = (input_or_output4.x - betta4.x)/gamma4.x;
      xhat.y = (input_or_output4.y - betta4.y)/gamma4.y;
      xhat.z = (input_or_output4.z - betta4.z)/gamma4.z;
      xhat.w = (input_or_output4.w - betta4.w)/gamma4.w;
      // printf("betta threadIdx.x %d %f %f %f %f\n", threadIdx.x, xhat.x, xhat.y, xhat.z, xhat.w);
    } else if (means)
    {
      float mean = means[row];
      float var_rsqrt = rsqrt(vars[row]);
      xhat.x = (input_or_output4.x - mean)*var_rsqrt;
      xhat.y = (input_or_output4.y - mean)*var_rsqrt;
      xhat.z = (input_or_output4.z - mean)*var_rsqrt;
      xhat.w = (input_or_output4.w - mean)*var_rsqrt;
    }
    float4 out_grad4 = out_grad4_ptr[col];

    dxhat4.x = out_grad4.x*gamma4.x;
    dxhat4.y = out_grad4.y*gamma4.y;
    dxhat4.z = out_grad4.z*gamma4.z;
    dxhat4.w = out_grad4.w*gamma4.w;
  }


  float dxhat_sum = dxhat4.x + dxhat4.y + dxhat4.z + dxhat4.w;
  float dxhat_xhat_sum = dxhat4.x*xhat.x + dxhat4.y*xhat.y + dxhat4.z*xhat.z + dxhat4.w*xhat.w;

  // Step 2
  blockReduce<ReduceType::kSum,1>(&dxhat_sum);
  blockReduce<ReduceType::kSum,1>(&dxhat_xhat_sum);

  __shared__ float sdxhat_sum;
  __shared__ float sdxhat_xhat_sum;
  __shared__ float svariance_inv;
  if (threadIdx.x == 0)
  {
    sdxhat_sum = dxhat_sum;
    sdxhat_xhat_sum = dxhat_xhat_sum;
    svariance_inv = rsqrt(vars[row]);
  }
  __syncthreads();
  float4 res;
  const float real_hidden_dim = hidden_dim*4;
  res.x = (dxhat4.x - (sdxhat_sum + xhat.x*sdxhat_xhat_sum)/real_hidden_dim)*svariance_inv;
  res.y = (dxhat4.y - (sdxhat_sum + xhat.y*sdxhat_xhat_sum)/real_hidden_dim)*svariance_inv;
  res.z = (dxhat4.z - (sdxhat_sum + xhat.z*sdxhat_xhat_sum)/real_hidden_dim)*svariance_inv;
  res.w = (dxhat4.w - (sdxhat_sum + xhat.w*sdxhat_xhat_sum)/real_hidden_dim)*svariance_inv;
  if (col < hidden_dim)
  {
    inp_grad4_ptr[col]=res;
  }
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
