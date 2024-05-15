#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>

// #define DBG
__device__
uint8_t B6S567(uint8_t now, uint8_t around) {
  if(now == 0) {
    if (around == 6) 
      return 1;
    else
      return 0;
  }
  else if(now == 1) {
    if(around == 5 || around == 6 || around ==7)
      return 1;
    else 
      return 0;
  }
}

__device__ 
void coord_mod(int M, int &x, int &y, int &z) {
  x += M;
  y += M;
  z += M;
  x %= M;
  y %= M;
  z %= M;
}

__device__ 
int coord_map(int M, int x, int y, int z) {
  coord_mod(M, x, y, z);
  return M * M * x + M * y + z;
}

__global__ 
void gameKernel(uint8_t* dst, const uint8_t* src, int M, int* dirx, int* diry, int* dirz) {
  int x = blockIdx.y;
  int y = blockIdx.z;
  int z = threadIdx.x;
  int tx, ty, tz;
  int cur_dst = coord_map(M, x, y, z);
  dst[cur_dst] = 0;
  for(int i=0; i<26; ++i) {
    tx = x + dirx[i];
    ty = y + diry[i];
    tz = z + dirz[i];
    int cur_src = coord_map(M, tx, ty, tz);
    dst[cur_dst] += src[cur_src];
  }
  dst[cur_dst] = B6S567(src[cur_dst], dst[cur_dst]);
}

int main(int argc, char *argv[]) {
  if(argc != 4) {
    std::cout << "Usage:" << argv[0] << "<input file> <output file> <number of iterations>";
  }
  std::ifstream input_file(argv[1], std::ios::binary);
  int64_t N = atoll(argv[3]), M, T;
  input_file.read(reinterpret_cast<char *>(&M), 8);
  input_file.read(reinterpret_cast<char *>(&T), 8);
  int space_size = M * M * M * (sizeof(uint8_t));
  uint8_t * buffer = (uint8_t *) malloc(2 * space_size);
  memset(buffer, 0, space_size*2);
  uint8_t * d_buffer;
  cudaMalloc(reinterpret_cast<void**>(&d_buffer), 2 * space_size);
  input_file.read(reinterpret_cast<char*>(buffer), space_size);
  cudaMemcpy(d_buffer, buffer, 2 * space_size, cudaMemcpyHostToDevice);
  int dirx[26], diry[26], dirz[26], cur = -1, *d_dirx, *d_diry, *d_dirz;
  for(int dx = -1; dx <= 1; dx++) {
    for(int dy = -1; dy <= 1; dy++) {
      for(int dz = -1; dz <= 1; dz++) {
        if(dx|dy|dz) ++cur; else continue;
        dirx[cur] = dx;
        diry[cur] = dy;
        dirz[cur] = dz;
      }
    } 
  }

  cudaMalloc((void **)&d_dirx, 26 * sizeof(int));
  cudaMalloc((void **)&d_diry, 26 * sizeof(int));
  cudaMalloc((void **)&d_dirz, 26 * sizeof(int));
  cudaMemcpy(d_dirx, dirx, 26 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_diry, diry, 26 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dirz, dirz, 26 * sizeof(int), cudaMemcpyHostToDevice);
  for(int i=0; i<N; ++i){
    if(i % 2 == 0)
      gameKernel<<<dim3(1,M,M),dim3(M,1,1)>>>(d_buffer+space_size, d_buffer, M, d_dirx, d_diry, d_dirz);
    else
      gameKernel<<<dim3(1,M,M),dim3(M,1,1)>>>(d_buffer, d_buffer+space_size, M, d_dirx, d_diry, d_dirz);
    cudaDeviceSynchronize();
  }
  cudaMemcpy(buffer, d_buffer, 2 * space_size, cudaMemcpyDeviceToHost);
  std::ofstream output_file(argv[2], std::ios::binary);
  output_file.write(reinterpret_cast<const char *>(&M), 8);
  output_file.write(reinterpret_cast<const char *>(&N), 8);
  if(N % 2) 
    output_file.write((const char*)buffer, space_size);
  else
    output_file.write((const char*)(buffer+space_size), space_size);

  free(buffer);
  cudaFree(d_buffer);
  cudaFree(d_dirx);
  cudaFree(d_diry);
  cudaFree(d_dirz);
  input_file.close();
  output_file.close(); 
  }