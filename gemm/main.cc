#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <filesystem>
#include <cstring>  
#include <omp.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
// void mul(double* a, double* b, double* c, uint64_t n1, uint64_t n2, uint64_t n3) {
//  for (int i = 0; i < n1; i++) {
//   for (int j = 0; j < n2; j++) {
//  for (int k = 0; k < n3; k++) {
//   c[i * n3 + k] += a[i * n2 + j] * b[j * n3 + k];
//  }
//   }
//  }
// }

/**
 * 
 * No Blocked
 */
// void mul(const int &size, vec &a, vec &b, vec &c) {
// double *bT;
// void mul(double* a, double* b, double* c, uint64_t n1, uint64_t n2, uint64_t n3) {
//   // double bT[n2*n3*sizeof(double)] {}; //too large for runtime stack
//   // #pragma omp parallel for schedule(static)
//   register uint64_t pbT = 0;
//   register uint64_t pb = 0;
//   for(int j = 0; j < n2; j++ /*,pb+=n3*/ )
 
//   {
//     pbT = j;
//     for(int k = 0; k < n3; k++, pbT+=n2, pb+=1)
//       bT[pbT] = b[pb];
//       // bT[k*n2+j] = b[j*n3+k];
//   }
//   register uint64_t pa,pc;
//   #pragma omp parallel for schedule(static) private(pa,pbT,pc) proc_bind(close)
//   for(int i = 0; i < n1; i++) {
//     pc = i * n3;
//     for(int k = 0; k < n3; k++, pc++) {
//       pa = i * n2;
//       pbT = k * n2;
//       __m512d _a[4],_bT[4],_sum[4];
//       _sum[0] = _mm512_setzero_pd();
//       _sum[1] = _mm512_setzero_pd();
//       _sum[2] = _mm512_setzero_pd();
//       _sum[3] = _mm512_setzero_pd();
//       static const int inc = 32;
//       for(int j = 0; j + inc <= n2; j += inc, pa+=inc, pbT+=inc)
//       { // debug: pa++,pbT++ -> loadu -> segfault
//         // _mm_prefetch(&a[i*n2+j+16], _MM_HINT_T0);
//         // _mm_prefetch(&b[k*n2+j+16], _MM_HINT_T0);
//         // _a[0] = _mm512_load_pd(&a[i*n2+j]);
//         // _bT[0] = _mm512_load_pd(&bT[k*n2+j]);
//         // // res = _mm512_mul_pd(_a,_bT);
//         // _sum[0] = _mm512_fmadd_pd(_a[0], _bT[0], _sum[0]);
// //===============OPTIMIZATION=================
//         _mm_prefetch(&a[pa+32], _MM_HINT_T0);
//         _mm_prefetch(&bT[pbT+32], _MM_HINT_T0);
//         _mm_prefetch(&a[pa+40], _MM_HINT_T0);
//         _mm_prefetch(&bT[pbT+40], _MM_HINT_T0);
//         _mm_prefetch(&a[pa+48], _MM_HINT_T0);
//         _mm_prefetch(&bT[pbT+48], _MM_HINT_T0);
//         _mm_prefetch(&a[pa+56], _MM_HINT_T0);
//         _mm_prefetch(&bT[pbT+56], _MM_HINT_T0);
//         _a[0] = _mm512_load_pd(&a[pa]);
//         _a[1] = _mm512_load_pd(&a[pa+8]);
//         _a[2] = _mm512_load_pd(&a[pa+16]);
//         _a[3] = _mm512_load_pd(&a[pa+24]);
//         _bT[0] = _mm512_load_pd(&bT[pbT]);
//         _bT[1] = _mm512_load_pd(&bT[pbT+8]);
//         _sum[0] = _mm512_fmadd_pd(_a[0],_bT[0],_sum[0]);
//         _sum[1] = _mm512_fmadd_pd(_a[1],_bT[1],_sum[1]);
//         _bT[2] = _mm512_load_pd(&bT[pbT+16]);
//         _bT[3] = _mm512_load_pd(&bT[pbT+24]);
//         _sum[2] = _mm512_fmadd_pd(_a[2],_bT[2],_sum[2]);
//         _sum[3] = _mm512_fmadd_pd(_a[3],_bT[3],_sum[3]);
//       }
//       // c[i*n3+k] = _mm512_reduce_add_pd(_sum[0]);
// //================================================
//       c[pc] = _mm512_reduce_add_pd(_sum[0]);
//       c[pc] += _mm512_reduce_add_pd(_sum[1]);
//       c[pc] += _mm512_reduce_add_pd(_sum[2]);
//       c[pc] += _mm512_reduce_add_pd(_sum[3]);

//     }
//   }
// }

constexpr int block_size = 4; // must multiple of 2
constexpr int block_size2 = block_size * block_size; 

inline
void mul4x4(const double* a, const double* bT, double* c, int64_t n1, int64_t n2, int64_t n3, int i, int k) {
  int vector_len = 8;
  //=====================================
  __m512d vc[block_size2], va[block_size], vbT[block_size];
  for (int l = 0; l < block_size2; ++ l) {
    vc[l] = _mm512_setzero_pd();
  }
  //=====================================
  for ( int j = 0; j + vector_len <= n2; j += vector_len ) {
    //============================================================
    // c[i * n3 + k] += a[i * n2 + j] * bT[k * n2 + j];
    // c[i * n3 + k+1] += a[i * n2 + j] * bT[(k+1) * n2 + j];
    // c[i * n3 + k+2] += a[i * n2 + j] * bT[(k+2) * n2 + j];
    // c[i * n3 + k+3] += a[i * n2 + j] * bT[(k+3) * n2 + j];
    // c[(i+1) * n3 + k] += a[(i+1) * n2 + j] * bT[k * n2 + j];
    // c[(i+1) * n3 + k+1] += a[(i+1) * n2 + j] * bT[(k+1) * n2 + j];
    // c[(i+1) * n3 + k+2] += a[(i+1) * n2 + j] * bT[(k+2) * n2 + j];
    // c[(i+1) * n3 + k+3] += a[(i+1) * n2 + j] * bT[(k+3) * n2 + j];
    // c[(i+2) * n3 + k] += a[(i+2) * n2 + j] * bT[k * n2 + j];
    // c[(i+2) * n3 + k+1] += a[(i+2) * n2 + j] * bT[(k+1) * n2 + j];
    // c[(i+2) * n3 + k+2] += a[(i+2) * n2 + j] * bT[(k+2) * n2 + j];
    // c[(i+2) * n3 + k+3] += a[(i+2) * n2 + j] * bT[(k+3) * n2 + j];
    // c[(i+3) * n3 + k] += a[(i+3) * n2 + j] * bT[k * n2 + j];
    // c[(i+3) * n3 + k+1] += a[(i+3) * n2 + j] * bT[(k+1) * n2 + j];
    // c[(i+3) * n3 + k+2] += a[(i+3) * n2 + j] * bT[(k+2) * n2 + j];
    // c[(i+3) * n3 + k+3] += a[(i+3) * n2 + j] * bT[(k+3) * n2 + j];
    //==============================================================
    int k_n2_j = k * n2 + j;
    int i_n2_j = i * n2 + j;
    double * p;
    p = (double*)bT+k_n2_j;
    for(int kk=0; kk < block_size; ++kk, p+=n2) {
      _mm_prefetch(p+sizeof(double)*vector_len, _MM_HINT_T1);
      vbT[kk] = _mm512_loadu_pd(p);
    }
    p = (double*)a+i_n2_j;
    for(int ii=0; ii < block_size; ++ii, p+=n2){ 
      _mm_prefetch(p+sizeof(double)*vector_len, _MM_HINT_T1);
      va[ii] = _mm512_loadu_pd(p);
    }
    int cur = 0;
    for(int ii=0; ii < block_size; ++ii) {
      for(int kk=0; kk < block_size; ++kk) {
        vc[cur] = _mm512_fmadd_pd(va[ii], vbT[kk], vc[cur]);
        ++cur;
        //====================================================================
        // c[(i+ii)*n3+(k+kk)] += _mm512_reduce_add_pd(_mm512_mul_pd(va[ii], vbT[kk]));
        // ++cur;
      }
    }
  }
  // ===================================================
  int cur = 0;
  int i_n3_k = i * n3 + k;
  int ii_n3 = 0;
  for(int ii=0; ii < block_size; ++ii, ii_n3+=n3) {
    for(int kk=0; kk < block_size; ++kk) {
      c[ii_n3 + kk + i_n3_k] = _mm512_reduce_add_pd(vc[cur]);
      ++cur;
    }
  }
}



/**
 * Blocked matrix multiplication
*/
double *bT;
void mul(double* a, double* b, double* c, int64_t n1, int64_t n2, int64_t n3) {
  uint64_t pbT = 0;
  uint64_t pb = 0;
  for(int j = 0; j < n2; j++/*, pb+=n3*/)
  {
    pbT = j;
    for(int k = 0; k < n3; k++, pbT+=n2, pb+=1)
      bT[pbT] = b[pb];
      // bT[k*n2+j] = b[j*n3+k];
  }



  int i;
  int num_blocks = n1 / block_size;
  /**
   * OpenMP 可能会引入浮点误差，但OJ可以过
   * 更正：OpenMP不会引入浮点误差，浮点误差的来源是向量指令的reduce引入了加法结合律 
  */ 
  #pragma omp parallel for schedule(static) proc_bind(close)
  for ( i = 0; i < num_blocks; i += 1 ) {
    int i_block_size = i * block_size;
    for ( int k = 0; k + block_size <= n3; k += block_size ) {
      mul4x4(a, bT, c, n1, n2, n3, i_block_size, k);
    }

    // for ( ; k < n3; ++k ) {
    //   for ( j = 0; j < n2; ++j ) {
    //     c[i * n3 + k] += a[i * n2 + j] * bT[k * n2 + j];
    //   }
    // }

  }

  // for( ; i < n1; ++i ) {
  //   for ( k = 0; k < n3; ++k ) {
  //     for ( j = 0; j < n2; ++j ) {
  //       c[i * n3 + k] += a[i * n2 + j] * bT[k * n2 + j];
  //     }
  //   }
  // }

}

// namespace fs = std::filesystem;

int main() {
  int64_t n1, n2, n3;
  FILE* fi;
  auto t0 = std::chrono::steady_clock::now();

  fi = fopen("conf.data", "rb");
  fread(&n1, 1, 8, fi);
  fread(&n2, 1, 8, fi);
  fread(&n3, 1, 8, fi);
  //===========================================
  // auto file_size = fs::file_size(fs::path("conf.data"));
  // int fd = open("conf.data", O_RDONLY);
  // void * addr = mmap(NULL, 8*(n1*n2+n2*n3)+24, PROT_READ, MAP_PRIVATE, fd, 0);
  // n1 = * (int64_t *) addr;
  // n2 = * (int64_t *) ((char*)addr+8);
  // n3 = * (int64_t *) ((char*)addr+16);
  //===========================================

  double* a = (double*)aligned_alloc(64, n1 * n2 * 8);
  double* b = (double*)aligned_alloc(64, n2 * n3 * 8);
  double* c = (double*)aligned_alloc(64, n1 * n3 * 8);
  bT = (double*)aligned_alloc(64, n3 * n2 * 8);

//=======================================
  // memcpy(a, ((char*)addr) + 24, n1 * n2 * 8);
  // memcpy(b, ((char*)addr) + 24 + n1 * n2 * 8, n2 * n3 * 8);
  // munmap(addr, 8*(n1*n2+n2*n3)+24);
  // close(fd);
//=======================================
    fread(a, 1, n1 * n2 * 8, fi);
    fread(b, 1, n2 * n3 * 8, fi);

//=======================================
  // #pragma omp parallel sections
  // {
  //   #pragma omp section
  //   fread(a, 1, n1 * n2 * 8, fi);

  //   #pragma omp section
  //   fread(b, 1, n2 * n3 * 8, fi);
  // }
//=======================================
  fclose(fi);


  memset(c, 0, n1 * n3 * sizeof(double));

  auto t1 = std::chrono::steady_clock::now();
  mul(a, b, c, n1, n2, n3);
  auto t2 = std::chrono::steady_clock::now();
  int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  printf("%d ms\n", d1);

  //=============================================================
  fi = fopen("out.data", "wb");
  fwrite(c, 1, n1 * n3 * 8, fi);
  fclose(fi);
  //=============================================================
  // fd = open("out.data", O_WRONLY);
  // addr = mmap(NULL, n1 * n3 * 8, PROT_WRITE, MAP_PRIVATE, fd, 0);
  // memcpy(addr, c, n1 * n3 * 8);
  // msync(addr, n1 * n3 * 8, MS_SYNC);
  // munmap(addr, file_size);
  // close(fd);


  auto t3 = std::chrono::steady_clock::now();
  int d2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t0).count();
  printf("%d ms\n", d2);
  return 0;
}
