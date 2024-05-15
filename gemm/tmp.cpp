#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstring>  
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
 * No Blocking
 */
// // void mul(const int &size, vec &a, vec &b, vec &c) {
// double *bT;
// void mul(double* a, double* b, double* c, uint64_t n1, uint64_t n2, uint64_t n3) {
//   // double bT[n2*n3*sizeof(double)] {}; //too large for runtime stack
//   // #pragma omp parallel for schedule(static)
//   register uint64_t pbT = 0;
//   register uint64_t pb = 0;
//   for(int j = 0; j < n2; j++, /*pb+=n3*/ )
//  
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



void mul4x4(double* a, double* bT, double* c, uint64_t n1, uint64_t n2, uint64_t n3, int i, int k) {
  for ( int j = 0; j < n2; ++j ) {
    c[i * n3 + k] += a[i * n2 + j] * bT[k * n2 + j];
    c[i * n3 + k+1] += a[i * n2 + j] * bT[(k+1) * n2 + j];
    c[i * n3 + k+2] += a[i * n2 + j] * bT[(k+2) * n2 + j];
    c[i * n3 + k+3] += a[i * n2 + j] * bT[(k+3) * n2 + j];
    c[(i+1) * n3 + k] += a[(i+1) * n2 + j] * bT[k * n2 + j];
    c[(i+1) * n3 + k+1] += a[(i+1) * n2 + j] * bT[(k+1) * n2 + j];
    c[(i+1) * n3 + k+2] += a[(i+1) * n2 + j] * bT[(k+2) * n2 + j];
    c[(i+1) * n3 + k+3] += a[(i+1) * n2 + j] * bT[(k+3) * n2 + j];
    c[(i+2) * n3 + k] += a[(i+2) * n2 + j] * bT[k * n2 + j];
    c[(i+2) * n3 + k+1] += a[(i+2) * n2 + j] * bT[(k+1) * n2 + j];
    c[(i+2) * n3 + k+2] += a[(i+2) * n2 + j] * bT[(k+2) * n2 + j];
    c[(i+2) * n3 + k+3] += a[(i+2) * n2 + j] * bT[(k+3) * n2 + j];
    c[(i+3) * n3 + k] += a[(i+3) * n2 + j] * bT[k * n2 + j];
    c[(i+3) * n3 + k+1] += a[(i+3) * n2 + j] * bT[(k+1) * n2 + j];
    c[(i+3) * n3 + k+2] += a[(i+3) * n2 + j] * bT[(k+2) * n2 + j];
    c[(i+3) * n3 + k+3] += a[(i+3) * n2 + j] * bT[(k+3) * n2 + j];
  }
}



/**
 * Blocked matrix multiplication
*/
double *bT;
void mul(double* a, double* b, double* c, uint64_t n1, uint64_t n2, uint64_t n3) {
  register uint64_t pbT = 0;
  register uint64_t pb = 0;
  for(int j = 0; j < n2; j++/*, pb+=n3*/)
  {
    pbT = j;
    for(int k = 0; k < n3; k++, pbT+=n2, pb+=1)
      bT[pbT] = b[pb];
      // bT[k*n2+j] = b[j*n3+k];
  }



  int i, j, k;
  int block_size = 4;
  for ( i = 0; i + block_size <= n1; i += block_size ) {
    for ( k = 0; k + block_size <= n3; k += block_size ) {
      mul4x4(a, bT, c, n1, n2, n3, i, k);
    }

    for ( ; k < n3; ++k ) {
      for ( j = 0; j < n2; ++j ) {
        c[i * n3 + k] += a[i * n2 + j] * bT[k * n2 + j];
      }
    }

  }

  for( ; i < n1; ++i ) {
    for ( k = 0; k < n3; ++k ) {
      for ( j = 0; j < n2; ++j ) {
        c[i * n3 + k] += a[i * n2 + j] * bT[k * n2 + j];
      }
    }
  }

}


int main() {
  uint64_t n1, n2, n3;
  FILE* fi;

  fi = fopen("conf.data", "rb");
  
  fread(&n1, 1, 8, fi);
  fread(&n2, 1, 8, fi);
  fread(&n3, 1, 8, fi);

  double* a = (double*)aligned_alloc(64, n1 * n2 * 8);
  double* b = (double*)aligned_alloc(64, n2 * n3 * 8);
  double* c = (double*)aligned_alloc(64, n1 * n3 * 8);
  bT = (double*)aligned_alloc(64, n3 * n2 * 8);
  fread(a, 1, n1 * n2 * 8, fi);
  fread(b, 1, n2 * n3 * 8, fi);
//=======================================
  // #pragma omp parallel 
  // {
  //   #pragma omp sections
  //   {
  //     #pragma omp section
  //     {
  //       fread(a, 1, n1 * n2 * 8, fi);
  //     }
  //     #pragma omp section
  //     {
  //       fread(b, 1, n2 * n3 * 8, fi);
  //     }
  //   }
  // }



  fclose(fi);

  memset(c, 0, n1 * n3 * sizeof(double));

  auto t1 = std::chrono::steady_clock::now();
  mul(a, b, c, n1, n2, n3);
  auto t2 = std::chrono::steady_clock::now();
  int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  printf("%d ms\n", d1);


  fi = fopen("out.data", "wb");
  fwrite(c, 1, n1 * n3 * 8, fi);
  fclose(fi);

  return 0;
}
