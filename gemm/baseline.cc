#include <iostream>
#include <chrono>
#include <omp.h>

void mul(float * __restrict a,  float * __restrict b, float * __restrict c, uint64_t n1, uint64_t n2, uint64_t n3) {

#pragma acc data copyin(a[0:n1*n2], b[0:n2*n3], c[0:n1*n3])
#pragma acc parallel

#pragma omp target map(to:a[0:n1*n2], b[0:n2*n3]) map(tofrom:c[0:n1*n3])

 #pragma omp parallel loop collapse(3)
 for (int i = 0; i < n1; i++) {
  #pragma acc loop
  for (int j = 0; j < n2; j++) {
    #pragma acc loop
   for (int k = 0; k < n3; k++) {
    c[i * n3 + k] += a[i * n2 + j] * b[j * n3 + k];
   }
  }
 }
#pragma acc data copyout(a[n1*n2], b[n2*n3], c[n1*n3])

}

int main() {
 uint64_t n1, n2, n3;
 FILE* fi;

 fi = fopen("conf.data", "rb");
 const size_t size_float = sizeof(float);
 fread(&n1, 1, sizeof(uint64_t), fi);
 fread(&n2, 1, sizeof(uint64_t), fi);
 fread(&n3, 1, sizeof(uint64_t), fi);

 float* a = (float*)malloc(n1 * n2 * size_float);
 float* b = (float*)malloc(n2 * n3 * size_float);
 float* c = (float*)malloc(n1 * n3 * size_float);

 fread(a, 1, n1 * n2 * size_float, fi);
 fread(b, 1, n2 * n3 * size_float, fi);
 fclose(fi);

 for (uint64_t i = 0; i < n1; i++) {
  for (uint64_t k = 0; k < n3; k++) {
   c[i * n3 + k] = 0;
  }
 }

 auto t1 = std::chrono::steady_clock::now();
 mul(a, b, c, n1, n2, n3);
 auto t2 = std::chrono::steady_clock::now();
 int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
 printf("%d\n", d1);


 fi = fopen("stdans.data", "wb");
 fwrite(c, 1, n1 * n3 * size_float, fi);
 fclose(fi);

 return 0;
}
