#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
inline
double it(double r, double x, int64_t itn) {
    int rx, m1x; 
	for (int64_t i = 0; i < itn; i++) {
        rx = r * x;
		m1x = 1.0 - x;
		x = rx * m1x;
    }
    return x;
}
inline
void itv(double r, double* x, int64_t n, int64_t itn) {
	#pragma acc parallel loop copy(x[0:n])
    for (int64_t i = 0; i < n; i++) {
        x[i] = it(r, x[i], itn);
    }
}

int main(){
    FILE* fi;
    fi = fopen("conf.data", "rb");

    int64_t itn;
    double r;
    int64_t n;
    double* x;

    fread(&itn, 1, 8, fi);
    fread(&r, 1, 8, fi);
    fread(&n, 1, 8, fi);
    x = (double*)malloc(n * 8);
    fread(x, 1, n * 8, fi);
    fclose(fi);


    auto t1 = std::chrono::steady_clock::now();
    itv(r, x, n, itn);
    auto t2 = std::chrono::steady_clock::now();
    int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    printf("%d\n", d1);

    fi = fopen("out.data", "wb");
    fwrite(x, 1, n * 8, fi);
    fclose(fi);

    return 0;
}