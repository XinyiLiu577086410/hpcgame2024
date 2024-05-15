#include <iostream>
#include <chrono>
#include <omp.h>
#include <immintrin.h>
inline void itv(double r, double* x, int64_t n, int64_t itn) {
    __m512d _r, _one, _n_one;
	_r = _mm512_set1_pd(r);
	_one = _mm512_set1_pd(1.0);
	_n_one = _mm512_set1_pd(-1.0);
    __m512d _x[5], _rx[5], _1mx[5];
    static const int64_t inc = 32; 
	#pragma omp parallel for private(_x,_rx,_1mx) proc_bind(close) schedule(auto)
    for (int64_t i = 0; i < n; i+=inc) {
        //==========================
        register double * xai = ( double* )( x + i );
        register double * px[4] {xai, xai + 8, xai + 16, xai + 24};
		_x[0] = _mm512_load_pd(px[0]);
		_x[1] = _mm512_load_pd(px[1]);
		_x[2] = _mm512_load_pd(px[2]);
		_x[3] = _mm512_load_pd(px[3]);
        _mm_prefetch(xai + 32, _MM_HINT_T0);
        _mm_prefetch(xai + 40, _MM_HINT_T0);
        _mm_prefetch(xai + 48, _MM_HINT_T0);
        _mm_prefetch(xai + 54, _MM_HINT_T0);

	  	for (int64_t j = 0; j < itn; j+=1) {
			// _1mx = _mm512_add_pd(_one, _mm512_mul_pd(_n_one, _x));
			// _rx = _mm512_mul_pd(_r,_x);
            // // _1mx = _mm512_add_pd(_one, _mm512_xor_pd(_xor_mask, _x));
            // _x = _mm512_mul_pd(_rx, _1mx);
    	    // number of instruction 4
        //=================================================================
            // // _1mx = _mm512_add_pd(_one, _mm512_mul_pd(_n_one, _x));
			// _1mx[] = _mm512_fmadd_pd(_x[], _n_one[], _one[]);
            // _rx[] = _mm512_mul_pd(_r[],_x[]);
            // _x[] = _mm512_mul_pd(_rx[], _1mx[]);
    	    // // number of instruction 3
        //==================================================================
            _1mx[0] = _mm512_fmadd_pd(_x[0], _n_one, _one);
            _1mx[1] = _mm512_fmadd_pd(_x[1], _n_one, _one);
            _1mx[2] = _mm512_fmadd_pd(_x[2], _n_one, _one);
            _1mx[3] = _mm512_fmadd_pd(_x[3], _n_one, _one);

            _rx[0] = _mm512_mul_pd(_r,_x[0]);
            _rx[1] = _mm512_mul_pd(_r,_x[1]);
            _rx[2] = _mm512_mul_pd(_r,_x[2]);
            _rx[3] = _mm512_mul_pd(_r,_x[3]);

            _x[0] = _mm512_mul_pd(_rx[0], _1mx[0]);
            _x[1] = _mm512_mul_pd(_rx[1], _1mx[1]);
            _x[2] = _mm512_mul_pd(_rx[2], _1mx[2]);
            _x[3] = _mm512_mul_pd(_rx[3], _1mx[3]);
        }
		_mm512_store_pd(px[0], _x[0]);
        _mm512_store_pd(px[1], _x[1]);
		_mm512_store_pd(px[2], _x[2]);
		_mm512_store_pd(px[3], _x[3]);
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
    x = (double*)aligned_alloc(64, n * 8);
    fread(x, 1, n * 8, fi);
    fclose(fi);


    // auto t1 = std::chrono::steady_clock::now();
    itv(r, x, n, itn);
    // auto t2 = std::chrono::steady_clock::now();
    // int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    // printf("%d\n", d1);

    fi = fopen("out.data", "wb");
    fwrite(x, 1, n * 8, fi);
    fclose(fi);

    return 0;
}