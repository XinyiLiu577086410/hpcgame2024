#include <vector>
#include <unordered_map>
#include <map>
#include <stdint.h>
#include <math.h>
#include <chrono>
#include <cstring>
#include <omp.h>
#include <immintrin.h>
#ifdef _WIN32
#include <windows.h>
#define popcnt __popcnt64
#else
#define popcnt __builtin_popcountll
#endif

#include <iostream>

typedef std::unordered_map<uint64_t, size_t> map_t;
//====================================================
// typedef std::map<uint64_t, size_t> map_t;

struct restrict_t {
    int offset, range, minocc, maxocc;
    int occ;
    uint64_t substate;
};

template <typename VT>
struct term_t {
    VT value;
    uint64_t an, cr, signmask, sign;
};

// struct sparse_t {
//     std::vector<size_t> row;
//     std::vector<size_t> col;
//     std::vector<double> data;
// };

struct sparse_t {
    std::vector<size_t> dataPtr;
    std::vector<size_t> indices;
    std::vector<double> data;
};

int itrest(restrict_t& rest) {
    if (!rest.substate) {
        goto next;
    }
    {
        uint64_t x = rest.substate & (-(int64_t)rest.substate);
        uint64_t y = x + rest.substate;
        rest.substate = y + (y ^ rest.substate) / x / 4;
    }
    if (rest.substate >> rest.range) {
    next:
        if (rest.occ == rest.maxocc) {
            rest.occ = rest.minocc;
            rest.substate = (uint64_t(1) << rest.occ) - 1;
            return 1;
        }
        rest.occ++;
        rest.substate = (uint64_t(1) << rest.occ) - 1;
        return 0;
    }
    return 0;
}

int itrest(std::vector<restrict_t>& rest) {
    for (restrict_t& re : rest) {
        if (!itrest(re)) {
            return 0;
        }
    }
    return 1;
}

uint64_t getstate(const std::vector<restrict_t>& rest) {
    uint64_t state = 0;
    for (const restrict_t& re : rest) {
        state |= re.substate << re.offset;
    }
    return state;
}

int generatetable(std::vector<uint64_t>& table, map_t& map, std::vector<restrict_t>& rest) {
    for (restrict_t& re : rest) {
        re.occ = re.minocc;
        re.substate = (uint64_t(1) << re.occ) - 1;
    }

    size_t index = 0;
    do {
        uint64_t state = getstate(rest);
        table.push_back(state);
        map.insert(std::make_pair(state, index));
        index++;
    } while (!itrest(rest));

    return 0;
}

template <typename VT>
term_t<VT> getterm(VT value, const std::vector<int>& cr, const std::vector<int>& an) {
    term_t<VT> term;
    term.value = value;
    term.an = 0;
    term.cr = 0;
    term.signmask = 0;
    uint64_t signinit = 0;

    for (int x : an) {
        uint64_t mark = uint64_t(1) << x;
        term.signmask ^= (mark - 1) & (~term.an);
        term.an |= mark;
    }
    for (int x : cr) {
        uint64_t mark = uint64_t(1) << x;
        signinit ^= (mark - 1) & term.cr;
        term.signmask ^= (mark - 1) & (~term.an) & (~term.cr);
        term.cr |= mark;
    }
    term.sign = popcnt(signinit ^ (term.signmask & term.an));
    term.signmask = term.signmask & (~term.an) & (~term.cr);

    return term;
}

// template <typename VT>
// int act(std::vector<size_t>& row, std::vector<size_t>& col, std::vector<VT>& data, const std::vector<term_t<VT>>& op, const std::vector<uint64_t>& table, const map_t& map) {
//     int64_t n = table.size();
//
//     for (int64_t i = 0; i < n; i++) {
//         uint64_t srcstate = table[i];
//
//         for (const term_t<VT>& term : op) {
//             if ((srcstate & term.an) == term.an) {
//                 uint64_t dststate = srcstate ^ term.an;
//                 if ((dststate & term.cr) == 0) {
//                     dststate ^= term.cr;
//
//                     auto it = map.find(dststate);
//                     if (it != map.end()) {
//                         uint64_t sign = term.sign + popcnt(srcstate & term.signmask);
//                         VT v = term.value;
//                         if (sign & 1) {
//                             v = -v;
//                         }
//                         data.push_back(v);
//                         col.push_back(i);
//                         row.push_back(it->second);
//                     }
//                 }
//             }
//         }
//     }
//
//     return 0;
// }

template <typename VT> int act(std::vector<size_t>& dataPtr, std::vector<size_t>& indices, std::vector<VT>& data, const std::vector<term_t<VT>>& op, const std::vector<uint64_t>& table, const map_t& map) {
    int64_t n = table.size();
    size_t cur = 0;
    // data.push_back(cur); // So stupid I am!?
    dataPtr.push_back(cur);
    // #pragma omp parallel for 
    for (int64_t i = 0; i < n; i++) {
        uint64_t srcstate = table[i];

        for (const term_t<VT>& term : op) {
            if ((srcstate & term.an) == term.an) {
                uint64_t dststate = srcstate ^ term.an;
                if ((dststate & term.cr) == 0) {
                    dststate ^= term.cr;

                    auto it = map.find(dststate);
                    if (it != map.end()) {
                        uint64_t sign = term.sign + popcnt(srcstate & term.signmask);
                        VT v = term.value;
                        if (sign & 1) {
                            v = -v;
                        }
                        //======BASELINE======
                        // data.push_back(v);
                        // col.push_back(i);
                        // row.push_back(it->second);
                        //======OPTIMIZED=====
                        ++cur;
                        data.push_back(v);
                        indices.push_back(it->second);
                    }
                }
            }
        }
        
        // indices.push_back(cur); // Segfault bcoz of this 
        dataPtr.push_back(cur);

    }

    return 0;
}


int readss(FILE* fi, std::vector<uint64_t>& table, map_t& map) {
    int n;
    fread(&n, 1, 4, fi);
    std::vector<restrict_t> restv(n);
    for (auto& rest : restv) {
        fread(&rest, 1, 16, fi);
    }
    generatetable(table, map, restv);
    return 0;
}

int readop(FILE* fi, std::vector<term_t<double>>& op) {
    int n, order;
    fread(&n, 1, 4, fi);
    fread(&order, 1, 4, fi);

    std::vector<double> v(n);
    fread(v.data(), 1, 8 * n, fi);

    std::vector<int> rawterm(order);
    std::vector<int> cr, an;

    for (int i = 0; i < n; i++) {
        fread(rawterm.data(), 1, 4 * order, fi);
        int tn = rawterm[0];

        for (int j = 0; j < tn; j++) {
            int type = rawterm[tn * 2 - 1 - j * 2];
            if (type) {
                cr.push_back(rawterm[tn * 2 - j * 2]);
            }
            else {
                an.push_back(rawterm[tn * 2 - j * 2]);
            }
        }

        op.push_back(getterm(v[i], cr, an));
        cr.clear();
        an.clear();
    }

    return 0;
}

//bottleneck
//out=m*v;
void mmv(std::vector<double>& out, const sparse_t& m, const std::vector<double>& v) {
    // for (auto& x : out) {
    //     x = 0;
    // }
    //
    // for (size_t i = 0; i < m.data.size(); i++) {
    //     out[m.row[i]] += m.data[i] * v[m.col[i]];
    // }
//===============OPTIMIZATION================
    //
    // auto _out = &out[0];
    // auto _m_data = &m.data[0];
    // auto _m_col = &m.col[0];
    // auto _m_row = &m.row[0];
    // auto _v = &v[0];
    // auto m_data_size = m.data.size();
    //
    // memset(_out, 0, sizeof(double) * out.size());
    // #pragma omp parallel for simd
    // for (size_t i = 0; i < m_data_size; i++) {
    //     _out[_m_row[i]] += _m_data[i] * _v[_m_col[i]];
    // }
//===============OPTIMIZATION================
    //
    // for (auto& x : out) {
    //     x = 0;
    // }
    // auto l = m.data.size();
    // static std::vector<double> tmp(l);
    // memset(&tmp[0], 0, tmp.size()*sizeof(double));
    // #pragma omp parallel for
    // for (size_t i = 0; i < l; i++) {
    //     tmp[i] = m.data[i] * v[m.col[i]];
    // }
    // // the add order matters!
    // for (size_t i = 0; i < l; i++) {
    //     out[m.row[i]] += tmp[i];
    // }
//===============OPTIMIZATION 1.0================
    // auto _out = &out[0];
    // auto _m_data = &m.data[0];
    // auto _m_col = &m.col[0];
    // auto _m_row = &m.row[0];
    // auto _v = &v[0];
    // auto l = m.data.size();
    // memset(&out[0], 0, out.size()*sizeof(double));
    // static std::vector<double> tmp(l);
    // auto _tmp = &tmp[0];
    // memset(&tmp[0], 0, tmp.size()*sizeof(double));
    // #pragma omp parallel for
    // for (size_t i = 0; i < l; i++) {
    //     _mm_prefetch(&_v[_m_col[i+1]], _MM_HINT_T0);
    //     _tmp[i] = _m_data[i] * _v[_m_col[i]];
    // }
    // // #pragma omp parallel for
    // // the add order matters!
    // for (size_t i = 0; i < l; i++) {
    //     _mm_prefetch(&_out[_m_row[i+8]], _MM_HINT_T0);
    //     _out[_m_row[i]] += _tmp[i];
    // }
//===============OPTIMIZATION 2.0: CSR================
    auto _out = &out[0];
    auto _m_data = &m.data[0];
    auto _m_indices = &m.indices[0];
    auto _m_dataPtr = &m.dataPtr[0];
    auto _v = &v[0];
    // auto l = m.data.size();
    auto n = m.dataPtr.size() - 1;
    memset(&out[0], 0, out.size()*sizeof(double));
    /*******************COO********************/
    // for (size_t i = 0; i < m.data.size(); i++) {
    //     out[m.row[i]] += m.data[i] * v[m.col[i]];
    // }
    /*******************CSR*******************/
    // printf("%lu", n);
    // exit(0);
    #pragma omp parallel for proc_bind(close) schedule(static)
    for(size_t row = 0; row < n; ++row) {
        register auto l = _m_dataPtr[row + 1];
        static const int inc = 4;
        size_t ptr = _m_dataPtr[row];
        register double tmp = 0.0;
        for( ; ptr + inc <= l; ptr += inc ) {
            // _out[row] += _m_data[ptr] * _v[_m_indices[ptr]];
            // _out[row] += _m_data[ptr + 1] * _v[_m_indices[ptr + 1]];
            // _out[row] += _m_data[ptr + 2] * _v[_m_indices[ptr + 2]];
            // _out[row] += _m_data[ptr + 3] * _v[_m_indices[ptr + 3]];
            //========================================================
            tmp += _m_data[ptr] * _v[_m_indices[ptr]];
            tmp += _m_data[ptr + 1] * _v[_m_indices[ptr + 1]];
            tmp += _m_data[ptr + 2] * _v[_m_indices[ptr + 2]];
            tmp += _m_data[ptr + 3] * _v[_m_indices[ptr + 3]];
        }
        for( ; ptr < l; ptr += 1 ) {
            tmp += _m_data[ptr] * _v[_m_indices[ptr]];
        }
        _out[row] = tmp;
    }
//===================BAD========================
    // auto _out = &out[0];
    // auto _m_data = &m.data[0];
    // auto _m_col = &m.col[0];
    // auto _m_row = &m.row[0];
    // auto _v = &v[0];
    // auto l = m.data.size();
    // memset(&out[0], 0, out.size()*sizeof(double));
    // static std::vector<double> tmp(l);
    // auto _tmp = &tmp[0];
    // memset(&tmp[0], 0, tmp.size()*sizeof(double));
    //
    // __m512d zmm_tmp, zmm_data, zmm_v;  
    // static const int inc = 8;
    // #pragma omp parallel for private(zmm_tmp, zmm_data, zmm_v)
    // for (size_t i = 0; i < l; i += inc) {
    //     zmm_v = _mm512_set_pd( _v[_m_col[i]], _v[_m_col[i + 1]], _v[_m_col[i + 2]], _v[_m_col[i + 3]],\
    //                          _v[_m_col[i + 4]], _v[_m_col[i + 5]], _v[_m_col[i + 6]], _v[_m_col[i + 7]] );
    //     zmm_data = _mm512_loadu_pd(&_m_data[i]);
    //     _mm512_storeu_pd(&_tmp[i], _mm512_mul_pd(zmm_data, zmm_v));
    // }
    // // the add order matters!
    // for (size_t i = 0; i < l; i++) {
    //     _out[_m_row[i]] += _tmp[i];
    // }
}

//v1'*v2;
double dot(const std::vector<double>& v1, const std::vector<double>& v2) {
    // double s = 0;
    // for (size_t i = 0; i < v1.size(); i++) {
    //     s += v1[i] * v2[i];
    // }
    // return s;
//===============OPTIMIZATION================

    double s = 0;
    auto l = v1.size();
    auto _v1 = &v1[0];
    auto _v2 = &v2[0];
    for (size_t i = 0; i < l; i++) {
        s += _v1[i] * _v2[i];
    }
    return s;
//===============OPTIMIZATION================
    // auto _v1 = &v1[0];
    // auto _v2 = &v2[0];
    // double s = 0;
    // auto l = v1.size();
    // __m512d _v1_avx, _v2_avx, _s_avx;
    // size_t i;
    // for (size_t i = 0; i + 8 <= l; i += 8) {
    //     _v1_avx = _mm512_loadu_pd(&_v1[i]);
    //     _v2_avx = _mm512_loadu_pd(&_v2[i]);
    //     _s_avx = _mm512_mul_pd(_v1_avx, _v2_avx);
    //     s += _mm512_reduce_add_pd(_s_avx);
    // }
    // for (; i<l; ++i) {
    //     s += _v1[i] * _v2[i];
    // }
    // return s;
}

//v1+=s*v2;
void avv(std::vector<double>& v1, const double s, const std::vector<double>& v2) {
    // for (size_t i = 0; i < v1.size(); i++) {
    //     v1[i] += s * v2[i];
    // }
//===============OPTIMIZATION================
    auto _v1 = &v1[0];
    auto _v2 = &v2[0];
    auto l = v1.size();
    for (size_t i = 0; i < l; i++) {
        _v1[i] += s * _v2[i];
    }
//============================================
    // auto _v1 = &v1[0];
    // auto _v2 = &v2[0];
    // auto l = v1.size();
    // __m512d zmm_s, zmm_v1[4], zmm_v2[4];
    // zmm_s = _mm512_set1_pd(s);
    // static const int inc = 32;
    // for (size_t i = 0; i + inc <= l; i += inc) {
    //     zmm_v2[0] = _mm512_loadu_pd(&_v2[i]);
    //     zmm_v2[1] = _mm512_loadu_pd(&_v2[i + 8]);
    //     zmm_v2[2] = _mm512_loadu_pd(&_v2[i + 16]);
    //     zmm_v2[3] = _mm512_loadu_pd(&_v2[i + 24]);
    //
    //     zmm_v1[0] = _mm512_mul_pd(zmm_s, zmm_v2[0]);
    //     zmm_v1[1] = _mm512_mul_pd(zmm_s, zmm_v2[1]);
    //     zmm_v1[2] = _mm512_mul_pd(zmm_s, zmm_v2[2]);
    //     zmm_v1[3] = _mm512_mul_pd(zmm_s, zmm_v2[3]);
    //
    //     _mm512_storeu_pd(&_v1[i], zmm_v1[0]);
    //     _mm512_storeu_pd(&_v1[i + 8], zmm_v1[1]);
    //     _mm512_storeu_pd(&_v1[i + 16], zmm_v1[2]);
    //     _mm512_storeu_pd(&_v1[i + 24], zmm_v1[3]);
    // }
}

//cause numeric unstable 
//v*=s;
void msv(const double s, std::vector<double>& v) {
    // for (auto& x : v) {
    //     x *= s;
    // }
//===============OPTIMIZATION================
    // auto _v = &v[0];
    // auto l = v.size();
    // __m512d _s = _mm512_set1_pd(s);
    // __m512d _x;
    // size_t i;
    // for (i = 0; i + 8 <= l; i += 8){
    //     _x = _mm512_loadu_pd(&_v[i]);
    //     _x = _mm512_mul_pd(_s, _x);
    //     _mm512_storeu_pd(&_v[i], _x);
    // }
    // for(; i < l; ++i)
    //     _v[i] *= s;
    //
    /*pool improvement*/
//=======DEAL WITH NUIMERIC UNSTABLE=========
    auto _v = &v[0];
    auto l = v.size();
    for (size_t i = 0; i < l; ++i, ++_v) {
        *_v *= s;
    }
}

//v'*v;
double norm2(const std::vector<double>& v) {
    // double s = 0;
    // for (auto& x : v) {
    //     s += x * x;
    // }
    // return s;
//=========OPTMIZATION===========
    // return dot(v, v);
//=========OPTMIZATION===========
    double s = 0;
    auto _v = &v[0];
    auto l = v.size();
    for (size_t i = 0; i < l; ++i) {
        s += _v[i] * _v[i];
    }
    return s;
}

void getsp(std::vector<double>& out, int itn, const sparse_t m, std::vector<double>& v) {
    out.resize(itn * 2);
    auto _out = &out[0];
    _out[0] = sqrt(norm2(v));
    msv(1.0 / _out[0], v);

    std::vector<double> a(itn), b(itn - 1);

    std::vector<double> v_(v.size()), v__(v.size());
    auto _a = &a[0];
    auto _b = &b[0];
    // for (int i = 0; i < itn; i++) {
    //     v__.swap(v_);
    //     v_.swap(v);
    //     mmv(v, m, v_);
    //     _a[i] = dot(v, v_);
    //
    //     if (i < itn - 1) {
    //         if (i == 0) {
    //             avv(v, -_a[i], v_);
    //         }
    //         else {
    //             avv(v, -_a[i], v_);
    //             avv(v, -_b[i - 1], v__);
    //         }
    //    
    //         _b[i] = sqrt(norm2(v));
    //         msv(1.0 / _b[i], v);
    //     }
    // }
    //===================================
    {
        int i = 0;
        v__.swap(v_);
        v_.swap(v);
        mmv(v, m, v_);
        _a[i] = dot(v, v_);

        avv(v, -_a[i], v_);

        _b[i] = sqrt(norm2(v));
        msv(1.0 / _b[i], v);

        for (i = 1; i < itn - 1; i++) {
            v__.swap(v_);
            v_.swap(v);
            mmv(v, m, v_);
            _a[i] = dot(v, v_);

            avv(v, -_a[i], v_);
            avv(v, -_b[i - 1], v__);

            _b[i] = sqrt(norm2(v));
            msv(1.0 / _b[i], v);
        }

        i = itn - 1;
        v__.swap(v_);
        v_.swap(v);
        mmv(v, m, v_);
        _a[i] = dot(v, v_);
    }




    for (int i = 0; i < itn; i++) {
        _out[1 + i] = _a[i];
    }
    for (int i = 0; i < itn - 1; i++) {
        _out[1 + itn + i] = _b[i];
    }
    //=====================================
    // memcpy( &_out[1], &_a[0], itn * sizeof(double) );
    // memcpy( &_out[1 + itn], &_b[0], (itn - 1) * sizeof(double) );
}

int main()
{
    FILE* fi;
    std::vector<uint64_t> table;
    map_t map;
    std::vector<term_t<double>> op;

    fi = fopen("conf.data", "rb");
    auto t1 = std::chrono::steady_clock::now();
    readss(fi, table, map);
    auto t2 = std::chrono::steady_clock::now();
    readop(fi, op);

    sparse_t opm;
    // act(opm.row, opm.col, opm.data, op, table, map);
//========================================================
    act(opm.dataPtr, opm.indices, opm.data, op, table, map);

    auto t3 = std::chrono::steady_clock::now();

    int itn;
    fread(&itn, 1, 4, fi);

    std::vector<double> v(table.size());
    fread(v.data(), 1, table.size() * 8, fi);

    fclose(fi);

    std::vector<double> result;
    getsp(result, itn, opm, v);

    auto t4 = std::chrono::steady_clock::now();
    fi = fopen("out.data", "wb");
    fwrite(result.data(), 1, 16 * itn, fi);
    fclose(fi);

    int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    int d2 = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
    int d3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    printf("%d,%d,%d\n", d1, d2, d3);
    std::cout << "Hello World!\n";
}
