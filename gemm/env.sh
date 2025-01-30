eval `spack load --sh intel-oneapi-compilers`
 # compile with 
icpx -std=c++17 ./baseline.cc -o gemm-acc -fiopenmp -fopenmp-targets=spir64
icpx -std=c++17 ./generate.cc -o generate       