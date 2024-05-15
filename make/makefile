main: hello_cuda hello_omp hello_mpi


hello_cuda:
	nvcc hello_cuda.cu -o hello_cuda 

hello_omp:
	g++ hello_omp.cpp -o hello_omp -fopenmp

hello_mpi:
	mpicxx hello_mpi.cpp -o hello_mpi 


.PHONY:clean
clean:
	rm hello_cuda hello_omp hello_mpi