#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <mpi.h>
#include <string>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <cstring>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <vector>
// #define DBG
namespace fs = std::filesystem;

constexpr size_t BLOCK_SIZE = 1024 * 1024;

void checksum(uint8_t *data, size_t len, uint8_t *obuf, int num_block, bool tail);
void print_checksum(std::ostream &os, uint8_t *md, size_t len);
int rank, nprocs;
int pioneer, successor;
int my_vrank;
int main(int argc, char *argv[]) {
  std::ios::sync_with_stdio(false);

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  char hostName[1024];
  gethostname(hostName, 1024);
  char theirHost[1024];
  char Bbuffer[1024];
  std::map<std::string,std::vector<int>> groups;
  for(int i = 0; i < nprocs; ++i) {
    strcpy(Bbuffer, hostName);
    MPI_Bcast(&Bbuffer, 1024, MPI_CHAR, i, MPI_COMM_WORLD);
    if(i != rank) {
      groups[Bbuffer].push_back(i);
      if(strcmp(Bbuffer, hostName))
        strcpy(theirHost, Bbuffer);
    }
  }
  #ifdef DBG
    std::cout << "hostName == " << hostName << " theirHost == " << theirHost << std::endl; 
    std::cout << "my len: " << strlen(hostName) << "their len: " << strlen(theirHost) << std::endl; 
  #endif
  groups[hostName].push_back(rank);
  std::sort(groups[hostName].begin(), groups[hostName].end());
  std::sort(groups[theirHost].begin(), groups[theirHost].end());
  int mapping[nprocs];
  auto it = std::find(groups[hostName].begin(), groups[hostName].end(), 0);
  int vrank = 0;
  if(it != groups[hostName].end()) {
    for (auto p : groups[hostName]) {
      mapping[p] = vrank++;
    }
    for(auto p : groups[theirHost]) {
      mapping[p] = vrank++;
    }
  }
  else {
    for(auto p : groups[theirHost]) {
      mapping[p] = vrank++;
    }
    for (auto p : groups[hostName]) {
      mapping[p] = vrank++;
    }
  }
  my_vrank = mapping[rank];
  int vpioneer = my_vrank ? my_vrank - 1 : nprocs - 1;
  int vsuccessor = my_vrank != nprocs - 1 ? my_vrank + 1 : 0; 
  for(int i = 0; i < nprocs; ++i) {
    if(mapping[i] == vpioneer)
      pioneer = i;
    if(mapping[i] == vsuccessor)
      successor = i;
  }
  #ifdef DBG
    for(int i = 0; i < nprocs; ++i)
      std::cout << "mapping[" << i << "] == " << mapping[i] << std::endl;
  #endif
  fs::path input_path = argv[1];
  fs::path output_path = argv[2];
  #ifdef DBG
    auto total_begin_time = std::chrono::high_resolution_clock::now();
  #endif

  auto file_size = fs::file_size(input_path);

  uint8_t *buffer = nullptr;
  int num_block_read;
  
  if (file_size != 0) {
   
    // read the file content in binary format
    // std::ifstream istrm(input_path, std::ios::binary)
//===========================================================
    buffer = new uint8_t[file_size];
    int fd = open(argv[1], O_RDONLY); 
    void * addr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      perror("mmap failed");
      exit(EXIT_FAILURE);
    }
    std::cout<<std::hex<<addr<<std::endl;
    int i;
    for(i = 0; (int64_t)file_size - (i * nprocs + my_vrank) * (int64_t)BLOCK_SIZE > 0 ; ++i) {
      madvise((char *)addr + ((i + 1) * nprocs + my_vrank) * BLOCK_SIZE, (int64_t)BLOCK_SIZE, MADV_SEQUENTIAL | MADV_WILLNEED);
      memcpy(buffer + i * BLOCK_SIZE, (char *)addr + (i * nprocs + my_vrank) * BLOCK_SIZE, std::min((int64_t)BLOCK_SIZE, (int64_t)file_size - (i * nprocs + my_vrank) * (int64_t)BLOCK_SIZE));
    }
    // int i = 0;
    // while (!istrm.fail() && !istrm.eof())
    // {
    //   istrm.seekg((i * nprocs + rank) * BLOCK_SIZE);
    //   istrm.read(reinterpret_cast<char *>(buffer + i * BLOCK_SIZE), std::min(BLOCK_SIZE, file_size - i * BLOCK_SIZE)); //this was fucking wrong?!
    //   istrm.read(reinterpret_cast<char *>(buffer + i * BLOCK_SIZE), std::min(BLOCK_SIZE, file_size - (i * nprocs + rank) * BLOCK_SIZE));//this is the right one!
    //   ++i;
    // }

    // num_block_read = i - 1;
//============================================================
    num_block_read = i;
    MPI_Barrier(MPI_COMM_WORLD);
    munmap(addr, file_size);
    close(fd);
  }
  
  
  bool tail = ( (file_size + BLOCK_SIZE - 1) / BLOCK_SIZE - 1) % nprocs == my_vrank % nprocs;
  
  #ifdef DBG
    // record begin time
    auto begin_time = std::chrono::high_resolution_clock::now();
  #endif

  // calculate the checksum
  uint8_t obuf[SHA512_DIGEST_LENGTH];
  checksum(buffer, file_size, obuf, num_block_read, tail);

  #ifdef DBG
    // record end time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - begin_time);
  #endif

  // write checksum to output file
  std::ofstream output_file(output_path);
  if (tail)
    print_checksum(output_file, obuf, SHA512_DIGEST_LENGTH);

  delete[] buffer;


  #ifdef DBG
    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end_time - total_begin_time);
  #endif

  #ifdef DBG
    std::cout << std::dec;
    if (rank == 0) {
      std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
      std::cout << "Checksum time: " << duration.count() << " ms" << std::endl;
    }
  #endif

  MPI_Finalize();

  return 0;
}

inline
void checksum(uint8_t *data, size_t len, uint8_t *obuf, int num_block, bool tail) {
  
  uint8_t prev_md[SHA512_DIGEST_LENGTH];

  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  EVP_MD *sha512 = EVP_MD_fetch(nullptr, "SHA512", nullptr);
  unsigned int l = 0;
  for (int i = 0; i < num_block; i++) {
    uint8_t buffer[BLOCK_SIZE]{};
    EVP_DigestInit_ex(ctx, sha512, nullptr);
    memcpy(buffer, data + i * BLOCK_SIZE, BLOCK_SIZE);
    EVP_DigestUpdate(ctx, buffer, BLOCK_SIZE);
    
    if (i == 0 && my_vrank == 0) 
      SHA512(nullptr, 0, prev_md);
    else 
    {
      MPI_Recv(prev_md, SHA512_DIGEST_LENGTH, MPI_BYTE, pioneer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    EVP_DigestUpdate(ctx, prev_md, SHA512_DIGEST_LENGTH);
    EVP_DigestFinal_ex(ctx, prev_md, &l);

    if (!tail || i != num_block - 1){
      MPI_Send(prev_md, SHA512_DIGEST_LENGTH, MPI_BYTE, successor, 0, MPI_COMM_WORLD);
    }
  }
  memcpy(obuf, prev_md, SHA512_DIGEST_LENGTH);
  EVP_MD_CTX_free(ctx);
  EVP_MD_free(sha512);
}


inline
void print_checksum(std::ostream &os, uint8_t *md, size_t len) {
  for (int i = 0; i < len; i++) {
    os << std::setw(2) << std::setfill('0') << std::hex
       << static_cast<int>(md[i]);
  }
}
