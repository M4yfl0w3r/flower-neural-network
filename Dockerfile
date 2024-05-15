FROM ubuntu:24.04

RUN apt update

RUN apt -y install build-essential \
                   make            \
                   clang           \
                   ninja-build     \
                   libssl-dev      \
                   clang-tidy      \
                   libc++-dev      \
                   gcc             \
                   git             \
                   wget

RUN wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz
RUN tar -zxf cmake-3.29.3.tar.gz 

WORKDIR "/cmake-3.29.3"

RUN ./bootstrap && make && make install

WORKDIR "/"

RUN git clone https://github.com/llvm/llvm-project.git

WORKDIR "/llvm-project"

RUN mkdir build 
RUN cmake -G Ninja -S runtimes -B build -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind"
RUN ninja -C build

WORKDIR "/"

