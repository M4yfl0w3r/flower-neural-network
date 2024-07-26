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
    neovim          \
    wget

RUN wget https://github.com/Kitware/CMake/releases/download/v3.30.1/cmake-3.30.1.tar.gz
RUN tar -zxf cmake-3.30.1.tar.gz

WORKDIR "/cmake-3.30.1"

RUN ./bootstrap && make && make install

WORKDIR "/"

# run git clone https://github.com/llvm/llvm-project.git
#
# workdir "/llvm-project"
#
# run mkdir build
# run cmake -g ninja -s runtimes -b build -dllvm_enable_runtimes="libcxx;libcxxabi;libunwind"
# run ninja -c build
#
# workdir "/"
