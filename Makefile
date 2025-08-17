# =========================
# RCKangaroo Makefile
# =========================
# Default CUDA install; override with:  make CUDA_PATH=/opt/cuda
CUDA_PATH ?= /usr/local/cuda

NVCC := $(CUDA_PATH)/bin/nvcc
CC   := g++

# -------------------------
# Flags
# -------------------------
CXXSTD   := -std=c++17
WARN     := -Wall -Wextra -Wno-unused-parameter
OPT      := -O3 -DNDEBUG
INCLUDES := -I. -I$(CUDA_PATH)/include

CCFLAGS   := $(OPT) $(CXXSTD) $(WARN) $(INCLUDES)

# Architectures present:
#  - Ada (RTX 4060 Ti): sm_89  (ship SASS + PTX)
#  - Pascal (Quadro P1000): sm_61 (ship SASS only to avoid old-PTX JIT surprises)
GENCODE   := \
  -gencode arch=compute_89,code=sm_89   -gencode arch=compute_89,code=compute_89

NVCCFLAGS := $(OPT) -rdc=true $(INCLUDES) $(GENCODE)

LDFLAGS   := -L$(CUDA_PATH)/lib64 -lcudart -Xcompiler -pthread

# -------------------------
# Sources / Objects
# -------------------------
CPU_SRC             := RCKangaroo.cpp Ec.cpp utils.cpp
GPU_HOST_WITH_DEVICE:= GpuKang.cpp          # must be built with nvcc (uses device symbols)
GPU_SRC             := RCGpuCore.cu

CPP_OBJECTS         := $(CPU_SRC:.cpp=.o)
NVCC_HOST_OBJECTS   := $(GPU_HOST_WITH_DEVICE:.cpp=.o)
CU_OBJECTS          := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

# -------------------------
# Phony targets
# -------------------------
.PHONY: all clean tests

all: $(TARGET) tamesgen

# -------------------------
# Link main binary with nvcc (device linking)
# -------------------------
$(TARGET): $(CPP_OBJECTS) $(NVCC_HOST_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

# -------------------------
# Compile rules
# -------------------------
%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Force GpuKang.cpp through nvcc so __constant__ symbols (BETA/BETA2) resolve
GpuKang.o: GpuKang.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# -------------------------
# tamesgen utility
# -------------------------
tamesgen.o: tamesgen.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

tamesgen: tamesgen.o utils.o
	$(CC) $(OPT) $(CXXSTD) -o $@ $^ $(INCLUDES)

# -------------------------
# Tiny GPU test binaries
#   (provide test_mul_gpu.cu, test_add_gpu.cu, test_phi_gpu.cu)
# -------------------------
TESTS := test_mul_gpu test_add_gpu test_phi_gpu

$(TESTS): %: %.o RCGpuCore.o Ec.o utils.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

test_mul_gpu.o: test_mul_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test_add_gpu.o: test_add_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

test_phi_gpu.o: test_phi_gpu.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

tests: $(TESTS)

# -------------------------
# Cleanup
# -------------------------
clean:
	rm -f $(CPP_OBJECTS) $(NVCC_HOST_OBJECTS) $(CU_OBJECTS) \
      tamesgen.o tamesgen $(TESTS) *.o $(TARGET)
