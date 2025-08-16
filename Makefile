CC := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_PATH ?= /usr/local/cuda

CCFLAGS := -O3 -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -rdc=true -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_75,code=compute_75 -I$(CUDA_PATH)/include -I.
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -Xcompiler -pthread

CPU_SRC := RCKangaroo.cpp Ec.cpp utils.cpp
GPU_SRC := RCGpuCore.cu
GPU_CPP_SRC := GpuKang.cpp

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)
GPU_CPP_OBJECTS := $(GPU_CPP_SRC:.cpp=.o)

TARGET := rckangaroo

TESTS := test_mul_gpu test_add_gpu test_phi_gpu
TEST_OBJS := $(addprefix tests/, $(addsuffix .o, $(TESTS)))
TEST_STUB := tests/gpu_test_stubs.o

all: $(TARGET) tamesgen

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS) $(GPU_CPP_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

GpuKang.o: GpuKang.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

$(TESTS): %: tests/%.o RCGpuCore.o Ec.o utils.o GpuKang.o $(TEST_STUB)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

tests/%.o: tests/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TEST_STUB): tests/gpu_test_stubs.cpp
	$(CC) $(CCFLAGS) -I. -c $< -o $@

.PHONY: tests
tests: $(TESTS)


clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) $(GPU_CPP_OBJECTS) tamesgen tamesgen.o

tamesgen: tamesgen.o utils.o
	$(CC) $(CCFLAGS) -o $@ tamesgen.o utils.o
