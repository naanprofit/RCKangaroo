CC := g++
NVCC := /usr/local/cuda/bin/nvcc
CUDA_PATH ?= /usr/local/cuda

CCFLAGS := -O3 -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -gencode=arch=compute_89,code=compute_89 -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_75,code=compute_75
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -Xcompiler -pthread

CPU_SRC := RCKangaroo.cpp Ec.cpp utils.cpp GpuKang.cpp
GPU_SRC := RCGpuCore.cu

CPP_OBJECTS := $(CPU_SRC:.cpp=.o)
CU_OBJECTS := $(GPU_SRC:.cu=.o)

TARGET := rckangaroo

all: $(TARGET) tamesgen

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@


clean:
	rm -f $(CPP_OBJECTS) $(CU_OBJECTS) tamesgen tamesgen.o

tamesgen: tamesgen.o utils.o
	$(CC) $(CCFLAGS) -o $@ tamesgen.o utils.o
