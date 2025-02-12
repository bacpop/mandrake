# makefile for CUDA compile via conda

CXXFLAGS+=-Wall -Wextra -std=c++14 -fopenmp -fPIC -m64
ifdef DEBUG
  CXXFLAGS+= -O0 -g
  CUDAFLAGS = -g -G
else
  CXXFLAGS+= -O3 -flto -fno-fat-lto-objects -fvisibility=hidden
endif

LIBLOC = ${CONDA_PREFIX}
LDLIBS = -lz -lgomp -lpthread -lm -ldl -lrt
LDFLAGS=-Wl,-as-needed

CPPFLAGS+=-I"." -I$(LIBLOC)/include -I$(LIBLOC)/include/boost
LDFLAGS+= -L$(LIBLOC)/lib
CUDA_LDLIBS=-lcudadevrt -lcudart_static $(LDLIBS)
# python specific options
CPPFLAGS += -DPYTHON_EXT -DNDEBUG -DSCE_EXPORTS $(shell python3 -m pybind11 --includes)
PYTHON_LIB = SCE$(shell python3-config --extension-suffix)

ifdef GENARCH
	GENCODE=${GENARCH}
else
	GENCODE=-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90
endif
CUDA_LDFLAGS =-L$(LIBLOC)/lib -L${CUDA_HOME}/targets/x86_64-linux/lib/stubs -L${CUDA_HOME}/targets/x86_64-linux/lib
CUDAFLAGS +=-Xcompiler -fPIC -Xptxas -dlcm=ca --cudart static --relocatable-device-code=true $(GENCODE)
CXXFLAGS += -DGPU_AVAILABLE


CPU_OBJS=wtsne_cpu.o python_bindings.o
GPU_OBJS=wtsne_gpu.cu.o

all: python

clean:
	$(RM) $(CPU_OBJS) $(GPU_OBJS) *.o *.so ~*

python: $(PYTHON_LIB)

$(PYTHON_LIB): $(CPU_OBJS) $(GPU_OBJS)
	nvcc $(CUDAFLAGS) $(CUDA_LDFLAGS) -Wno-deprecated-gpu-targets -shared -dlink $^ -o device_link.o -Xnvlink $(CUDA_LDLIBS)
	$(LINK.cpp) $(CUDA_LDFLAGS) $(LDFLAGS) -shared $^ device_link.o -o $(PYTHON_LIB) $(CUDA_LDLIBS)

install_python: python
	install -d $(PYTHON_LIB_PATH)
	install $(PYTHON_LIB) $(PYTHON_LIB_PATH)

wtsne_gpu.cu.o:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -x cu -c wtsne_gpu.cu -o $@

.PHONY: all clean install python $(PYTHON_LIB) install_python
