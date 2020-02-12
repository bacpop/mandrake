
CXXFLAGS=-Wall -Wextra -std=c++11 -march=native -ffast-math -fopenmp
ifdef DEBUG
  CXXFLAGS+= -O0 -g
else ifdef PROFILE
  CXXFLAGS+= -O2 -g
else
  CXXFLAGS+= -Ofast
endif

UNAME_S := $(shell uname -s)
LIBLOC = ${CONDA_PREFIX}
ifeq ($(UNAME_S),Linux)
	CXXFLAGS+= -DMKL_ILP64 -m64
	ifdef PROFILE
		CXXFLAGS+= -Wl,--compress-debug-sections=none
	endif
	LDFLAGS=-Wl,-as-needed
endif
ifeq ($(UNAME_S),Darwin)
	LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
endif

CPPFLAGS=-I$(LIBLOC)/include
LDFLAGS+= -L$(LIBLOC)/lib -lm -lgsl -lgslcblas -lgomp

PROGRAMS=wtsne_cpu wtsne_gpu
#nvcc wtsne_gpu.cu -o  wtsne_gpu --cudart static -O3 --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -lgsl -lgslcblas

all: $(PROGRAMS)

static: $(STATIC_PROGRAMS)

clean:
	$(RM) *.o ~* $(PROGRAMS)

install: all
	install -d $(BINDIR)
	install $(PROGRAMS) $(BINDIR)

wtsne_cpu: wtsne_cpu.o
	$(LINK.cpp) $^ -o $@

wtsne_gpu:
	nvcc $@.cu -o $@ --cudart static -O3 --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -lgsl -lgslcblas

.PHONY: all clean install