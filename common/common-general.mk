# This is common makefile config for projects
CC := clang 
CXX := clang++

COMMONFLAGS = -O2 -w

ifeq ($(dbg), 1)
	COMMONFLAGS += -g
endif


CFLAGS := $(COMMONFLAGS)
CC_FLAGS := $(CFLAGS)
CCFLAGS := $(CFLAGS)
CXXFLAGS := $(COMMONFLAGS)



# flags for Openmp
## legacy rodinia command
OFFLOAD_CC = icc
OFFLOAD_CC_FLAGS = -offload-option,mic,compiler,"-no-opt-prefetch"
