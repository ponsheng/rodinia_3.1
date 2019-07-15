# This is common makefile config for projects
CC := gcc
CXX := g++

COMMONFLAGS = -O2 -w

ifeq ($(dbg), 1)
	COMMONFLAGS += -g
endif


CFLAGS := $(COMMONFLAGS)
CC_FLAGS := $(CFLAGS)
CXXFLAGS := $(COMMONFLAGS)
