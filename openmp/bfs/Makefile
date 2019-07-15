include ../../common/common-general.mk

CC_FLAGS += -fopenmp

all: bfs bfs_offload

bfs: bfs.cpp 
	$(CXX) $(CC_FLAGS) bfs.cpp -o bfs 

bfs_offload: bfs.cpp
	$(CXX) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs_offload result.txt
