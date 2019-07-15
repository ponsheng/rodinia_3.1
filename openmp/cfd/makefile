include ../../common/common-general.mk

OMPFLAGS= -fopenmp $(CXXFLAGS)

OMP_NUM_THREADS = 8
MIC_OMP_NUM_THREADS = 216

all: euler3d_cpu euler3d_cpu_double pre_euler3d_cpu pre_euler3d_cpu_double euler3d_cpu_offload

euler3d_cpu: euler3d_cpu.cpp
	$(CXX) $(OMPFLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu

euler3d_cpu_offload:
	$(CXX) $(OMPFLAGS) -DOMP_OFFLOAD -Dblock_length=$(MIC_OMP_NUM_THREADS) euler3d_cpu.cpp -o euler3d_cpu_offload

# Old command for icc
# icc -g -O3 -DOMP_OFFLOAD -Dblock_length=$(MIC_OMP_NUM_THREADS) -fopenmp -offload-option,mic,compiler,"-no-opt-prefetch" euler3d_cpu.cpp -o euler3d_cpu_offload

euler3d_cpu_double: euler3d_cpu_double.cpp
	$(CXX) $(OMPFLAGS) -Dblock_length=$(OMP_NUM_THREADS) euler3d_cpu_double.cpp -o euler3d_cpu_double

pre_euler3d_cpu: pre_euler3d_cpu.cpp
	$(CXX) $(OMPFLAGS) -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu.cpp -o pre_euler3d_cpu

pre_euler3d_cpu_double: pre_euler3d_cpu_double.cpp
	$(CXX) $(OMPFLAGS) -O3 -Dblock_length=$(OMP_NUM_THREADS) pre_euler3d_cpu_double.cpp -o pre_euler3d_cpu_double


clean:
#	rm -f euler3d
	rm -f euler3d_cpu
	rm -f euler3d_cpu_offload
	rm -f euler3d_cpu_double
	rm -f pre_euler3d_cpu
	rm -f pre_euler3d_cpu_double
