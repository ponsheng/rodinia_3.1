include ../../common/common-general.mk

CCFLAGS += -fopenmp
OUTPUT = *.out

3D:
	$(CC) -o 3D $(CCFLAGS) 3D.c -lm 

clean:
	rm -f 3D $(OUTPUT)
