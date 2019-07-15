include ../../common/common-general.mk
ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

CFLAGS += -fopenmp

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) $(CFLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -lm -o heartwall

# compile main function file into object (binary)
main.o: main.c define.c kernel.c
	$(CC) $(CFLAGS) $(OUTPUT) main.c -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall
