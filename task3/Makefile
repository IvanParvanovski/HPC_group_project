CC = gcc-16
MPICC = mpicc

CFLAGS = -O2 -Wall
LDFLAGS = -lm

jacobi_mpi: jacobi_mpi.c system_gen.c system_gen.h
	$(MPICC) $(CFLAGS) -o $@ jacobi_mpi.c system_gen.c $(LDFLAGS)

clean:
	rm -f jacobi_mpi