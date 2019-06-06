#include <stdlib.h>
#include <stdio.h>
#include <string.h>


void naive_multi(int *A, int *B, int *C, int size);
void matrix_alloc(int **ptr, int size);
void print_matrix(int *M, int size);
void msum(int *A, int *B, int *C, int cl, int cc, int size_ab);
void mmulti(int *A, int *B,
            int al, int ac,
            int bl, int bc,
            int *C, int s, int size);