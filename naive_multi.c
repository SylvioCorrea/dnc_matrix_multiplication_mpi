//13750.69 seconds for matrix_dim 8192

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "mmulti.h"

#define MATRIX_DIM (1<<10)

void main() {
    printf("Program start.\n");
    printf("Multiplication of 2 %dx%d matrices.\n", MATRIX_DIM, MATRIX_DIM);
    int *A;
    int *B;
    int *C;
    
    clock_t t;
    
    matrix_alloc(&A, MATRIX_DIM);
    matrix_alloc(&B, MATRIX_DIM);
    matrix_alloc(&C, MATRIX_DIM);
    
    matrix_init(A, MATRIX_DIM, 0);
    matrix_init(B, MATRIX_DIM, 2);
    
    printf("Multiplication start.\n");
    t = clock();
    
    naive_multi(A, B, C, MATRIX_DIM);
    
    t = clock() - t;
    printf("Multiplication done.\n");
    printf("Time taken: %.2f seconds\n", ((double)t)/CLOCKS_PER_SEC);
}
