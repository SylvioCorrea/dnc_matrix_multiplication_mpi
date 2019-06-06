#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void naive_multi(int *A, int *B, int *C, int size) {
    int i, j, k;
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++) {
            C[i*size+j] = 0;
            for(k=0; k<size; k++) {
                C[i*size+j] += A[i*size+k] * B[k*size+j];
            }
        }
    }
}

void matrix_alloc(int **ptr, int size) {
    (*ptr) =  malloc(size*size*sizeof(int));
    if((*ptr)==NULL) {
        printf("malloc failed!\n");
        exit(1);
    }
}

void print_matrix(int *M, int size) {
    int i, j;
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++) {
            printf("%3d, ", M[i*size+j]);
        }
        printf("\n");
    }
}
/*
Params:
A and B= matrices of dimensions size_ab*size_ab being summed.
C = matrix of dimensions (size_ab*2)*(size_ab*2) that will store the result of this AND other calculations.
cl cc = line and colum of the top left element of the submatrix of C we are currently working with.
size_ab = size_ab*size_ab are the dimensions of both A and B
*/
void msum(int *A, int *B, int *C, int cl, int cc, int size_ab) {
    int size_c = size_ab*2;
    int i,j;
    for(i=0; i<size_ab; i++) {
        for(j=0; j<size_ab; j++) {
            //Clc = Alc + Blc
            C[(cl+i)*size_c + cc+j] = A[i*size_ab + j]+B[i*size_ab + j];
        }
    }
}



/*
Params:
A = pointer to the original matrix A being multiplied.
B = pointer to the original matrix B being multiplied.
al ac = line and colum indicating the top left element of the submatrix of A 
in the recursion.
bl bc = line and colum indicating the top left element of the submatrix of B 
in the recursion.
C = pointer to the matrix of size s where results of this recursion will be 
stored.
s = size of the submatrixes of A and B at this point in the recursion. Also 
the whole size of the result matrix C which was allocated in the previous 
node of the recursion tree.
size = size*size are the original dimensions of both A and B. Is used to 
correctly traverse the original matrices lines during the recursion.
*/
void mmulti(int *A, int *B,
            int al, int ac,
            int bl, int bc,
            int *C, int s, int size) {
    //printf("s, size: %d, %d\n", s, size);
    if(s==2) {
        //printf("test\n");
        //Regular multiplication for small 2x2 matrix
        //printf("al = %d, ac = %d, bl = %d, bc = %d\n", al, ac, bl, bc);
        
        int a11 = A[al*size + ac];
        int a12 = A[al*size + ac+1];
        int a21 = A[(al+1)*size + ac];
        int a22 = A[(al+1)*size + ac+1];
        //printf("%d,%d\n%d,%d\n\n", a11, a12, a21, a22);
        
        //printf("test2\n");
        int b11 = B[bl*size + bc];
        int b12 = B[bl*size + bc+1];
        int b21 = B[(bl+1)*size + bc];
        int b22 = B[(bl+1)*size + bc+1];
        //printf("%d,%d\n%d,%d\n\n", b11, b12, b21, b22);
        
        //printf("test3\n");
        C[0] = a11*b11 + a12*b21;
        //printf("test4\n");
        C[1] = a11*b12 + a12*b22;
        //printf("test5\n");
        C[2] = a21*b11 + a22*b21;
        //printf("test6\n");
        C[3] = a21*b12 + a22*b22;
        //printf("test7\n");
        return;
    }
    //printf("despair\n");
    int half = s/2;
    
    int *tempM1;
    int *tempM2;
    //Pass pointer to the pointer so that the function can allocate.
    matrix_alloc(&tempM1, half);
    matrix_alloc(&tempM2, half);
    
    mmulti(A, B, al, ac, bl, bc, tempM1, half, size); //A11B11
    mmulti(A, B, al, ac+half, bl+half, bc, tempM2, half, size); //A12B21
    msum(tempM1, tempM2, C, 0, 0, half); //A11B11+A12B21=C11
    //printf("despair1\n");
    
    mmulti(A, B, al, ac, bl, bc+half, tempM1, half, size); //A11B12
    mmulti(A, B, al, ac+half, bl+half, bc+half, tempM2, half, size); //A12B22
    msum(tempM1, tempM2, C, 0, half, half); //A11B12+A12B22=2C12
    //printf("despair2\n");
    
    mmulti(A, B, al+half, ac, bl, bc, tempM1, half, size); //A21B11
    mmulti(A, B, al, ac+half, bl+half, bc, tempM2, half, size); //A12B21
    msum(tempM1, tempM2, C, half, 0, half); //A21B11+A22B21=2C21
    //printf("despair3\n");
    
    mmulti(A, B, al+half, ac, bl, bc+half, tempM1, half, size); //A21B12
    mmulti(A, B, al+half, ac+half, bl+half, bc+half, tempM2, half, size); //A22B22
    msum(tempM1, tempM2, C, half, half, half); //A21B12+A22B22=2C22
    //printf("despair4\n");
    
    free(tempM1);
    free(tempM2);
}

/*
void main(int argc, char **argv) {
    int M1[] = { 1,  2,  3,  4,
                 5,  6,  7,  8,
                 9, 10, 11, 12,
                13, 14, 15, 16};
    
    int m_size = 4;
    
    int M2[m_size*m_size];
    int i;
    for(i=0; i<m_size*m_size; i++) {
        M2[i] = M1[i]+1;
    }
    
    int C[m_size*m_size];
    
    print_matrix(M1, m_size);
    printf("\n");
    print_matrix(M2, m_size);
    printf("\n");
    naive_multi(M1, M2, C, m_size);
    print_matrix(C, m_size);
    printf("\n");
    
    int D[m_size*m_size*4];
    for(i=0; i<m_size*m_size*4; i++) {
        D[i] = 0;
    }
    
    msum(M1, M2, D, 0, 0, m_size);
    print_matrix(D, m_size*2);
    printf("\n");
    msum(M1, M2, D, 0, 0+m_size, m_size);
    print_matrix(D, m_size*2);
    printf("\n");
    msum(M1, M2, D, 0+m_size, 0, m_size);
    print_matrix(D, m_size*2);
    printf("\n");
    msum(M1, M2, D, 0+m_size, 0+m_size, m_size);
    print_matrix(D, m_size*2);
    printf("\n");
    
    for(i=0; i<m_size*m_size*4; i++) {
        D[i] = 0;
    }
    int size2 = 2;
    int M3[size2*size2];
    int M4[size2*size2];
    int E[size2*size2];
    for(i=0; i<size2*size2; i++) {
        M3[i] = M1[i];
        M4[i] = M2[i];
        E[i] = 0;
    }
    print_matrix(M3, size2);
    printf("\n");
    print_matrix(M4, size2);
    printf("\n");
    
    mmulti(M3, M4, 0, 0, 0, 0, E, size2, size2);
    print_matrix(E, size2);
    printf("\n");
    
    naive_multi(M3, M4, E, size2);
    print_matrix(E, size2);
    printf("\n");
    
    mmulti(M1, M2, 0, 0, 0, 0, C, m_size, m_size);
    print_matrix(C, m_size);
    printf("\n");
    
}
*/
