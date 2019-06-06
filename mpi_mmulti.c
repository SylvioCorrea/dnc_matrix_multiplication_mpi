/*
Modelos:
A: divisao e conquista pura. No máximo um processo por core
    (ou 16 por nodo com hyperthread).
B: inflar propositalmente o número de processos para mais
    de um por core. O SO vai fazer o balanceamento de carga
    ao deixar os processos menos trabalhosos em espera
C: modelo do artigo onde o pai divide o trabalho com ele
    mesmo mais um filho

IMPORTANT: expected number of process to run this
implementation: 2^(TREE_HEIGHT) - 1
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "mmulti.h"

//MATRIX_DIM **must** be a power of 2.
#define MATRIX_DIM 2

//Once division makes matrices of dimensions DELTA,
//the receiving process must conquer. Must also be
//a power of 2.
//2 is the minimum delta.
#define DELTA 2

//Initializes a matrix containing sequential numbers between 0+offset and 8+offset.
void matrix_init(int *M, int size, int offset) {
    int i,j,n;
    n=0;
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++) {
            M[i*size + j] = n+offset;
            n = (n+1)%9;
        }
    }
}

void main(int argc, char** argv) {
    
    //A and B are square matrices of same size
    int A[MATRIX_DIM*MATRIX_DIM];
    int B[MATRIX_DIM*MATRIX_DIM];
    
    //C points to the resulting matrix
    int *C;
    
	//These numbers store the current line and colum of the top left elements
	//of the submatrices of A and B we are working with in the current level
	//of the recursion. This will allow use of the original A and B matrices
	//in multiplication stages of the computation avoiding having to actually
	//allocate and copy the submatrices.
	int al, ac, bl, bc;
	
	//Current dimensions of the matrices we are working with;
	int curr_dim;
	
	//Message buffer for the above numbers.
	int div_buffer[5];
	
	int half;
	int child1, child2, child3, child4,
	    child5, child6, child7, child8;
	    
	int i;
	
	int my_rank; //Process id.
	int proc_n; //Total number of processes
	MPI_Status status;
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);
    
    matrix_init(A, MATRIX_DIM, 0);
    matrix_init(B, MATRIX_DIM, 2);
    
    if ( my_rank != 0 ) { //not root
        
        //div_buffer receives indexes for the submatrices of A and B of the division
        //and also the size of the submatrices dimensions (same dimensions for both).
        //Receive some division of the job
        MPI_Recv(div_buffer, 5, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        al = div_buffer[0];
        ac = div_buffer[1];
        bl = div_buffer[2];
        bc = div_buffer[3];
        curr_dim = div_buffer[4];
        
        printf("[%d] received from %d.\ncurr_dim = %d\n", my_rank, status.MPI_SOURCE, curr_dim);

        
    } else { //root
        printf("Dimensions of the matrices: %dx%d\n", MATRIX_DIM, MATRIX_DIM);
        printf("conquering point: %d\n", DELTA);
        printf("number of processes: %d\n\n", proc_n);
        curr_dim = MATRIX_DIM;
    }
    
    
    //Now that curr_dim is known we can allocate C.
    matrix_alloc(C, curr_dim);
    
    
    if (curr_dim <= DELTA) { //conquer
        printf("[%d]: curr_dim = %d. Conquering.\n", curr_dim, my_rank);
        matrix_alloc(C, curr_dim);
        mmulti(A, B, div_buffer[0], div_buffer[1],
               div_buffer[2], div_buffer[3],
               C, curr_dim, MATRIX_DIM);
        
        
        
    } else { //divide
        printf("[%d] dividing\n", my_rank);
        
        half = curr_dim/2;
        
        child1 = my_rank*8 + 1;
        child2 = child1+1;
        child3 = child2+1;
        child4 = child3+1;
        child5 = child4+1;
        child6 = child5+1;
        child7 = child6+1;
        child8 = child7+1;
        
        int A11B11_buffer[5] = {al,      ac,      bl,      bc,      half};
        int A12B21_buffer[5] = {al,      ac+half, bl+half, bc,      half};
        int A11B12_buffer[5] = {al,      ac,      bl,      bc+half, half};
        int A12B22_buffer[5] = {al,      ac+half, bl+half, bc+half, half};
        int A21B11_buffer[5] = {al+half, ac,      bl,      bc,      half};
        int A22B21_buffer[5] = {al+half, ac+half, bl+half, bc,      half};
        int A21B12_buffer[5] = {al+half, ac,      bl,      bc+half, half};
        int A22B22_buffer[5] = {al+half, ac+half, bl+half, bc+half, half};
        
        MPI_Send (A11B11_buffer, 5, MPI_INT, child1, 1, MPI_COMM_WORLD);
        MPI_Send (A12B21_buffer, 5, MPI_INT, child2, 1, MPI_COMM_WORLD);
        MPI_Send (A11B12_buffer, 5, MPI_INT, child3, 1, MPI_COMM_WORLD);
        MPI_Send (A12B22_buffer, 5, MPI_INT, child4, 1, MPI_COMM_WORLD);
        MPI_Send (A21B11_buffer, 5, MPI_INT, child5, 1, MPI_COMM_WORLD);
        MPI_Send (A22B21_buffer, 5, MPI_INT, child6, 1, MPI_COMM_WORLD);
        MPI_Send (A21B12_buffer, 5, MPI_INT, child7, 1, MPI_COMM_WORLD);
        MPI_Send (A22B22_buffer, 5, MPI_INT, child8, 1, MPI_COMM_WORLD);
        
        
        
        //hope for the best
        
        
        
        //Time to receive
        
        //8 matrix multiplications will be performed.
        //Declare and allocate 8 matrices. Note that matrices
        //names tell which multiplications they will store.
        int *A11B11;
        int *A12B21;
        int *A11B12;
        int *A12B22;
        int *A21B11;
        int *A22B21;
        int *A21B12;
        int *A22B22;
        
        //One extra matrix will store the final results
        //of the sums of the above multiplications
        int *C;
        
        matrix_alloc(A11B11, half);
        matrix_alloc(A12B21, half);
        matrix_alloc(A11B12, half);
        matrix_alloc(A12B22, half);
        matrix_alloc(A21B11, half);
        matrix_alloc(A22B21, half);
        matrix_alloc(A21B12, half);
        matrix_alloc(A22B22, half);
        
        matrix_alloc(C, curr_dim);
        
        //Divide the work
        MPI_Recv (A11B11, half*half, MPI_INT, child1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child1.\n", my_rank);
        MPI_Recv (A12B21, half*half, MPI_INT, child2, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child2.\n", my_rank);
        MPI_Recv (A11B12, half*half, MPI_INT, child3, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child3.\n", my_rank);
        MPI_Recv (A12B22, half*half, MPI_INT, child4, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child4.\n", my_rank);
        MPI_Recv (A21B11, half*half, MPI_INT, child5, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child5.\n", my_rank);
        MPI_Recv (A22B21, half*half, MPI_INT, child6, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child6.\n", my_rank);
        MPI_Recv (A21B12, half*half, MPI_INT, child7, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child7.\n", my_rank);
        MPI_Recv (A22B22, half*half, MPI_INT, child8, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child8.\n", my_rank);
        
        //Time to sum the multiplication results. Each sum will be stored in
        //one quarter of the result matrix C.
        msum(A11B11, A12B21, C,    0,    0, half); //C11
        msum(A11B12, A12B22, C,    0, half, half); //C12
        msum(A21B11, A22B21, C, half,    0, half); //C21
        msum(A21B12, A22B22, C, half, half, half); //C22
        
        free(A11B11);
        free(A12B21);
        free(A11B12);
        free(A12B22);
        free(A21B11);
        free(A22B21);
        free(A21B12);
        free(A22B22);
    }

    // Send back to father
    if ( my_rank !=0 ) { //not root
        MPI_Send(C, curr_dim*curr_dim, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD);
        
    
    } else { //root
        printf("Root results:\n");
        print_matrix(C, curr_dim);
    }
    
    free(C);
    printf("[%d] done\n", my_rank);
    MPI_Finalize();
}
