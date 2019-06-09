//Example use:
//ladrun -np 73 mpi_mmulti

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "mmulti.h"

/*
The computation tree of this divide and conquer strategy has
branching factor 8. The computation tree will have height equal
to the number of divisions needed + 1. The program requires this
tree to be full, that is, the number of processes must be equal
to sum(8^k), 0<=k<=tree height.

Examples:
    height  divisions   required number of processes
    1       0           1
    2       1           9
    3       2           73
    4       3           585
*/

//Dimensions of matrices being multiplied
//will be 2^MATRIX_DIM_EXP.
#define MATRIX_DIM_EXP 13

//This number defines how many divisions should be
//performed before conquering.
#define N_OF_DIVISIONS 2

//Number of lines/colums of matrices being multiplied.
#define MATRIX_DIM (1<<MATRIX_DIM_EXP)

//Matrices with this number of lines/colums should be conquered
#define DELTA (1<<(MATRIX_DIM_EXP - N_OF_DIVISIONS))


void main(int argc, char** argv) {
    
    //A and B are square matrices of same size
    int *A;
    int *B;
    
    //C points to the resulting matrix
    int *C;
    
	//These numbers store the current line and colum of the top left elements
	//of the submatrices of A and B we are working with in the current level
	//of the recursion. This will allow use of the original A and B matrices
	//in multiplication stages of the computation avoiding having to actually
	//allocate and copy the submatrices.
	int al, ac, bl, bc;
	al = ac = bl = bc = 0;
	
	//Current dimensions of the matrices we are working with;
	int curr_dim;
	
	//Message buffer for the above numbers.
	int div_buffer[5] = {al, ac, bl, bc, MATRIX_DIM};
	
	int half;
	int father;
	int child1, child2, child3, child4,
	    child5, child6, child7, child8;
	    
	//For execution time measuring.
	double t1, t2;
	
	int i;
	
	int my_rank; //Process id.
	int proc_n; //Total number of processes
	MPI_Status status;
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);
    
    
    //===========================================
    //Test if the number of processes is correct
    int required_procs = 0;
    for(i=0; i<=N_OF_DIVISIONS; i++) {
        required_procs += simple_pow(8, i);
    }
    if( proc_n != required_procs ) {
        if(my_rank == 0) {
            int req_procs = 
            printf("Error. required number of processes to perform %d divisions is %d.\n",
                   N_OF_DIVISIONS, required_procs);
            printf("Number of processes given by the user: %d.\n", proc_n);
            printf("Aborting.\n");
        }
        exit(1);
    }
    //============================================
    //Test passed
    
    
    matrix_alloc(&A, MATRIX_DIM);
    matrix_alloc(&B, MATRIX_DIM);
    matrix_init(A, MATRIX_DIM, 0);
    matrix_init(B, MATRIX_DIM, 2);
    
    printf("[%d]start\n", my_rank);
    
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
        father = status.MPI_SOURCE;
        printf("[%d] received from %d. curr_dim = %d\n", my_rank, status.MPI_SOURCE, curr_dim);

        
    } else { //root
        printf("Dimensions of the matrices: %dx%d\n", MATRIX_DIM, MATRIX_DIM);
        printf("conquering point: %d\n", DELTA);
        printf("Number of consecutive divisions to be performed before conquering: %d", N_OF_DIVISIONS);
        printf("number of processes: %d\n\n", proc_n);
        //printf("matrix A:\n");
        //print_matrix(A, MATRIX_DIM);
        //printf("\nmatrix B:\n");
        //print_matrix(B, MATRIX_DIM);
        
        curr_dim = MATRIX_DIM;
        t1 = MPI_Wtime();
    }
    
    
    //Now that curr_dim is known we can allocate C.
    matrix_alloc(&C, curr_dim);
    
    
    if (curr_dim <= DELTA) { //conquer
        printf("[%d]: curr_dim = %d. Conquering.\n", my_rank, curr_dim);
        mmulti(A, B, div_buffer[0], div_buffer[1],
               div_buffer[2], div_buffer[3],
               C, curr_dim, MATRIX_DIM);
        printf("[%d]: mmulti done.\n", my_rank);
        
        
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
        
        matrix_alloc(&A11B11, half);
        matrix_alloc(&A12B21, half);
        matrix_alloc(&A11B12, half);
        matrix_alloc(&A12B22, half);
        matrix_alloc(&A21B11, half);
        matrix_alloc(&A22B21, half);
        matrix_alloc(&A21B12, half);
        matrix_alloc(&A22B22, half);
        
        //Receive the results
        MPI_Recv (A11B11, half*half, MPI_INT, child1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child1[%d].\n", my_rank, child1);
        MPI_Recv (A12B21, half*half, MPI_INT, child2, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child2[%d].\n", my_rank, child2);
        MPI_Recv (A11B12, half*half, MPI_INT, child3, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child3[%d].\n", my_rank, child3);
        MPI_Recv (A12B22, half*half, MPI_INT, child4, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child4[%d].\n", my_rank, child4);
        MPI_Recv (A21B11, half*half, MPI_INT, child5, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child5[%d].\n", my_rank, child5);
        MPI_Recv (A22B21, half*half, MPI_INT, child6, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child6[%d].\n", my_rank, child6);
        MPI_Recv (A21B12, half*half, MPI_INT, child7, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child7[%d].\n", my_rank, child7);
        MPI_Recv (A22B22, half*half, MPI_INT, child8, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        printf("[%d] receives from child8[%d].\n", my_rank, child8);
        
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
        MPI_Send(C, curr_dim*curr_dim, MPI_INT, father, 1, MPI_COMM_WORLD);
        
    
    } else { //root
        //printf("Root results:\n");
        //print_matrix(C, curr_dim);
        t2 = MPI_Wtime();
        printf("Multiplication done. Time taken: %.2f seconds", t2-t1);
    }
    
    free(C);
    printf("[%d] done\n", my_rank);
    MPI_Finalize();
}
