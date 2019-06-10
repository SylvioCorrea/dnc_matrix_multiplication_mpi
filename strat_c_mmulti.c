/*
Divide and conquer strategy where each process uses a recursive function that
keeps one unity of the divided job for itself to compute, sending the other
pieces to different processes. This prevents subutilization of hardware where
some processes sit idle most of the time waiting for processes further down
on the computation tree to finish their jobs before joining their results.

The computation tree of this divide and conquer strategy has
branching factor 7. The computation tree will expand unevenly. The
required number of processes for n divisions is given by the recurrence
        { 1, n=0
procs(n){
        { 1 + sum(7*proc(k)) ,0<=k<=n-1, n>1

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include "mmulti.h"

//Dimensions of matrices being multiplied
//will be 2^MATRIX_DIM_EXP.
#define MATRIX_DIM_EXP 3

//This number defines how many divisions should be
//performed before conquering.
#define N_OF_DIVISIONS 2

//Number of lines/colums of matrices being multiplied.
#define MATRIX_DIM (1<<MATRIX_DIM_EXP)

//Matrices with this number of lines/colums should be conquered
#define DELTA (1<<(MATRIX_DIM_EXP - N_OF_DIVISIONS))


//====================================================================
//Struct used to keep track of important variables of the submatrices being worked on
//during the the recursive computation in the function process_recursion()
typedef struct {
    int al; //Row of the top left element of the current submatrix of A
    int ac; //Colum of the top left element of the current submatrix of A
    int bl; //Row of the top left element of the current submatrix of B
    int bc; //Colum of the top left element of the current submatrix of B
    int dim; //Number of rows/colums of the current submatrices
    int division_n; //How many divisions have already ocurred at this point
} recursion_struct;
//====================================================================


int my_rank;
int proc_n;
MPI_Status status;
int *A;
int *B;



void print_rec_str(recursion_struct *rec_str) {
    printf("al: %d, ac: %d, bl: %d, bc: %d\ndim: %d, division_n: %d\n\n",
           rec_str->al, rec_str->ac, rec_str->bl, rec_str->bc,
           rec_str->dim, rec_str->division_n);
}

//Calculates the number of required processes to perform
//N_OF_DIVISIONS using this algorithm.
int required_procs() {
    return required_procs_aux(N_OF_DIVISIONS);
}
int required_procs_aux(int n) {
    if(n==0)
        return 1;
    
    int i;
    int res = 1;
    for(i=0; i<n; i++) {
        res += 7*required_procs_aux(i);
    }
    return res;
}

//Recursive function executed by every process.
//Implements the divide and conquer method of matrix multiplication,
//but ensures one of the eight pieces of the division stays with the dividing process.
void process_recursion(recursion_struct *rec_ptr, int *C) {
    
    if(rec_ptr->dim <= DELTA) { //conquer
        printf("[%d] conquering.\n", my_rank);
        mmulti(A, B,
               rec_ptr->al, rec_ptr->ac,
               rec_ptr->bl, rec_ptr->bc,
               C, rec_ptr->dim, MATRIX_DIM);
        return;
    
    
    
    }// else divide
    
    printf("[%d] dividing\n", my_rank);
    
    int half = rec_ptr->dim/2;
    int new_division = rec_ptr->division_n + 1;
    int i;
    int sum = 0;
    //First children of the recursion can be calculated as
    //summation(7^k) + myrank*7, where 0<=k<=current number
    //of divides already performed at this point.
    for(i=0; i <= rec_ptr->division_n; i++) {
        sum += simple_pow(7, i);
    }
    int child1 = sum + my_rank*7;
    int child2 = child1+1;
    int child3 = child2+1;
    int child4 = child3+1;
    int child5 = child4+1;
    int child6 = child5+1;
    int child7 = child6+1;
    
    int al = rec_ptr->al;
    int ac = rec_ptr->ac;
    int bl = rec_ptr->bl;
    int bc = rec_ptr->bc;
    
    recursion_struct A11B11_buffer = {al,      ac,      bl,      bc,      half, new_division};
    recursion_struct A12B21_buffer = {al,      ac+half, bl+half, bc,      half, new_division};
    recursion_struct A11B12_buffer = {al,      ac,      bl,      bc+half, half, new_division};
    recursion_struct A12B22_buffer = {al,      ac+half, bl+half, bc+half, half, new_division};
    recursion_struct A21B11_buffer = {al+half, ac,      bl,      bc,      half, new_division};
    recursion_struct A22B21_buffer = {al+half, ac+half, bl+half, bc,      half, new_division};
    recursion_struct A21B12_buffer = {al+half, ac,      bl,      bc+half, half, new_division};
    recursion_struct A22B22_buffer = {al+half, ac+half, bl+half, bc+half, half, new_division};
    
    printf("[%d]: children: %d, %d, %d, %d, %d, %d, %d\n",
           my_rank, child1, child2, child3, child4, child5, child6, child7);
    
    //Sending jobs to other processes
    MPI_Send (&A22B22_buffer, sizeof(recursion_struct), MPI_BYTE, child7, 1, MPI_COMM_WORLD);
    MPI_Send (&A12B21_buffer, sizeof(recursion_struct), MPI_BYTE, child1, 1, MPI_COMM_WORLD);
    MPI_Send (&A11B12_buffer, sizeof(recursion_struct), MPI_BYTE, child2, 1, MPI_COMM_WORLD);
    MPI_Send (&A12B22_buffer, sizeof(recursion_struct), MPI_BYTE, child3, 1, MPI_COMM_WORLD);
    MPI_Send (&A21B11_buffer, sizeof(recursion_struct), MPI_BYTE, child4, 1, MPI_COMM_WORLD);
    MPI_Send (&A22B21_buffer, sizeof(recursion_struct), MPI_BYTE, child5, 1, MPI_COMM_WORLD);
    MPI_Send (&A21B12_buffer, sizeof(recursion_struct), MPI_BYTE, child6, 1, MPI_COMM_WORLD);
    //MPI_Send (&A22B22_buffer, sizeof(recursion_struct), MPI_BYTE, child7, 1, MPI_COMM_WORLD);
    
    //The process still needs to take care of its own multiplication before joining results.
    //Lets allocate the matrices that will hold the results to be summed;
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
    
    //The recursion will give us A11B11.
    process_recursion(&A11B11_buffer, A11B11);
    //The other processes will give us all other matrices.
    //printf("[%d]Expecting matrices of dim %d\n", my_rank, half);
    MPI_Recv (A12B21, half*half, MPI_INT, child1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child1[%d].\n", my_rank, child1);
    MPI_Recv (A11B12, half*half, MPI_INT, child2, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child2[%d].\n", my_rank, child2);
    MPI_Recv (A12B22, half*half, MPI_INT, child3, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child3[%d].\n", my_rank, child3);
    MPI_Recv (A21B11, half*half, MPI_INT, child4, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child4[%d].\n", my_rank, child4);
    MPI_Recv (A22B21, half*half, MPI_INT, child5, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child5[%d].\n", my_rank, child5);
    MPI_Recv (A21B12, half*half, MPI_INT, child6, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child6[%d].\n", my_rank, child6);
    MPI_Recv (A22B22, half*half, MPI_INT, child7, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    printf("[%d] receives from child7[%d].\n", my_rank, child7);
    
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



void main(int argc, char** argv) {
    
    //A and B are square matrices of same size
    matrix_alloc(&A, MATRIX_DIM);
    matrix_alloc(&B, MATRIX_DIM);
    
    //C points to the resulting matrix
    int *C;
    
	//These numbers store the current line and colum of the top left elements
	//of the submatrices of A and B we are working with in the current level
	//of the recursion. This will allow use of the original A and B matrices
	//in multiplication stages of the computation avoiding having to actually
	//allocate and copy the submatrices.
	int al, ac, bl, bc;
	al = ac = bl = bc = 0;
	
	//Dimensions of the resulting matrix to be returned to father processes.
	int C_dim;
	
	//Message buffer for the above numbers.
	recursion_struct rec_str;
	
	int half;
	int father;
	int child1, child2, child3, child4,
	    child5, child6, child7, child8;
	    
	//For execution time measuring.
	double t1, t2;
	
	int i;
	
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);
    
    
    //===========================================
    //Test if the number of processes is correct
    int required_procs = required_procs();
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
    
    
    matrix_init(A, MATRIX_DIM, 0);
    matrix_init(B, MATRIX_DIM, 2);
    
    printf("[%d]start\n", my_rank);
    
    
    
    if ( my_rank != 0 ) { //not-root
        //Receive some division of the job
        MPI_Recv(&rec_str, sizeof(recursion_struct), MPI_BYTE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        father = status.MPI_SOURCE;
        printf("[%d] received from %d. Current dimensions of the matrices = %d\n",
               my_rank, father, rec_str.dim);
        print_rec_str(&rec_str);
        C_dim = rec_str.dim;
        
    } else { //root
        printf("Dimensions of the matrices: %dx%d\n", MATRIX_DIM, MATRIX_DIM);
        printf("Conquering point: %d\n", DELTA);
        printf("Number of consecutive divisions to be performed before conquering: %d.\n", N_OF_DIVISIONS);
        printf("number of processes: %d\n\n", proc_n);
        //printf("matrix A:\n");
        //print_matrix(A, MATRIX_DIM);
        //printf("\nmatrix B:\n");
        //print_matrix(B, MATRIX_DIM);
        rec_str.al = 0;
        rec_str.ac = 0;
        rec_str.bl = 0;
        rec_str.bc = 0;
        rec_str.dim = MATRIX_DIM;
        rec_str.division_n = 0;
        C_dim = MATRIX_DIM;
        t1 = MPI_Wtime();
    }
    
    
    
    //Start computation.
    //Allocate matrix to hold results.
    matrix_alloc(&C, C_dim);
    process_recursion(&rec_str, C);
    
    
    
    if(my_rank!=0) { //not-root
        //Non-root nodes still need to send back their results
        //printf("[%d]Sending back matrix of dim %d\n", my_rank, C_dim);
        //print_matrix(C, C_dim);
        MPI_Send(C, C_dim*C_dim, MPI_INT, father, 1, MPI_COMM_WORLD);
        
    } else { //root
        t2 = MPI_Wtime();
        print_matrix(C, MATRIX_DIM);
        printf("Time taken: %.2f\n", t2-t1);
    }
    
    free(C);
    
    printf("[%d]done.\n", my_rank);
    
    MPI_Finalize();
}
