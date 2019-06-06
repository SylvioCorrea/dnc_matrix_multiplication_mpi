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


#include "headers_mmulti.h"

//MATRIX_DIM must be a power of 2.
#define MATRIX_DIM 256

//Once division makes matrices of dimensions DELTA,
//the receiving process must conquer. Must also be
//a power of 2
#define DELTA 2

//Initializes a matrix containing numbers between 0+offset and 9+offset.
void matrix_init(int *M, int size, int offset) {
    int i,j,n;
    n=0;
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++) {
            M[i*size + j] = n+offset;
            n = (n+1)%10;
        }
    }
}

void make_arr(int arr[], int size) {
    int i;
    for(i=0; i<size; i++) {
        arr[i] = size-i;
    }
}

void copy_arr(int arr1[], int arr2[], int size) {
    int i;
    for(i=0; i<size; i++) {
        arr2[i] = arr1[i];
    }
}

void bubblesort(int arr[], int size) {
    int i, temp;
    int ordered = 0;
    while(!ordered) {
        ordered = 1;
        for(i=0; i<size-1; i++) {
            if(arr[i]>arr[i+1]) {
                temp = arr[i];
                arr[i] = arr[i+1];
                arr[i+1] = temp;
                ordered = 0;
            }
        }
    }
}

void merge(int arr1[], int arr2[], int size, int res[]) {
    int size1 = size/2;
    int size2 = size-size1;
    int i = 0;
    int j = 0;
    int k = 0;
    while(j<size1 && k<size2) {
        if(arr1[j]<arr2[k]) {
            res[i] = arr1[j];
            j++;
        } else {
            res[i] = arr2[k];
            k++;
        }
        i++;
    }
    
    if(j<size1) {
        for(j; j<size1; j++) {
            res[i] = arr1[j];
            i++;
        }
    } else {
        for(k; k<size2; k++) {
            res[i] = arr2[k];
            i++;
        }
    }   
}

int calc_father(int child) {
    if(child%2 == 0) {
        return (child-2)/2;
    }
    return (child-1)/2;
}

void main(int argc, char** argv) {
    
    //A and B are square matrices of same size
    int A[MATRIX_DIM*MATRIX_DIM];
    int B[MATRIX_DIM*MATRIX_DIM];
    
    int i;
	int my_rank;       //Process id.
	int proc_n;        //Total number of processes
	int al, ac, bl, bc;
	int curr_dim;
	int half;
	int father, child1, child2, child3, child4;
	MPI_Status status; // estrutura que guarda o estado de retorno          
    MPI_Init(&argc , &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);  // pega pega o numero do processo atual (rank)
    MPI_Comm_size(MPI_COMM_WORLD, &proc_n);
    
    matrix_init(A, MATRIX_DIM, 0);
    matrix_init(B, MATRIX_DIM, 2);
    
    
    
    
    if ( my_rank != 0 ) { //not root
        //div_buffer contains indexes for the submatrices of A and B of the division
        //and also the size of the submatrices dimensions (same dimensions for both).
        int div_buffer[5];
        //Receive some division of the job
        MPI_Recv(div_buffer, 5, MPI_INT, MPI_ANY_SOURCE, DIVISION, MPI_COMM_WORLD, &status);
        father = status.MPI_SOURCE;
        al = div_buffer[0];
        ac = div_buffer[1];
        bl = div_buffer[2];
        bc = div_buffer[3];
        curr_dim = div_buffer[4];
        printf("[%d] received from %d.\ncurr_dim = %d\n", my_rank, father, curr_dim);
        
        
        
    } else { //root
        printf("Dimensions of the matrices: %dx%d\n", MATRIX_DIM, MATRIX_DIM);
        printf("conquering point: %d\n", DELTA);
        printf("number of processes: %d\n\n", proc_n);
        curr_dim = MATRIX_DIM;
    }
    
    
    if (curr_dim <= DELTA) { //conquer
        printf("[%d] conquering\n", my_rank);
        int *C;
        matrix_alloc(C, curr_dim);
        mmulti(A, B, div_buffer[0], div_buffer[1],
               div_buffer[2], div_buffer[3],
               C, curr_dim, MATRIX_DIM);
        //Send back results
        MPI_Send(C, curr_dim*curr_dim, MPI_INT,
                 father, 1, MPI_COMM_WORLD);
        free(C);
        
        
        
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
        
        
        //receive
        int *A11B11;
        int *A12B21;
        int *A11B12;
        int *A12B22;
        int *A21B11;
        int *A22B21;
        int *A21B12;
        int *A22B22;
        
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
        
        msum()
        
        free(A11B11);
        free(A12B21);
        free(A11B12);
        free(A12B22);
        free(A21B11);
        free(A22B21);
        free(A21B12);
        free(A22B22);
    }

    // mando para o pai
    if ( my_rank !=0 ) {
        MPI_Send(res, curr_dim, MPI_INT, status.MPI_SOURCE,
                 1, MPI_COMM_WORLD);  // tenho pai, retorno vetor ordenado pra ele
        
    }
    else {
        printf("Root results:\n");
        for(i=0; i<ARR_SIZE; i++) {
            printf("%d, ", res[i]);
        }
        printf("\n");
    }
    
    free(res);
    printf("[%d] done\n", my_rank);
    MPI_Finalize();
}
