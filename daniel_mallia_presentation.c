/*****************************************************************************
  Title:          daniel_mallia_presentation.c
  Author:         Daniel Mallia
  Created on:     November 2, 2021
  Description:    Uses the simulated annealing algorithm, as described by
                  Professor Weiss in Chapter 8 of the lecture notes, to
                  determine a (hopefully optimal!) solution to the room
                  assignment problem given an nxn incompatibility matrix for
                  n students and only n/2 rooms. The best solution of the p
                  processes is the one printed.
  Purpose:        This program demonstrates usage of the simulated annealing
                  algorithm, a Monte Carlo method, applied to the room
                  assignment problem. The problem consists of minimizing cost
                  for room assignments of n students into n/2 rooms (each room
                  can hold 2 students), where cost is defined as some sum of
                  the student "incompatibilities" as given in a nxn matrix.
                  As the algorithm starts with a random assignment and
                  conducts a stochastic solution search, this program makes
                  use of parallelism to conduct multiple of these searches in
                  parallel, increasing the chance that a good, if not optimal,
                  solution is found. Ultimately, the best solution, i.e. the
                  one that minimizes the cost function, found by any of the
                  processes involved is printed.
                  PLEASE NOTE: The handling of matrices in this program is
                  inspired by the mpi_floyd.c program by Professor Weiss,
                  presented in Chapter 5 of the CSCI 79524 lecture notes.
  Usage:          mpirun -np <N> room <filename> [seed]
                  where N is a positive integer, filename is the name of the
                  incompatibility matrix file, and seed is an optional integer
                  which should be used to seed the random number generator for
                  repeatable behavior. 
  Build with:     mpicc -Wall -g -o room daniel_mallia_presentation.c -lm
  Modifications:  None at this time.
  NOTE:           This program expects the incompatibility matrix values to be
                  in the range [0,10] as described by Quinn in Chapter 10 of
                  Parallel Programming in C with MPI and OpenMP. 
*****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ROOT 0 /* Good practice borrowed from estimate_pi.c by Prof. Weiss */
#define MALLOC_ERROR 1
#define DEBUG 0
#define LAG_TABLE_SIZE 64

/*
  PLEASE NOTE: this function is borrowed from page 488 of Parallel Programming
  in C with MPI and OpenMP by Michael J. Quinn, but it has been slightly
  modified for output on stderr instead of stdout (thus using fprintf instead
  of printf, and flushing stderr instead of stdout). It is a succint and
  convenient way of handling error output from only the root process (0) and
  ensuring that all processes exit cleanly upon an error.

  @param id             The id of the calling process.
  @param error_message  The error message that should be printed by root.
  @Pre-Condition        An error has occurred requiring printing an error
                        message and quitting.
  @Post-Condition       The error message has been printed to stderr by root
                        and all calling processes have shut down MPI and
                        terminated.
*/
void terminate(int id, char* error_message) {
    if(!id) {
        fprintf(stderr, "Error: %s\n", error_message);
        fflush(stderr);
    }
    MPI_Finalize();
    exit(-1);
}

/*
  PLEASE NOTE: This function is inspired by the init_random function provided
  as an example by Professor Weiss.
*/
void init_random_state(int id, int seed) {
    char* state = malloc(LAG_TABLE_SIZE * sizeof(char));
    if(NULL == state) {
        MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
    }

    if(seed) {
       initstate(((id + 1) * seed), state, LAG_TABLE_SIZE);
    } else {
       initstate(((id + 1) * time(NULL)), state, LAG_TABLE_SIZE);
    }
}


/*

*/
double get_unif_random() {
    return (((double) random()) / RAND_MAX);
}

/*

*/
void random_solution(int* assignments, int n) {

    int i; /* Loop counter */
    int room; /* Hold room selections */
    int num_rooms = n / 2;
    /* Use calloc to initialize all counts to 0 */
    int* occupancy = calloc(num_rooms, sizeof(int));
    if(NULL == occupancy) {
        MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
    }

    for(i = 0; i < n; i++) { /* Select a room for each student */
        room = floor((get_unif_random()) * num_rooms);
        while(occupancy[room] == 2) { /* Select unoccupied room */
            room = floor((get_unif_random()) * num_rooms);
        }
        assignments[i] = room;
        occupancy[room]++;
    }

    if(DEBUG){
        for(i = 0; i < n; i++) {
            printf("%d ", assignments[i]);
        }
        printf("\n");
    }

}

/*

*/
void room_asst_sim_anneal(int id, int seed, int n, int* assignments,
    double** incompat_matrix, double* solution_cost) {

    double temperature = 10; /* Temperature in the annealing */
    int fail_count = 0; /* Count of temperatures with insufficient updates */
    /* This function uses the passed assignments array as the current solution
    and the passed solution cost as the current cost, therefore only one new
    array and new double are needed, for tracking the "new" results. */
    int* new_solution = malloc((n) * sizeof(int));
    double new_cost;

    if(NULL == new_solution) {
        MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
    }

    /* Initialize random number generator */
    init_random_state(id, seed);
    if(DEBUG) {
        printf("id: %d, random number: %ld\n", id, random());
    }

    /* Initialize to a random room assignment (solution) */
    random_solution(assignments, n);

    /* Main loop */

    free(new_solution);
}


/*

*/
void read_and_distribute_incompat_matrix(char* filename, int id, int num_p,
    int* n, int** assignments, double** incompat_storage,
    double*** incompat_matrix) {

    FILE* matrix_file; /* File handle */
    int i; /* Loop count var */
    int ncols; /* Used only for checking matrix is square. */

    if(ROOT == id) { /* Only root (0) reads the file. */
        matrix_file = fopen(filename, "r");
        if(NULL == matrix_file) {
            *n = 0;
        } else {
            fread(n, sizeof(int), 1, matrix_file);
            fread(&ncols, sizeof(int), 1, matrix_file);
            if(ncols != (*n)){ /* Matrix not square */
                *n = 0;
            }
        }
    }

    /* Transmit n to all processes */
    MPI_Bcast(n, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    /* For debugging */
    if(DEBUG){
        printf("Id %d, n = %d\n", id, *n);
    }

    /* All terminate if file could not be opened */
    if(0 == (*n)) {
        terminate(id, "Could not open matrix file.");
    }

    /* All quit if n is not an even number */
    if((*n) % 2 != 0) {
        terminate(id, "There should be an even number of students.");
    }

    /* All allocate memory */
    *assignments = malloc((*n) * sizeof(int));
    *incompat_storage = malloc(((*n) * (*n)) * sizeof(double));
    *incompat_matrix = malloc((*n) * sizeof(double*));
    if((NULL == (*assignments)) || (NULL == (*incompat_storage)) ||
        (NULL == (*incompat_matrix))) {
        MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
    }

    /* Configure incompat_matrix */
    for(i = 0; i < (*n); i++) {
        (*incompat_matrix)[i] = &((*incompat_storage)[i * (*n)]);
    }

    if(ROOT == id) { /* Only root (0) reads the file. */
        fread((*incompat_storage), sizeof(double), ((*n) * (*n)), matrix_file);
        fclose(matrix_file);
    }

    /* Share matrix */
    MPI_Bcast((*incompat_storage), ((*n) * (*n)), MPI_DOUBLE, ROOT,
        MPI_COMM_WORLD);

    /* For debugging */
    if(DEBUG) {
        int j; /* Second loop counter */
        printf("id: %d\n", id);
        for(i = 0; i < (*n); i++) {
            for(j = 0; j < (*n); j++) {
                printf("%f ", (*incompat_matrix)[i][j]);
            }
            printf("\n");
        }
    }

}

int main(int argc, char* argv[]) {
    int id; /* id of current process */
    int num_p; /* Number of total processes */
    char error_string[127]; /* Buffer for error string if needed */
    int seed = 0; /* Seed for RNG initial state. Zero (default) signals no
                     seeding - i.e. truly "random" behavior. */
    int n; /* Number of students */
    int* assignments; /* Array of room assignments */
    double* incompat_storage; /* Storage for incompatibility matrix */
    double** incompat_matrix; /* Incompatibility matrix pointers */
    double solution_cost; /* Cost of proposed solution */

    /* Initialize MPI, retrieve process id and total # of processes */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    /* Check arguments - filename present, seed optional */
    if((2 != argc) && (3 != argc)) {
        sprintf(error_string, "Usage: %s <filename> [seed]", argv[0]);
        terminate(id, error_string);
    }

    if(3 == argc) {
        seed = atoi(argv[2]);
        if(seed <= 0) {
            terminate(id, "Seed must be a positive integer.");
        }
    }

    /* Read and distribute incompatibility matrix */
    read_and_distribute_incompat_matrix(argv[1], id, num_p, &n, &assignments,
        &incompat_storage, &incompat_matrix);

    /* Conduct simulated annealing */
    room_asst_sim_anneal(id, seed, n, assignments, incompat_matrix,
        &solution_cost);

    /* Determine and print best solution among all processes */

    MPI_Finalize();
    free(assignments);
    free(incompat_storage);
    free(incompat_matrix);

    return 0;
}

