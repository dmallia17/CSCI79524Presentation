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
                  Thus it expects a BINARY file of the type output by the
                  matrix2binary program provided by Professor Weiss, for the
                  incompatibility matrix file.
  Usage:          mpirun -np <N> room <filename> [seed]
                  where N is a positive integer, filename is the name of the
                  BINARY (see above) incompatibility matrix file, and seed is
                  an optional integer which should be used to seed the random
                  number generator for repeatable behavior.
  Build with:     mpicc -Wall -g -o room daniel_mallia_presentation.c -lm
  Modifications:  None at this time.

*****************************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ROOT 0 /* Good practice borrowed from estimate_pi.c by Prof. Weiss */
#define MALLOC_ERROR 1
#define DEBUG 0
#define LAG_TABLE_SIZE 1000

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
  This function initializes the random number generator via use of the
  initstate() function, which seeds the lagged Fibonacci generator made
  available by the random() function. It does this in an independent
  sequencing fashion such that each calling process will have an id-dependent
  seeding. However, a seed can be provided to make this process reproducible;
  otherwise the seeding is based on the id and the current time.

  @param id        The id of the calling process.
  @param seed      Either 0 to seed based on time or a positive integer to
                   seed in a reproducible manner.
  @return          A pointer to the array newly allocated for the lag table.
  @Pre-Condition   id is a valid process id and seed is non-negative.
  @Post-Condition  The random number generator has been seeded in accordance
                   with the calling process id: either in a reproducible
                   manner thanks to a seed, or even more "randomly" by basing
                   on the time.
*/
char* init_random_state(int id, int seed) {
    char* state = malloc(LAG_TABLE_SIZE * sizeof(char));
    if(NULL == state) {
        MPI_Abort(MPI_COMM_WORLD, MALLOC_ERROR);
    }

    if(seed) {
       initstate(((id + 1) * seed), state, LAG_TABLE_SIZE);
    } else {
       initstate(((id + 1) * time(NULL)), state, LAG_TABLE_SIZE);
    }

    return state;
}


/*
  A convenience function to generate a random number in the interval [0,1)
  (from a uniform distribution). Note the half-open interval: this enables
  convenient multiplication of the returned value by other values (for
  example, taking the floor of multiplying by the length of the array, such
  that no out of bounds value is generated). This is accomplished by dividing
  by one greater than the largest possible value returned by random()
  (RAND_MAX).

  @return          A double in the range [0,1).
  @Pre-Condition   As this function uses the random() function, the
                   init_random_state() function should have already been
                   called to seed the generator.
  @Post-Condition  A random value has been returned.
*/
double get_unif_random() {
    return (((double) random()) / (((long) RAND_MAX) + 1));
}

/*
  This function implements the room assignment objective function,
  calculating the "cost" of a given solution (assignments) for a given
  incompatibility matrix. PLEASE NOTE: this is basically an implementation
  of the summation given on page 41 of Professor Weiss' Chapter 8 Lecture
  Notes.

  @param assignments      An n-long array representing a solution in the form
                          of an assigned room for each of the n students.
  @param n                The number of students.
  @param incompat_matrix  A doubly subscriptable n x n incompatibility matrix
                          which is symmetric and contains values indicating
                          the degree to which two students are incompatible.
  @return                 The cost of the solution.
  @Pre-Condition          assignments and incompat_matrix should be properly
                          initialized in accordance with their semantics
                          (i.e. no more than 2 students assigned to a room
                          and incompatibility values where higher values
                          indicate greater incompatibility).
  @Post-Condition         The cost of the proposed solution has been returned.
*/
double cost(int* assignments, int n, double** incompat_matrix) {
    int i, j; /* Loop counters */
    double total = 0; /* Cost accumulator */
    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            if((i != j) && (assignments[i] == assignments[j])) {
                total += incompat_matrix[i][j];
            }
        }
    }
    return (total / 2);
}

/*
  This function provides a starting point for the simulated annealing
  algorithm - an initial solution (set of room assignments). It proceeds by
  maintaining an occupancy array, tracking the number of students assigned to
  each of the (n/2) rooms, and randomly selects a room for each student: if
  the chosen room is already full, random selections are continuously made
  until a room with a free slot is chosen.
  @param assignments  An array of length n to be filled with random room
                      assignments
  @param n            The number of students (and length of assignments array)
  @Pre-Condition      As this function uses the get_unif_random() function,
                      init_random_state() should already have been called.
                      Also, the memory for assignments should already have
                      been allocated.
  @Post-Condition     assignments has been filled with a random solution.
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
  This function manages the printing of the best solution found among the
  num_p processes, along with the id of that process and the cost of the
  solution. In short, ROOT assumes its solution is best and then polls all
  other processes for the cost of their solutions to see if a better one has
  been found; then all processes participate in a broadcast whereby ROOT
  shares the id of the process with the best solution; that process is then
  responsible for the final output (printing).
  @param id             The id of the calling process
  @param num_p          The total number of processes
  @param n              The number of students
  @param solution_cost  The cost of the last solution accepted by the
                        calling process
  @param assignments    The last solution accepted by the calling process
  @Pre-Condition        room_asst_sim_anneal() should already have been run
                        such that assignments is the last accepted solution
                        by a process and solution_cost is its cost.
  @Post-Condition       The process that has the best solution has printed
                        its id, the solution cost, and the solution.
*/
void print_best_solution(int id, int num_p, int n, double solution_cost,
    int* assignments) {
    int i; /* Loop counter */
    int signal; /* Signal to transmit */
    MPI_Status status; /* Structure for transmission info */
    int best_solution_id = 0;
    double best_solution_cost = solution_cost;
    double temp_solution_cost;

    if(ROOT == id) { /* Root polls all other processes for best results */
        for(i = 1; i < num_p; i++) {
            MPI_Send(&signal, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Recv(&temp_solution_cost, 1, MPI_DOUBLE, i, 1,
                MPI_COMM_WORLD, &status);
            if(temp_solution_cost < best_solution_cost) {
                best_solution_cost = temp_solution_cost;
                best_solution_id = i;
            }
        }
    } else { /* All other processes wait to send their cost */
        MPI_Recv(&signal, 1, MPI_INT, ROOT, 1, MPI_COMM_WORLD, &status);
        MPI_Send(&best_solution_cost, 1, MPI_DOUBLE, ROOT, 1, MPI_COMM_WORLD);
    }

    /* ROOT sends the id of the process with the best solution */
    MPI_Bcast(&best_solution_id, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    /* Process with best solution prints the cost and the solution */
    if(id == best_solution_id) {
        printf("Process %d found the following solution with cost %f\n",
            id, solution_cost);
        for(i = 0; i < n; i++) {
            printf("%d ", assignments[i]);
        }
        printf("\n\n");
    }

}

/*
  PLEASE NOTE: This function is an implementation of the pseudocode given in
  Listing 8.7 on page 43 in Chapter 8 of Professor Weiss' Lecture Notes.
  This function implements the simulated annealing algorithm for solving an
  instance of the room assignment problem, using the temperature decreasing
  process proposed by Kirkpatrick et. al in their 1983 article "Optimization
  by Simulated Annealing" (Science, New Series, Vol. 220, No. 4598. (May 13,
  `983), pp. 671-680.)
  @param id           The id of the calling process
  @param seed         A value to be passed to init_random_state
  @param n            The number of students
  @param assignments  A pre-allocated array of length n
  @incompat_matrix    A doubly subscriptable n x n incompatibility matrix
                      which is symmetric and contains values indicating
                      the degree to which two students are incompatible.
  @solution_cost      Pointer to a double where the last solution cost
                      should be recorded
  @Pre-Condition      assignments should point to an array of length n and
                      incompat_matrix should contain the contents of the
                      incompatibility matrix read from a file
  @Post-Condition     assignments contains the last accepted solution and
                      solution_cost points to its cost
*/
void room_asst_sim_anneal(int id, int seed, int n, int* assignments,
    double** incompat_matrix, double* solution_cost) {

    double temperature = 10; /* Temperature in the annealing */
    int fail_count = 0; /* Count of temperatures with insufficient updates */
    int acceptance_counter; /* # of times new solution accepted */
    int number_of_attempts; /* # of times a new solution was tested */
    int attempt_cap = 100 * n; /* Limit on attempts - calculated once */
    double u; /* Random number for deciding on accepting worse solution */
    double p; /* Probability of accepting a worse solution */
    int s1, s2; /* Students 1 and 2 for swapping */
    int temp_room; /* Swap variable for room assignments */
    double delta; /* Change in cost */
    double new_cost; /* Cost of new solution */
    char* random_state; /* Array for initializing RNG */

    /* Initialize random number generator */
    random_state = init_random_state(id, seed);
    if(DEBUG) {
        printf("id: %d, random number: %ld\n", id, random());
    }

    /* Initialize to a random room assignment (solution) */
    random_solution(assignments, n);

    /* Main loop */
    while(1) {
        acceptance_counter = 0;
        number_of_attempts = 0;

        while((acceptance_counter < 10) &&
            (number_of_attempts < attempt_cap)) {

            /* Select two students to swap between rooms */
            s1 = floor((get_unif_random()) * n);
            s2 = floor((get_unif_random()) * n);

            /* Check not same student or student not in same room as s1 */
            while((s1 == s2) || (assignments[s1] == assignments[s2])) {
                s2 = floor((get_unif_random()) * n);
            }

            /* Calculate score for current solution */
            *solution_cost = cost(assignments, n, incompat_matrix);

            /* Swap students */
            temp_room = assignments[s1];
            assignments[s1] = assignments[s2];
            assignments[s2] = temp_room;

            /* Calculate score of new solution */
            new_cost = cost(assignments, n, incompat_matrix);
            delta = new_cost - (*solution_cost);

            u = get_unif_random();
            p = exp((((-1) * delta) / temperature));

            if((delta < 0) || (u < p)) { /* Accept new solution */
                acceptance_counter++;
                *solution_cost = new_cost;
            } else { /* Reject new solution */
                number_of_attempts++;
                /* Swap back students */
                temp_room = assignments[s1];
                assignments[s1] = assignments[s2];
                assignments[s2] = temp_room;
            }
        }

        if(10 == acceptance_counter) {
            fail_count = 0;
        } else {
            fail_count++;
            /* Quit if insufficient change in 3 consecutive temperatures */
            if(3 == fail_count) {
                break;
            }
        }

        /* Decrease temperature and quit if low and change is unlikely */
        temperature = (0.9 * temperature);
        if(temperature < 0.001) {
            break;
        }
    }

    if(DEBUG) {
        int i;
        printf("id: %d, cost: %f, solution:\n", id, *solution_cost);
        for(i = 0; i < n; i++) {
            printf("%d ", assignments[i]);
        }
        printf("\n");
    }

    free(random_state);
}


/*
  This function handles reading of the BINARY file containing the
  incompatibility matrix for the problem instance, allocating memory for a
  solution array, the incompatibility matrix in contiguous memory and an
  array of pointers offering simple doubly subscriptable [row][col] access
  to the matrix. As all processes need the same matrix, it is shared in full
  via a broadcast.
  @param filename          The name of the BINARY file containing the
                           incompatibility matrix (in the format output by
                           the matrix2binary program provided by Professor
                           Weiss).
  @param id                The id of the calling process
  @param num_p             The number of processes
  @param n                 A pointer to an integer location where the number
                           of students should be recorded
  @param assignments       A pointer to the assignments pointer, which will
                           be updated to point to memory newly allocated for
                           an integer array of length n
  @param incompat_storage  A pointer to the incompatibility matrix linear
                           storage pointer, which will be updated to point to
                           a contiguous block of memory for the n x n doubles
  @param incompat_matrix   A pointer to the incompatibility matrix row
                           pointers array pointer, which will be updated to
                           point to an array of pointers, each pointing to the
                           start of a new row in the incompat_storage memory
  @Pre-Condition           filename contains a proper filename
  @Post-Condition          Memory has been allocated for assignments;
                           incompat_storage points to a pointer to the matrix
                           stored in contiguous memory; incompat_matrix points
                           to an array of double pointers, each of which
                           points to one of the n rows in incompat_storage
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
    print_best_solution(id, num_p, n, solution_cost, assignments);

    MPI_Finalize();
    free(assignments);
    free(incompat_storage);
    free(incompat_matrix);

    return 0;
}

