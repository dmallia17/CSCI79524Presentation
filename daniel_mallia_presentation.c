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
  Usage:          mpirun -np <N> room <filename> [seed]
                  where N is a positive integer, filename is the name of the
                  incompatibility matrix file, and seed is an optional integer
                  which should be used to seed the random number generator for
                  repeatable behavior. 
  Build with:     mpicc -Wall -g -o room daniel_mallia_presentation.c
  Modifications:  None at this time.
  NOTE:           This program expects the incompatibility matrix values to be
                  in the range [0,10] as described by Quinn in Chapter 10 of
                  Parallel Programming in C with MPI and OpenMP. 
*****************************************************************************/

#include <mpi.h>
#include <stdio.h>

#define ROOT 0 /* Good practice borrowed from estimate_pi.c by Prof. Weiss */

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


int main(int argc, char* argv[]) {
    int id; /* id of current process */
    int num_p; /* Number of total processes */
    char error_string[127]; /* Buffer for error string if needed */
    int seed = 0; /* Seed for RNG initial state. Zero (default) signals no
                     seeding - i.e. truly "random" behavior. */

    /* Initialize MPI, retrieve process id and total # of processes */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_p);

    /* Check arguments - filename present, seed optional */

    /* Read and distribute incompatibility matrix */

    /* Conduct simulated annealing */

    /* Determine and print best solution among all processes */

    MPI_Finalize();

    return 0;
}

