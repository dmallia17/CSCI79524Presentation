# CSCI79524 Presentation: The Room Assignment Problem
**Author: Daniel Mallia**

This repository contains the source code for an implementation of the
simulated annealing algorithm for solving an instance of the room assignment
problem, which allows for parallel execution of multiple instances of the
algorithm for the given problem.

To compile the program, run:  
`mpicc -Wall -g -o room daniel_mallia_presentation.c -lm`

To run the program, use:  
`mpirun -np <N> room <filename> [seed]`

Also, for testing purposes, a Python script is provided to generate test
incompatibility matrices which are constructed such that the best solution
is a "bookend" pattern where the first student should be matched with the
nth, the second with the (n-1), etc. To run the script use:  
`python3 gen_incompat_matrix.py <N> [--seed SEED]`

This script was used to generate a couple of sample matrices, provided in the
test_incompatibility_matrices directory.


