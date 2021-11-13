# CSCI79524 Presentation

This repository contains the source code for an implementation of the
simulated annealing algorithm for solving an instance of the room assignment
problem, which allows for parallel execution of multiple instances of the
algorithm for the given problem.

To compile the program, run:  
`mpicc -Wall -g -o room daniel_mallia_presentation.c -lm`

To run the program, use:  
`mpirun -np <N> room <filename> [seed]`


