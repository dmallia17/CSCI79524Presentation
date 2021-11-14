##############################################################################
# Title:         gen_incompat_matrix.py
# Author:        Daniel Mallia
# Created on:    November 14, 2021
# Description:   Enables generation of test incompatibility matrices for the
#                room assignment problem. To make the matrices usable as
#                simple test cases, a "bookend" pattern is used whereby the
#                matrix is constructed such that the best solution would be
#                to match the first student with the nth, second with the 
#                (n-1), etc. The output of this program is a .txt file
#                named in accordance with n and the current date and time, so
#                you can have more than one n x n test case! The text file is
#                written in the format accepted by the matrix2binary program
#                written by Professor Weiss, so you can then binarize it and
#                run the simulated annealing algorithm on it.
# Purpose:       Testing the simulated annealing algorithm for solving the
#                room assignment problem requires incompatibility matrices.
#                However, how can one tell if the best solution has actually
#                been found without checking the nightmare number of possible
#                solutions?! This program solves that problem by creating
#                test matrices that have an easily spottable best solution
#                - see above for the description of the bookend pattern used.
# Usage:         python3 gen_incompat_matrix.py <N> [--seed SEED]
#                where N is an even positive integer number of students and
#                SEED is an optional positive integer value to enable
#                reproducible behavior
# Modifications: None.
##############################################################################

import argparse, datetime, random

def odd_fact(n):
    total = 1
    for i in list(range(1, n, 2)):
        total *= i
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = ("Generate incompatibility matrices with a 'bookend'" +
        " pattern"))
    parser.add_argument("n", type=int, 
        help="Even positive integer number of students")
    parser.add_argument("--seed", type=int, default=0,
        help="Seed for random")
    args = parser.parse_args()

    # Check args
    if ((args.n <= 0) or (args.n % 2 != 0)):
        print("n must be a positive, even integer")
        exit()

    if args.seed < 0:
       print("Seed should not be negative")
       exit()

    n = args.n
    print("Given", n, "students there are", odd_fact(n),
        "possible assignments") 

    # Seed RNG if seed given
    if args.seed:
        random.seed(args.seed)
    else:
        random.seed() # Uses system time

    # Prepare a incompatibility matrix of all 0s (floating point)
    incompat_matrix = [[0. for j in list(range(n))] for i in list(range(n))]

    # Fill in the upper part matrix such that each student j is "matched" with
    # the ((n-1) - j) student and thus has an incompatibility of 1.0, and all
    # values w.r.t. other students are random between 2.0 and 10.0
    max_num = (n-1)
    for i in list(range(n)):
        match =  max_num - i
        for j in range((i + 1), n):
            if(j == match):
                incompat_matrix[i][j] = 1.0
            else:
                incompat_matrix[i][j] = random.uniform(2.0, 10.0)

    # Make matrix symmetric (copying into lower part)
    for i in list(range(n)):
        for j in range((i + 1), n):
            incompat_matrix[j][i] = incompat_matrix[i][j]

    # Prepare a filename using current date and time - minute precision
    dt = datetime.datetime.today()
    filename = "incompat_mat_" + str(n) + "x" + str(n) + "_" + \
        datetime.datetime.today().strftime("%Y_%m_%d_%H_%M") + ".txt"

    # Write incompatibility matrix file.
    with open(filename, "w") as f:    
        print(n, n, file=f)
        for i in list(range(n)):
            for j in list(range(max_num)):
                print(round(incompat_matrix[i][j],1), end=" ", file=f)
            # Print only newline (no extra space) after final value
            print(round(incompat_matrix[i][max_num],1), file=f)
        

