import itertools
import json
from sat import CNF

######################################
# Definitions of the CNF for Sudoku
# atom a(R,C,N) represents "number N is assigned at row R column C"
######################################

"""
1. UEC on row index for each column C and each value N
1.1 EC constraint
a(1,C,N) v ... v a(9,C,N)
1.2 UC constraint for i,j in {0,...,9} such that i<j
-a(i,C,N) v -a(j,C,N)

2. UEC on column index for each row R and each value N
2.1 EC constraint
a(R,1,N) v ... v a(R,9,N)
2.2 UC constraint for i,j in {0,...,9} such that i<j
-a(R,i,N) v -a(R,j,N)

3. UEC on 9 values for each row R and each column C
3.1 EC constraint
a(R,C,1) v ... v a(R,C,9)
3.2 UC constraint for i,j in {0,...,9} such that i<j
-a(R,C,i) v -a(R,C,j)

4. UEC on 9 cells in the same 3*3 box for each value N
4.1 EC constraints
Disj_{i in {1,2,3} and j in {1,2,3}} a(i,j,N)
Disj_{i in {1,2,3} and j in {4,5,6}} a(i,j,N)
Disj_{i in {1,2,3} and j in {7,8,9}} a(i,j,N)
Disj_{i in {4,5,6} and j in {1,2,3}} a(i,j,N)
Disj_{i in {4,5,6} and j in {4,5,6}} a(i,j,N)
Disj_{i in {4,5,6} and j in {7,8,9}} a(i,j,N)
Disj_{i in {7,8,9} and j in {1,2,3}} a(i,j,N)
Disj_{i in {7,8,9} and j in {4,5,6}} a(i,j,N)
Disj_{i in {7,8,9} and j in {7,8,9}} a(i,j,N)
4.2 UC constraints for every 2 atoms a(i1,j1,N) and a(i2,j2,N) in a EC clause
-a(i1,j1,N) v -a(i2,j2,N)
"""

def write_cnf(path_cnf, path_atom2idx):
    # initialize all atoms and assign an index to each of them
    atom2idx = {}
    idx = 1 # the indices start from 1, which follows the dimacs format

    # define 9*9*9 atoms for a(R,C,N)
    for r in range(1,10):
        for c in range(1,10):
            for n in range(1,10):
                atom2idx[f'a({r},{c},{n})'] = idx
                idx += 1

    # construct the CNF
    numClauses = 0
    cnf2 = ''
    # 1. UEC on row index for each column C and each value N
    for c in range(1,10):
        for n in range(1,10):
            atoms = [str(atom2idx[f'a({r},{c},{n})']) for r in range(1,10)]
            # (EC) add a rule "a(1,C,N) v ... v a(9,C,N)"
            ec = ' '.join(atoms)
            cnf2 += f'{ec} 0\n'
            numClauses += 1
            # (UC) add "-a(i,C,N) v -a(j,C,N)" for all combinations of atoms
            for i,j in itertools.combinations(atoms, 2):
                cnf2 += f'-{i} -{j} 0\n'
                numClauses += 1

    # 2. UEC on column index for each row R and each value N
    for r in range(1,10):
        for n in range(1,10):
            atoms = [str(atom2idx[f'a({r},{c},{n})']) for c in range(1,10)]
            # (EC) add a rule "a(R,1,N) v ... v a(R,9,N)"
            ec = ' '.join(atoms)
            cnf2 += f'{ec} 0\n'
            numClauses += 1
            # (UC) add "-a(R,i,N) v -a(R,j,N)" for all combinations of atoms
            for i,j in itertools.combinations(atoms, 2):
                cnf2 += f'-{i} -{j} 0\n'
                numClauses += 1

    # # 3. UEC on 9 values for each row R and each column C
    # for r in range(1,10):
    #     for c in range(1,10):
    #         atoms = [str(atom2idx[f'a({r},{c},{n})']) for n in range(1,10)]
    #         # (EC) add a rule "a(R,C,1) v ... v a(R,C,9)"
    #         ec = ' '.join(atoms)
    #         cnf2 += f'{ec} 0\n'
    #         numClauses += 1
    #         # (UC) add "-a(R,C,i) v -a(R,C,j)" for all combinations of atoms
    #         for i,j in itertools.combinations(atoms, 2):
    #             cnf2 += f'-{i} -{j} 0\n'
    #             numClauses += 1

    # 4. UEC on 9 cells in the same 3*3 box for each value N
    positions = ((1,2,3), (4,5,6), (7,8,9))
    for n in range(1,10):
        for rs in positions:
            for cs in positions:
                atoms = [str(atom2idx[f'a({r},{c},{n})']) for r in rs for c in cs]
                # (EC) add a rule "a(R1,C1,N) v ... v a(R2,C2,N)" for atoms in the same box
                ec = ' '.join(atoms)
                cnf2 += f'{ec} 0\n'
                numClauses += 1
                # (UC) add "-a(R1,C1,N) v -a(R2,C2,N)" for all combinations of atoms
                for i,j in itertools.combinations(atoms, 2):
                    cnf2 += f'-{i} -{j} 0\n'
                    numClauses += 1

    cnf1 = f'p cnf {idx-1} {numClauses}\n'
    with open(path_cnf, 'w') as f:
        f.write(cnf1 + cnf2)
    json.dump(atom2idx, open(path_atom2idx,'w'))
    return atom2idx

def read_cnf(path_cnf, path_atom2idx):
    # try:
    #     cnf = CNF(dimacs=path_cnf)
    #     atom2idx = json.load(open(path_atom2idx))
    # except:
    #     atom2idx = write_cnf(path_cnf, path_atom2idx)
    #     cnf = CNF(dimacs=path_cnf) 
    atom2idx = write_cnf(path_cnf, path_atom2idx)
    cnf = CNF(dimacs=path_cnf)    
    return cnf.C, atom2idx