#define CC 52
#define BLOCK_THREADS 64
#define BLOCK_DIM 16

#ifndef IDX2C
#define IDX2C(i,j,ld) (j * ld + i) // i is column, j is row, ld is total number of columns
#endif

#define NUM_BLOCKS(M) (M + BLOCK_THREADS - 1) / BLOCK_THREADS