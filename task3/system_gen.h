/*
  Generates rows [start_row, start_row + num_rows) of the n x n linear system
  A x = b.

  A_local must hold num_rows * n doubles (row-major, local row i is global
  row start_row + i) and b_local must hold num_rows doubles.

  Each row uses its own random-number state derived from (seed, global row),
  so the generated system is identical no matter how the rows are split
  across MPI processes.
*/
void generate_system(double *A_local, double *b_local, int n,
                     int start_row, int num_rows,
                     unsigned int seed, double margin);
