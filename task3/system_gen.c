#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "./system_gen.h"

/*
  Thread-safe uniform random number in [lo, hi). The state is passed in
  explicitly, so every row can have its own independent sequence.
*/
static double random_uniform(unsigned int *state, double lo, double hi) {
  return lo + (hi - lo) * ((double) rand_r(state) / ((double) RAND_MAX + 1.0));
}

void generate_system(double *A_local, double *b_local, int n,
                     int start_row, int num_rows,
                     unsigned int seed, double margin) {

  for (int i = 0; i < num_rows; i++) {

    // Global index of this row in the full matrix A
    int global_i = start_row + i;

    // Independent random state for this row, so the row's values do not
    // depend on which process generates it or on any earlier rows
    unsigned int row_state = seed + (unsigned int) global_i * 2654435761u;

    // Pointer to the start of local row i (size_t avoids int overflow
    // for large n)
    double *row = A_local + (size_t) i * n;
    double row_sum = 0.0;

    /* Fill off-diagonal entries of row global_i */
    for (int j = 0; j < n; j++) {
      if (j == global_i)
        continue; // Skip the diagonal.
      row[j] = random_uniform(&row_state, 0.0, 1.0);
      row_sum += fabs(row[j]);
    }

    /* Fill the diagonal entry; margin controls the dominance. */
    row[global_i] = row_sum * (1 + margin);

    /* Matching element of b, random in [-10, 10], from the same row state. */
    b_local[i] = random_uniform(&row_state, -10.0, +10.0);
  }
}
