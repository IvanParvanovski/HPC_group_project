
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "./system_gen.h"

/*
  Computes a 2-norm of a vector v.
*/
// static double vector_norm(double *v, int n) {

//   double sum_square = 0.0;
//   for (int i = 0; i < n; i++)
//     sum_square += v[i] * v[i];

//   return sqrt(sum_square);
// }

/*
  Computes a 2-norm of Ax-b.
*/
// static double residual_norm(double *A, double *x, double *b, int n) {

//   double sum_square = 0.0;

//   for (int i = 0; i < n; i++) {
//     double Ax_i = 0.0;

//     for (int j = 0; j < n; j++)
//       Ax_i += A[i*n + j] * x[j];

//     double r = Ax_i - b[i];
//     sum_square += r*r;
//   }

//   return sqrt(sum_square);
// }

static double *jacobi(double *A, double *b, double *x,
                      int n, int local_rows, int start_row,
                      int *row_counts, int *row_start_indices,
                      int rank, int max_iter, double tolerance) {

  // Allocate space for the local and complete new solution vectors
  double *local_x_new = calloc(local_rows > 0 ? local_rows : 1, sizeof(double));
  double *x_new = calloc(n, sizeof(double));

  // Check memory allocation across all processes
  int ok = (local_x_new && x_new);
  int all_ok = 0;

  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (!all_ok) {
    if (rank == 0)
      fprintf(stderr, "Memory allocation failed.\n");

    free(local_x_new);
    free(x_new);
    return NULL;
  }

  // Calculate the local contribution to the squared norm of b
  double local_b_sq = 0.0;
  double global_b_sq = 0.0;

  for (int i = 0; i < local_rows; i++)
    local_b_sq += b[i] * b[i];

  // Sum the contributions from all processes
  MPI_Allreduce(&local_b_sq, &global_b_sq, 1,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double b_norm = sqrt(global_b_sq);

  int iter;
  for (iter = 1; iter <= max_iter; iter++) {

    // Calculate new values for the rows assigned to this process
    for (int i = 0; i < local_rows; i++) {

      // Convert the local row index to its global index
      int global_i = start_row + i;
      double sigma = 0.0;

      // Sum the off-diagonal terms using the previous solution
      for (int j = 0; j < n; j++)
        if (j != global_i)
          sigma += A[(size_t)i*n + j] * x[j];

      // Compute the new solution value using the Jacobi formula
      local_x_new[i] = (b[i] - sigma) / A[(size_t)i*n + global_i];
    }

    // Collect all parts of the new solution on every process
    MPI_Allgatherv(local_x_new, local_rows, MPI_DOUBLE,
                   x_new, row_counts, row_start_indices,
                   MPI_DOUBLE, MPI_COMM_WORLD);

    // Calculate the local squared residual ||Ax-b||^2
    double local_residual_sq = 0.0;
    double global_residual_sq = 0.0;

    for (int i = 0; i < local_rows; i++) {
      double Ax_i = 0.0;

      // Multiply the local matrix row by the new solution vector
      for (int j = 0; j < n; j++)
        Ax_i += A[(size_t)i*n + j] * x_new[j];

      double r = Ax_i - b[i];
      local_residual_sq += r * r;
    }

    // Sum the squared residuals across all processes
    MPI_Allreduce(&local_residual_sq, &global_residual_sq, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Calculate the relative residual error
    double accuracy = sqrt(global_residual_sq);
    double relative_error = b_norm > 0.0 ? accuracy / b_norm : accuracy;

    // Swap pointers so x contains the newest solution
    double *tmp = x;
    x = x_new;
    x_new = tmp;

    // Stop if the relative error is below the tolerance
    if (relative_error < tolerance) {
      if (rank == 0)
        printf("Solved with error %e in %d iterations.\n",
               relative_error, iter);
      break;
    }
  }

  // Free the temporary solution arrays
  free(local_x_new);
  free(x_new);

  if (iter > max_iter && rank == 0)
    printf("Solver did not converge in %d iterations.\n", max_iter);

  return x;
}

int main(int argc, char *argv[]) {

  // Initialise MPI and get the rank and total number of processes
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Read the matrix size and diagonal dominance margin
  if (argc < 3) {
    if (rank == 0)
      fprintf(stderr, "Please specify n and margin.\n");

    MPI_Finalize();
    return 1;
  }

  int n = atoi(argv[1]);
  double margin = atof(argv[2]);

  if (n <= 0) {
    if (rank == 0)
      fprintf(stderr, "n must be positive.\n");

    MPI_Finalize();
    return 1;
  }

  // Store how many rows each process receives and their starting positions
  int row_counts[size], row_start_indices[size];
  int element_counts[size], element_start_indices[size];

  int offset = 0;

  // Divide the matrix rows between the processes
  for (int p = 0; p < size; p++) {
    int rows = n / size;

    // Give one extra row to the first processes if n is not divisible by size
    if (p < n % size)
      rows++;

    row_counts[p] = rows;
    row_start_indices[p] = offset;

    // Each matrix row contains n elements
    element_counts[p] = rows * n;
    element_start_indices[p] = offset * n;

    offset += rows;
  }

  // Find the number of rows and starting row for this process
  int local_rows = row_counts[rank];
  int start_row = row_start_indices[rank];

  // Allocate memory for the local matrix, vector b, and full solution x
  double *local_A = calloc(
    (size_t)(local_rows > 0 ? local_rows : 1) * n, sizeof(double));

  double *local_b = calloc(
    local_rows > 0 ? local_rows : 1, sizeof(double));

  double *x = calloc(n, sizeof(double));

  // Only rank 0 stores the full linear system
  double *A = NULL;
  double *b = NULL;

  if (rank == 0) {
    A = calloc((size_t)n * n, sizeof(double));
    b = calloc(n, sizeof(double));
  }

  // Check whether memory allocation succeeded on every process
  int ok = local_A && local_b && x && (rank != 0 || (A && b));
  int all_ok = 0;

  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (!all_ok) {
    if (rank == 0)
      fprintf(stderr, "Memory allocation failed.\n");

    free(A);
    free(b);
    free(local_A);
    free(local_b);
    free(x);

    MPI_Finalize();
    return 1;
  }

  // Synchronise processes before measuring the runtime
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  // Generate the complete linear system on rank 0
  if (rank == 0)
    generate_system(A, b, n, 0, margin);

  // Distribute the rows of A between processes
  MPI_Scatterv(A, element_counts, element_start_indices, MPI_DOUBLE,
               local_A, local_rows * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  // Distribute the corresponding elements of b
  MPI_Scatterv(b, row_counts, row_start_indices, MPI_DOUBLE,
               local_b, local_rows, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

  // Set the convergence tolerance and maximum number of iterations
  double tolerance = 1e-10;
  int max_iter = 500;

  // Solve the system using the distributed Jacobi method
  double *result = jacobi(local_A, local_b, x,
                          n, local_rows, start_row,
                          row_counts, row_start_indices,
                          rank, max_iter, tolerance);

  // Handle memory allocation failure in Jacobi
  if (!result) {
    free(local_A);
    free(local_b);
    free(A);
    free(b);
    free(x);

    MPI_Finalize();
    return 1;
  }

  x = result;

  // Measure the elapsed time for this process
  double local_runtime = MPI_Wtime() - t0;
  double runTime = 0.0;

  // Use the longest process runtime as the total runtime
  MPI_Allreduce(&local_runtime, &runTime, 1,
                MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0)
    printf("Total run time: %f seconds\n", runTime);

  // Free allocated memory
  free(local_A);
  free(local_b);
  free(A);
  free(b);
  free(x);

  MPI_Finalize();
  return 0;
}
