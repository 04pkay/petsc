// mpicc -o benchmark_baijxsmm benchmark_baijxsmm.c -I${FORK_DIR}/include -I${FORK_DIR}/${FORK_ARCH}/include -I${HOME}/libxsmm/include -L${FORK_DIR}/${FORK_ARCH}/lib -lpetsc -L${HOME}/libxsmm/lib -lxsmm -Wl,-rpath,${FORK_DIR}/${FORK_ARCH}/lib -Wl,-rpath,${HOME}/libxsmm/lib
// mpirun -np 1 ./benchmark_baijxsmm -file block4.petsc -columns 16 -mat_type baij

#include <petsc.h>
#include <libxsmm.h>

static char help[] = "Benchmark SpMM (Sparse Matrix-Matrix) performance for BAIJXSMM format with libxsmm and MatDense.\n\n";

int main(int argc, char **argv) {
    Mat            A, A_xsmm;
    Mat            X, Y, Y_ref;
    PetscInt       iterations = 1000;
    PetscInt       K = 1; // Default to SpMV if -columns is not provided
    PetscLogDouble t1, t2;
    PetscLogDouble loc_t, max_t, min_t, avg_t;
    PetscLogDouble loc_flops, min_flops, max_flops, total_flops;
    PetscLogDouble loc_gflops, max_gflops, min_gflops, tot_gflops;
    PetscLogDouble bytes, bandwidth_gb;
    PetscMPIInt    rank, size;
    char           file[PETSC_MAX_PATH_LEN];
    PetscBool      file_set;
    PetscViewer    viewer;
    PetscInt       block_size;
    MatType        type;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    // Get runtime options
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-iterations", &iterations, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-columns", &K, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file", file, sizeof(file), &file_set));
    if (!file_set) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Please provide a matrix file using -file <filename>");

    // Load Sparse Matrix A as SeqBAIJ
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // Convert to native BAIJXSMM type
    PetscCall(MatConvert(A, MATSEQBAIJXSMM, MAT_INITIAL_MATRIX, &A_xsmm));

    PetscCall(MatGetBlockSize(A_xsmm, &block_size));
    PetscCall(MatGetType(A_xsmm, &type));
    if (rank == 0) printf("SpMM Benchmark: Matrix %s, BlockSize %d, K %d\n", type, (int)block_size, (int)K);

    /* Create Dense Matrix X */
    PetscInt m_local, n_local;
    PetscCall(MatGetLocalSize(A_xsmm, &m_local, &n_local));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, n_local, K, PETSC_DETERMINE, K, NULL, &X));

    // Initialize X with 1.0 (Fixed from MatSet error)
    PetscScalar *px_init;
    PetscCall(MatDenseGetArray(X, &px_init));
    for (int i = 0; i < n_local * K; i++) px_init[i] = 1.0;
    PetscCall(MatDenseRestoreArray(X, &px_init));

    PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));

    // --- Matrix Verification ---
    PetscReal total_error = 0, n_y, n_diff;
    // 1. Calculate ground truth using native PETSc SeqBAIJ
    PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_ref));
    // 2. Calculate using our native BAIJXSMM type (MAT_INITIAL_MATRIX triggers Symbolic phase)
    PetscCall(MatMatMult(A_xsmm, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y));
    // 3. Compute Frobenius norm of the reference
    PetscCall(MatNorm(Y_ref, NORM_FROBENIUS, &n_y));
    // 4. Compute Y = Y - Y_ref
    PetscCall(MatAXPY(Y, -1.0, Y_ref, SAME_NONZERO_PATTERN));
    // 5. Compute Frobenius norm of the difference
    PetscCall(MatNorm(Y, NORM_FROBENIUS, &n_diff));

    total_error = (n_y > 0) ? (n_diff / n_y) : n_diff;
    if (rank == 0) printf("Verification: %s (Error: %g)\n", total_error < 1e-10 ? "SUCCESS" : "FAILURE", (double)total_error);
    PetscCall(MatDestroy(&Y_ref));

    // Warm-up
    for (int i = 0; i < 10; i++) {
        PetscCall(MatMatMult(A_xsmm, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    }

    // Performance Measurement
    PetscCall(PetscBarrier((PetscObject)A_xsmm));
    t1 = MPI_Wtime();
    for (int i = 0; i < iterations; i++) {
        PetscCall(MatMatMult(A_xsmm, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    }
    t2 = MPI_Wtime();
    loc_t = t2 - t1;

    MatInfo loc_info;
    PetscCall(MatGetInfo(A_xsmm, MAT_LOCAL, &loc_info));
    loc_flops = 2.0 * loc_info.nz_used * K * iterations;
    loc_gflops = (loc_t > 0) ? (loc_flops / loc_t) * 1e-9 : 0;

    // Reductions
    PetscCallMPI(MPI_Reduce(&loc_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    avg_t /= size;
    PetscCallMPI(MPI_Reduce(&loc_flops, &total_flops, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_flops, &max_flops, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_flops, &min_flops, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_gflops, &max_gflops, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_gflops, &min_gflops, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
    tot_gflops = (max_t > 0) ? (total_flops / max_t) * 1e-9 : 0;

    // Memory Traffic
    MatInfo info;
    PetscInt M_glob, N_glob;
    PetscCall(MatGetInfo(A_xsmm, MAT_GLOBAL_SUM, &info));
    PetscCall(MatGetSize(A_xsmm, &M_glob, &N_glob));

    bytes = (info.nz_used * (sizeof(PetscScalar) + sizeof(PetscInt)) +
             (M_glob + 1) * sizeof(PetscInt) +
             2.0 * M_glob * K * sizeof(PetscScalar)) * iterations;
    bandwidth_gb = (bytes / max_t) * 1e-9;

    if (rank == 0) {
        printf("\n--- SpMM Benchmark Results (COLUMNS=%d) ---\n", (int)K);
        printf("%-15s %-12s %-12s %-12s %-12s\n", "Metric", "Max", "Min", "Avg", "Total");
        printf("%-15s %-12.3e %-12.3e %-12.3e %-12.3e\n", "Time (sec):", max_t, min_t, avg_t, max_t);
        printf("%-15s %-12.3e %-12.3e %-12.3e %-12.3e\n", "Flops:", max_flops, min_flops, total_flops/size, total_flops);
        printf("%-15s %-12.3f %-12.3f %-12.3f %-12.3f\n", "GFlops/s:", max_gflops, min_gflops, tot_gflops/size, tot_gflops);
        printf("Memory Traffic GB/s: %.3f\n", bandwidth_gb);
        printf("--------------------------------------------\n");
    }

    // Cleanup
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&Y));
    PetscCall(MatDestroy(&A_xsmm));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}