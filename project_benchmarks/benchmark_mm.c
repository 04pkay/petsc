// mpicc -o benchmark_mm benchmark_mm.c -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc

#include <petsc.h>

static char help[] = "Benchmark MatMatMult performance for AIJ and BAIJ formats.\n\n";

int main(int argc, char **argv) {
    Mat            A, X, Y;
    PetscInt       iterations = 1000; 
    PetscInt       K = 1; 
    PetscLogDouble t1, t2;
    MatType        type;
    PetscLogDouble loc_t, max_t, min_t, avg_t;
    PetscLogDouble loc_flops, min_flops, max_flops, avg_flops, total_flops;
    PetscLogDouble loc_gflops, max_gflops, min_gflops, tot_gflops;
    PetscLogDouble bytes, bandwidth_gb;
    PetscMPIInt    rank, size;
    char           file[PETSC_MAX_PATH_LEN];
    PetscBool      file_set;
    PetscViewer    viewer;
    PetscInt       m, n, M, N;

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    // Get runtime options
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-iterations", &iterations, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-k", &K, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file", file, sizeof(file), &file_set));
    if (!file_set) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Please provide a matrix file using -file <filename>");

    // Create Matrix from file
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // Create the dense matrices
    PetscCall(MatGetSize(A, &M, &N));
    PetscCall(MatGetLocalSize(A, &m, &n));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, n, PETSC_DECIDE, N, K, NULL, &X));
    // We create Y initially using MAT_INITIAL_MATRIX in the warm-up, so we don't need to pre-allocate here.
    
    // Initialize X
    PetscCall(MatSetOption(X, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
    PetscCall(MatSetRandom(X, NULL)); 

    PetscCall(MatGetType(A, &type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Running SpMM benchmark: %s x Dense(K=%d)\n", type, (int)K));

    // Warm-up & Symbolic Phase (This creates Y)
    PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y));
    for (int i = 0; i < 9; i++) {
        PetscCall(MatMatMult(A, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    }

    // Performance Measurement
    PetscCall(PetscBarrier((PetscObject)A));
    t1 = MPI_Wtime();
    for (int i = 0; i < iterations; i++) {
        PetscCall(MatMatMult(A, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    }
    t2 = MPI_Wtime();
    loc_t = t2 - t1;

    // Get local non-zeros to calculate local Flops
    MatInfo loc_info;
    PetscCall(MatGetInfo(A, MAT_LOCAL, &loc_info));
    // Flops for SpMM is 2 * NNZ * K
    loc_flops = 2.0 * loc_info.nz_used * K * iterations;
    loc_gflops = (loc_t > 0) ? (loc_flops / loc_t) * 1e-9 : 0;
    
    // Time Reductions
    PetscCallMPI(MPI_Reduce(&loc_t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_t, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_t, &avg_t, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    avg_t /= size;

    // Flops Reductions
    PetscCallMPI(MPI_Reduce(&loc_flops, &max_flops, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_flops, &min_flops, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_flops, &total_flops, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD));
    avg_flops = total_flops / size;
    
    // GFlops/s Reductions
    PetscCallMPI(MPI_Reduce(&loc_gflops, &max_gflops, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD));
    PetscCallMPI(MPI_Reduce(&loc_gflops, &min_gflops, 1, MPI_DOUBLE, MPI_MIN, 0, PETSC_COMM_WORLD));
    tot_gflops = (max_t > 0) ? (total_flops / max_t) * 1e-9 : 0;

    // Bandwidth computation (Updated to include K columns of X and Y)
    MatInfo info;
    PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
    bytes = (info.nz_used * (sizeof(PetscScalar) + sizeof(PetscInt)) + 
             (M + 1) * sizeof(PetscInt) + 
             (double)(M + N) * K * sizeof(PetscScalar)) * iterations;
    bandwidth_gb = (bytes / max_t) * 1e-9;

    if (rank == 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n--- Benchmark Results (%d processes, K=%d) ---\n", size, (int)K));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12s %-12s %-12s %-12s\n", "Metric", "Max", "Min", "Avg", "Total"));
        
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12.3e %-12.3e %-12.3e %-12.3e\n", 
                  "Time (sec):", max_t, min_t, avg_t, max_t));
        
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12.3e %-12.3e %-12.3e %-12.3e\n", 
                  "Flops:", max_flops, min_flops, avg_flops, total_flops));
        
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12.3f %-12.3f %-12.3f %-12.3f\n", 
                  "GFlops/s:", max_gflops, min_gflops, tot_gflops/size, tot_gflops));
                  
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Max/Min Ratio:  %.3f\n", max_gflops/min_gflops));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Memory Traffic GB/s: %-12.3f\n", bandwidth_gb));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "---------------------------------------------------------\n"));
    }

    // Cleanup
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&Y));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}