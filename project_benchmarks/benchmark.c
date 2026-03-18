// mpicc -o benchmark benchmark.c -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc

#include <petsc.h>

static char help[] = "Benchmark MatMult performance for AIJ and BAIJ formats.\n\n";

int main(int argc, char **argv) {
    Mat            A;
    Vec            x, y;
    PetscInt       iterations = 1000; // Default number of iterations
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

    PetscFunctionBeginUser;
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    // Get runtime options
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-iterations", &iterations, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file", file, sizeof(file), &file_set));
    if (!file_set) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Please provide a matrix file using -file <filename>");

    // Create Matrix from file
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // Setup Vectors
    PetscCall(MatCreateVecs(A, &x, &y));
    PetscCall(VecSet(x, 1.0));

    PetscCall(MatGetType(A, &type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Running benchmark with Matrix Type: %s\n", type));

    // Warm-up
    for (int i = 0; i < 10; i++) {
        PetscCall(MatMult(A, x, y));
    }

    // Performance Measurement
    PetscCall(PetscBarrier((PetscObject)A));
    t1 = MPI_Wtime();
    for (int i = 0; i < iterations; i++) {
        PetscCall(MatMult(A, x, y));
    }
    t2 = MPI_Wtime();
    loc_t = t2 - t1;

    // Get local non-zeros to calculate local Flops
    MatInfo loc_info;
    PetscCall(MatGetInfo(A, MAT_LOCAL, &loc_info));
    loc_flops = 2.0 * loc_info.nz_used * iterations;
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

    // Bandwidth computation
    MatInfo info;
    PetscInt M, N;
    PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
    PetscCall(MatGetSize(A, &M, &N));
    bytes = (info.nz_used * (sizeof(PetscScalar) + sizeof(PetscInt)) + (M + 1) * sizeof(PetscInt) + 2.0 * M * sizeof(PetscScalar)) * iterations;
    bandwidth_gb = (bytes / max_t) * 1e-9;

    if (rank == 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n--- Benchmark Results (%d processes) ---\n", size));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12s %-12s %-12s %-12s\n", "Metric", "Max", "Min", "Avg", "Total"));
        
        // Time Row
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12.3e %-12.3e %-12.3e %-12.3e\n", 
                  "Time (sec):", max_t, min_t, avg_t, max_t));
        
        // Flops Row
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12.3e %-12.3e %-12.3e %-12.3e\n", 
                  "Flops:", max_flops, min_flops, avg_flops, total_flops));
        
        // GFlops/s Row
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%-15s %-12.3f %-12.3f %-12.3f %-12.3f\n", 
                  "GFlops/s:", max_gflops, min_gflops, tot_gflops/size, tot_gflops));
                  
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Max/Min Ratio:  %.3f\n", max_gflops/min_gflops));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Memory Traffic GB/s: %-12.3f\n", bandwidth_gb));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "---------------------------------------------------------\n"));
    }

    // Cleanup
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}
