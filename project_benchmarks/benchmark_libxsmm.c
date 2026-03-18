// mpicc -o benchmark_libxsmm benchmark_libxsmm.c -I/${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -I${HOME}/libxsmm/include -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -L${HOME}/libxsmm/lib -lxsmm
// mpirun -np 1 ./benchmark_libxsmm -file block4.petsc -mat_type baij
#include <petsc.h>
#include <libxsmm.h>

static char help[] = "Benchmark MatMult performance for BAIJ format with libxsmm.\n\n";

// Structure to hold LIBXSMM data
typedef struct {
    libxsmm_gemmfunction kernel;
    PetscInt            block_size;
    PetscInt            num_block_rows;
    PetscScalar         *v;    // Array of all blocks
    const PetscInt      *col;  // Column index for each block
    const PetscInt      *ptr;  // Starting index in 'v'/'col' for each row
} UserCtx;

PetscErrorCode XSMMMatMult(Mat A, Vec x, Vec y) {
    UserCtx           *ctx;
    const PetscScalar *px;
    PetscScalar       *py;
    PetscInt          i, j, row_start, row_end, col_idx;
    PetscInt          bs;

    MatShellGetContext(A, &ctx);
    bs = ctx->block_size;

    VecGetArrayRead(x, &px);
    VecZeroEntries(y); // Ensure y is zeroed before accumulation (needed because LIBXSMM kernel does C += A*B)
    VecGetArray(y, &py);

    // Loop over rows of blocks
    for (i = 0; i < ctx->num_block_rows; i ++) {
        row_start = ctx->ptr[i];
        row_end   = ctx->ptr[i + 1];

        // Loop over non-zero blocks in this row
        for (j = row_start; j < row_end; j++) {
            col_idx = ctx->col[j];

            libxsmm_gemm_param param;
            param.a.primary = &ctx->v[j * bs * bs];  // Block A
            param.b.primary = &px[col_idx * bs];     // Block x
            param.c.primary = &py[i * bs];           // Block y
            // Call the LIBXSMM kernel for this block
            ctx->kernel(&param);
        }
    }

    VecRestoreArrayRead(x, &px);
    VecRestoreArray(y, &py);
    return 0;
}

int main(int argc, char **argv) {
    Mat            A, A_verify;
    Vec            x, y;
    PetscInt       iterations = 1000; // Default number of iterations
    PetscInt       bs = 1; // Default block size
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
    UserCtx        ctx;

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
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A_verify)); // For verification

    // Setup LIBXSMM context
    PetscBool   done;
    PetscCall(MatGetBlockSize(A, &ctx.block_size));
    PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_TRUE, &ctx.num_block_rows, &ctx.ptr, &ctx.col, &done));
    PetscCall(MatSeqBAIJGetArray(A, &ctx.v));

    libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
        ctx.block_size, 1, ctx.block_size, // M, N, K
        ctx.block_size, ctx.block_size, ctx.block_size, // LDA, LDB, LDC
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64
    );
    ctx.kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE);

    // Create the Shell Matrix that uses the LIBXSMM kernel
    Mat A_shell;
    PetscInt m_local, n_local;
    PetscCall(MatGetLocalSize(A, &m_local, &n_local));
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, m_local, n_local, PETSC_DETERMINE, PETSC_DETERMINE, &ctx, &A_shell));
    PetscCall(MatShellSetOperation(A_shell, MATOP_MULT, (void(*)(void))XSMMMatMult));

    // Setup Vectors
    PetscCall(MatCreateVecs(A_shell, &x, &y));
    PetscCall(VecSet(x, 1.0));

    PetscCall(MatGetType(A, &type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Running benchmark with Matrix Type: %s and block size %d\n", type, ctx.block_size));

    // Warm-up
    for (int i = 0; i < 10; i++) {
        PetscCall(MatMult(A_shell, x, y));
    }

    // --- Verification Step ---
    Vec y_ref;
    PetscCall(VecDuplicate(y, &y_ref));

    // 1. Compute reference result using PETSc's native BAIJ mult
    PetscCall(MatMult(A_verify, x, y_ref));

    // 2. Compute LIBXSMM result (your shell matrix)
    PetscCall(MatMult(A_shell, x, y));

    PetscReal norm_y, norm_diff;
    PetscCall(VecNorm(y_ref, NORM_2, &norm_y));      // ||y_ref||
    PetscCall(VecAXPY(y, -1.0, y_ref));             // y = y - y_ref
    PetscCall(VecNorm(y, NORM_2, &norm_diff));      // ||y - y_ref||

    PetscReal relative_error = norm_diff / norm_y;

    if (rank == 0) {
        if (relative_error < 1e-12) {
            PetscPrintf(PETSC_COMM_SELF, "Verification: SUCCESS (Rel. Error: %g)\n", (double)relative_error);
        } else if (relative_error < 1e-8) {
            PetscPrintf(PETSC_COMM_SELF, "Verification: ACCEPTABLE DRIFT (Rel. Error: %g)\n", (double)relative_error);
        } else {
            PetscPrintf(PETSC_COMM_SELF, "Verification: FAILURE (Rel. Error: %g)\n", (double)relative_error);
        }
    }
    PetscCall(VecDestroy(&y_ref));
    // -------------------------

    // Performance Measurement
    PetscCall(PetscBarrier((PetscObject)A_shell));
    t1 = MPI_Wtime();
    for (int i = 0; i < iterations; i++) {
        PetscCall(MatMult(A_shell, x, y));
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
    // Destroy all PETSc objects first
    PetscCall(VecDestroy(&y_ref));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(MatDestroy(&A_shell));
    PetscCall(MatDestroy(&A_verify));
    
    // NOW restore the internal pointers of the original matrix
    PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_TRUE, &ctx.num_block_rows, &ctx.ptr, &ctx.col, &done));
    PetscCall(MatSeqBAIJRestoreArray(A, &ctx.v));
    
    // Finally destroy the original matrix
    PetscCall(MatDestroy(&A));

    PetscCall(PetscFinalize());
    return 0;
}
