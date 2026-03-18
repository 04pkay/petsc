// mpicc -o benchmark_libxsmm_mm benchmark_libxsmm_mm.c -I/${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -I${HOME}/libxsmm/include -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc -L${HOME}/libxsmm/lib -lxsmm
// mpirun -np 1 ./benchmark_libxsmm_mm -file block4.petsc -columns 16 -mat_type baij

#include <petsc.h>
#include <libxsmm.h>

static char help[] = "Benchmark SpMM (Sparse Matrix-Matrix) performance for BAIJ format with libxsmm and MatDense.\n\n";

// Matrix-specific data
typedef struct {
    PetscScalar *v;
    const PetscInt *col, *ptr;
    PetscInt block_size, num_block_rows;
} UserCtx;

// Product-specific data
typedef struct {
    libxsmm_gemmfunction kernel;
    PetscInt K;
} XSMM_ProductCtx;

// Symbolic phase: Prepares the result matrix C and handles allocation logic.
static PetscErrorCode XSMMSpMM_Symbolic(Mat A, Mat B, Mat C, void **prodctx) {
    PetscInt m, n_B, k_B, lda;
    XSMM_ProductCtx *pctx;
    UserCtx *ctx;

    PetscFunctionBegin;

    PetscCall(MatShellGetContext(A, &ctx));

    // Get dimensions from Sparse A and Dense B
    PetscCall(MatGetLocalSize(A, &m, NULL));
    PetscCall(MatGetLocalSize(B, &n_B, &k_B));

    // Get leading dimension from Dense B
    PetscCall(MatDenseGetLDA(B, &lda));

    // Define the layout of the result matrix C (Dense)
    PetscCall(MatSetSizes(C, m, k_B, PETSC_DETERMINE, k_B));
    PetscCall(MatSetType(C, MATDENSE));
    PetscCall(MatSetUp(C));

    PetscCall(PetscNew(&pctx));
    pctx->K = k_B;

    // Dispatch the kernel
    libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
        ctx->block_size, pctx->K, ctx->block_size,
        ctx->block_size, lda, lda, 
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64
    );
    pctx->kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE);

    *prodctx = (void*)pctx;

    PetscFunctionReturn(PETSC_SUCCESS);
}

// Numeric phase: fills C with the actual SpMM result using the LIBXSMM kernel
static PetscErrorCode XSMMSpMM_Numeric(Mat A, Mat B, Mat C, void *prodctx) {
    UserCtx           *ctx;
    XSMM_ProductCtx   *pctx = (XSMM_ProductCtx*)prodctx;
    const PetscScalar *px;
    PetscScalar       *py;
    PetscInt           i, j, row_start, row_end, col_idx;
    PetscInt           bs, lda;
    libxsmm_gemm_param param;

    PetscCall(MatShellGetContext(A, &ctx));
    bs = ctx->block_size;
    PetscCall(MatDenseGetArrayRead(B, &px));
    PetscCall(MatDenseGetLDA(B, &lda));
    PetscCall(MatDenseGetArray(C, &py));
    PetscCall(MatZeroEntries(C)); // Ensure y is zeroed before accumulation (needed because LIBXSMM kernel does C += A*B)

    // Loop over rows of blocks
    for (i = 0; i < ctx->num_block_rows; ++i) {
        row_start = ctx->ptr[i];
        row_end   = ctx->ptr[i + 1];

         // Loop over non-zero blocks in this row
        for (j = row_start; j < row_end; ++j) {
            col_idx = ctx->col[j];
            param.a.primary = &ctx->v[j * bs * bs];
            param.b.primary = (void*)&px[col_idx * bs];
            param.c.primary = (void*)&py[i * bs];
            pctx->kernel(&param);
        }
    }

    PetscCall(MatDenseRestoreArrayRead(B, &px));
    PetscCall(MatDenseRestoreArray(C, &py));
    return PETSC_SUCCESS;
}

static PetscErrorCode XSMMSpMM_Destroy(void *prodctx) {
    XSMM_ProductCtx *pctx = (XSMM_ProductCtx*)prodctx;
    PetscCall(PetscFree(pctx));
    return PETSC_SUCCESS;
}

int main(int argc, char **argv) {
    Mat            A, A_shell;
    Mat            X, Y, Y_ref;
    PetscInt       iterations = 1000;
    PetscInt       K = 1; // Default to SpMV if -columns is not provided
    PetscLogDouble t1, t2;
    MatType        type;
    PetscLogDouble loc_t, max_t, min_t, avg_t;
    PetscLogDouble loc_flops, min_flops, max_flops, total_flops;
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
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-columns", &K, NULL));
    PetscCall(PetscOptionsGetString(NULL, NULL, "-file", file, sizeof(file), &file_set));
    if (!file_set) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Please provide a matrix file using -file <filename>");

    // Load Sparse Matrix A
    PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatLoad(A, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // Setup LIBXSMM context
    PetscCall(MatGetBlockSize(A, &ctx.block_size));
    PetscCall(MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_TRUE, &ctx.num_block_rows, &ctx.ptr, &ctx.col, NULL));
    PetscCall(MatSeqBAIJGetArray(A, &ctx.v));

    /* Create Dense Matrix X */
    PetscInt m_local, n_local, lda;
    PetscCall(MatGetLocalSize(A, &m_local, &n_local));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD, n_local, K, PETSC_DETERMINE, K, NULL, &X));

    // Initialize X with 1.0 (Fixed from MatSet error)
    PetscScalar *px_init;
    PetscCall(MatDenseGetArray(X, &px_init));
    for (int i = 0; i < n_local * K; i++) px_init[i] = 1.0;
    PetscCall(MatDenseRestoreArray(X, &px_init));

    // Create Shell Matrix and register the product operation (Symbolic and Numeric)
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, m_local, n_local, PETSC_DETERMINE, PETSC_DETERMINE, &ctx, &A_shell));
    PetscCall(MatShellSetMatProductOperation(A_shell, MATPRODUCT_AB, XSMMSpMM_Symbolic, XSMMSpMM_Numeric, XSMMSpMM_Destroy, MATDENSE, MATDENSE));

    PetscCall(MatGetType(A, &type));
    if (rank == 0) printf("SpMM Benchmark: Matrix %s, BlockSize %d, K %d\n", type, (int)ctx.block_size, (int)K);

    // --- Matrix Verification ---
    PetscReal total_error = 0, n_y, n_diff;
    // 1. Calculate ground truth using native PETSc
    PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y_ref));
    // 2. Calculate using our libxsmm shell (MAT_INITIAL_MATRIX now triggers Symbolic phase)
    PetscCall(MatMatMult(A_shell, X, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y));
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
        PetscCall(MatMatMult(A_shell, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    }

    // Performance Measurement
    PetscCall(PetscBarrier((PetscObject)A_shell));
    t1 = MPI_Wtime();
    for (int i = 0; i < iterations; i++) {
        PetscCall(MatMatMult(A_shell, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
    }
    t2 = MPI_Wtime();
    loc_t = t2 - t1;

    MatInfo loc_info;
    PetscCall(MatGetInfo(A, MAT_LOCAL, &loc_info));
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
    PetscCall(MatGetInfo(A, MAT_GLOBAL_SUM, &info));
    PetscCall(MatGetSize(A, &M_glob, &N_glob));

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
    PetscCall(MatDestroy(&A_shell));
    PetscCall(MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_TRUE, &ctx.num_block_rows, &ctx.ptr, &ctx.col, NULL));
    PetscCall(MatSeqBAIJRestoreArray(A, &ctx.v));
    PetscCall(MatDestroy(&A));
    PetscCall(PetscFinalize());
    return 0;
}