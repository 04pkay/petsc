#include <petsc/private/matimpl.h>
#include <../src/mat/impls/baij/seq/baij.h> /*I   "petscmat.h"   I*/
#include <libxsmm.h>

/* Product-specific data */
typedef struct {
    libxsmm_gemmfunction kernel;
    PetscInt K;
} XSMM_ProductCtx;

/* Destroy routine */
static PetscErrorCode XSMMSpMM_Destroy(void *prodctx) {
    XSMM_ProductCtx *pctx = (XSMM_ProductCtx*)prodctx;
    PetscFunctionBegin;
    printf("[DIAG] Destroying pctx at address %p\n", prodctx);
    PetscCall(PetscFree(pctx));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Numeric phase: fills C with the actual SpMM result using the LIBXSMM kernel */
static PetscErrorCode XSMMSpMM_Numeric(Mat C) {
    Mat                A = C->product->A;
    Mat                B = C->product->B;
    Mat_SeqBAIJ       *ctx = (Mat_SeqBAIJ *)A->data;
    XSMM_ProductCtx   *pctx = (XSMM_ProductCtx*)C->product->data;
    const PetscScalar *px;
    PetscScalar       *py;
    PetscInt           i, j, row_start, row_end, col_idx;
    PetscInt           bs;
    libxsmm_gemm_param param;

    /* Map PETSc's internal arrays */
    PetscScalar    *v = ctx->a;
    const PetscInt *col = ctx->j;
    const PetscInt *ptr = ctx->i;
    PetscInt       block_size = A->rmap->bs;
    PetscInt       num_block_rows = ctx->mbs;

    PetscFunctionBegin;

    bs = block_size;

    PetscCheck(pctx->kernel, PETSC_COMM_SELF, PETSC_ERR_LIB,
        "LIBXSMM kernel is NULL — JIT dispatch failed (unsupported arch or shape)");

    PetscCall(MatDenseGetArrayRead(B, &px));
    PetscCall(MatZeroEntries(C)); // Ensure C is zeroed before accumulation
    PetscCall(MatDenseGetArray(C, &py));

    // Loop over rows of blocks
    for (i = 0; i < num_block_rows; ++i) {
        row_start = ptr[i];
        row_end   = ptr[i + 1];

         // Loop over non-zero blocks in this row
        for (j = row_start; j < row_end; ++j) {
            col_idx = col[j];
            param.a.primary = &v[j * bs * bs];
            param.b.primary = (void*)&px[col_idx * bs];
            param.c.primary = (void*)&py[i * bs];
            pctx->kernel(&param);
        }
    }

    PetscCall(MatDenseRestoreArrayRead(B, &px));
    PetscCall(MatDenseRestoreArray(C, &py));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Symbolic phase: Prepares the result matrix C and handles allocation logic. */
static PetscErrorCode XSMMSpMM_Symbolic(Mat C) {
    Mat              A = C->product->A;
    Mat              B = C->product->B;
    PetscInt         m, k_B, lda, lda_C, block_size;
    XSMM_ProductCtx *pctx;

    PetscFunctionBegin;
    /* 1. Dimensions */
    m = A->rmap->n;
    block_size = A->rmap->bs;
    PetscCall(MatGetLocalSize(B, NULL, &k_B));
    PetscCall(MatDenseGetLDA(B, &lda));

    /* 2. COMPLETELY SET UP THE MATRIX FIRST */
    /* This ensures PETSc finishes all its internal 'product' resets */
    PetscCall(MatSetSizes(C, m, k_B, m, k_B));
    PetscCall(MatSetType(C, MATSEQDENSE));
    PetscCall(MatSetUp(C)); 
    PetscCall(MatDenseGetLDA(C, &lda_C));

    /* 3. NOW ALLOCATE YOUR DATA */
    PetscCall(PetscNew(&pctx));
    pctx->K = k_B;

    /* JIT Dispatch */
    libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
        block_size, pctx->K, block_size,
        block_size, lda, lda_C,
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64
    );
    pctx->kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE);

    /* 4. ATTACH TO THE FINISHED MATRIX */
    C->product->data       = pctx;
    C->product->destroy    = XSMMSpMM_Destroy;
    C->ops->productnumeric = XSMMSpMM_Numeric;
    
    printf("[DIAG] Allocated pctx at address %p\n", (void*)pctx);
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- PETSc Integration Hooks --- */

/* Route MatMatMult to XSMM if Dense, else fallback */
static PetscErrorCode MatProductSetFromOptions_SeqBAIJXSMM(Mat C) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatProductSetFromOptions_SeqBAIJXSMM called\n"));
    Mat_Product *product = C->product;
    Mat          B = product->B;
    PetscBool    isdense;

    PetscFunctionBegin;
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSE, &isdense));
    
    if (isdense && product->type == MATPRODUCT_AB) {
        C->ops->productsymbolic = XSMMSpMM_Symbolic;
    } else {
        printf("FALLBACK\n");
        /* FALLBACK LOGIC */
        PetscErrorCode (*f)(Mat);
        // We ask the object C for the function pointer registered by the base SeqBAIJ class
        PetscCall(PetscObjectQueryFunction((PetscObject)C, "MatProductSetFromOptions_seqbaij_C", &f));
        
        if (f) {
            PetscCall((*f)(C));
        } else {
            // Fallback to the generic BAIJ name if the specific Seq one isn't found
            PetscCall(PetscObjectQueryFunction((PetscObject)C, "MatProductSetFromOptions_baij_C", &f));
            if (f) PetscCall((*f)(C));
        }
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Reverts a SeqBAIJXSMM matrix back to a standard SeqBAIJ */
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJXSMM_SeqBAIJ(Mat A, MatType type, MatReuse reuse, Mat *newmat) {
    Mat B = *newmat;

    PetscFunctionBegin;
    if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));

    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqbaijxsmm_seqbaij_C", NULL));
    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJ));

    *newmat = B;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Converts a standard SeqBAIJ into a SeqBAIJXSMM */
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJXSMM(Mat A, MatType type, MatReuse reuse, Mat *newmat) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "[DIAG] MatConvert_SeqBAIJ_SeqBAIJXSMM called\n"));
    Mat B = *newmat;

    PetscFunctionBegin;
    if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
    }

    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJXSMM));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqbaijxsmm_seqdense_C", MatProductSetFromOptions_SeqBAIJXSMM));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqbaijxsmm_seqbaij_C", MatConvert_SeqBAIJXSMM_SeqBAIJ));

    *newmat = B;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Factory function for creating the matrix */
PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJXSMM(Mat A) {
    PetscFunctionBegin;
    PetscCall(MatSetType(A, MATSEQBAIJ));
    PetscCall(MatConvert_SeqBAIJ_SeqBAIJXSMM(A, MATSEQBAIJXSMM, MAT_INPLACE_MATRIX, &A));
    PetscFunctionReturn(PETSC_SUCCESS);
}