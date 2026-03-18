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
    PetscInt           bs, lda;
    libxsmm_gemm_param param;

    /* Map PETSc's internal arrays */
    PetscScalar    *v = ctx->a;
    const PetscInt *col = ctx->j;
    const PetscInt *ptr = ctx->i;
    PetscInt       block_size = A->rmap->bs;
    PetscInt       num_block_rows = ctx->mbs;

    PetscFunctionBegin;
    bs = block_size;
    PetscCall(MatDenseGetArrayRead(B, &px));
    PetscCall(MatDenseGetLDA(B, &lda));
    PetscCall(MatDenseGetArray(C, &py));
    PetscCall(MatZeroEntries(C)); // Ensure C is zeroed before accumulation

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
    PetscInt         m, k_B, lda, block_size;
    XSMM_ProductCtx *pctx;

    PetscFunctionBegin;

    // Get dimensions directly from the Mat objects
    m = A->rmap->n;
    block_size = A->rmap->bs;
    PetscCall(MatGetLocalSize(B, NULL, &k_B));
    PetscCall(MatDenseGetLDA(B, &lda));

    // Define the layout of the result matrix C (Dense)
    PetscCall(MatSetSizes(C, m, k_B, PETSC_DETERMINE, k_B));
    PetscCall(MatSetType(C, MATSEQDENSE));
    PetscCall(MatSetUp(C));

    PetscCall(PetscNew(&pctx));
    pctx->K = k_B;

    // Dispatch the kernel
    libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
        block_size, pctx->K, block_size,
        block_size, lda, lda, 
        LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64
    );
    pctx->kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE);

    // Attach user logic to the PETSc product object
    C->product->data       = pctx;
    C->product->destroy    = XSMMSpMM_Destroy;
    C->ops->productnumeric = XSMMSpMM_Numeric;

    PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- PETSc Integration Hooks --- */

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqBAIJ(Mat);

/* Route MatMatMult to XSMM if Dense, else fallback */
static PetscErrorCode MatProductSetFromOptions_SeqBAIJXSMM(Mat C) {
    Mat_Product *product = C->product;
    Mat          B = product->B;
    PetscBool    isdense;

    PetscFunctionBegin;
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQDENSE, &isdense));
    
    if (isdense && product->type == MATPRODUCT_AB) {
        C->ops->productsymbolic = XSMMSpMM_Symbolic;
    } else {
        PetscCall(MatProductSetFromOptions_SeqBAIJ(C));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Converts a standard SeqBAIJ into a SeqBAIJXSMM */
PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJXSMM(Mat A, MatType type, MatReuse reuse, Mat *newmat) {
    Mat B = *newmat;

    PetscFunctionBegin;
    if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
    }

    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJXSMM));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_C", MatProductSetFromOptions_SeqBAIJXSMM));

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