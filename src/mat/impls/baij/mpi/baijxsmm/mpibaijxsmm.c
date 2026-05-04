#include <../src/mat/impls/baij/mpi/mpibaij.h> /*I   "petscmat.h"   I*/

/* Forward declaration of the product dispatcher */
static PetscErrorCode MatProductSetFromOptions_MPIBAIJXSMM(Mat C);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIBAIJ(Mat);

/* Reverts back to standard MPIBAIJ */
PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJXSMM_MPIBAIJ(Mat A, MatType type, MatReuse reuse, Mat *newmat) {
    Mat          B = *newmat;
    Mat_MPIBAIJ *mpibaij;

    PetscFunctionBegin;
    if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
    }

    mpibaij = (Mat_MPIBAIJ*)B->data;
    /* Convert sub-matrices back to standard SeqBAIJ */
    PetscCall(MatConvert(mpibaij->A, MATSEQBAIJ, MAT_INPLACE_MATRIX, &mpibaij->A));
    PetscCall(MatConvert(mpibaij->B, MATSEQBAIJ, MAT_INPLACE_MATRIX, &mpibaij->B));

    /* Remove the XSMM-specific functions */
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijxsmm_mpidense_C", NULL));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijxsmm_mpibaij_C", NULL));
    
    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJ));

    *newmat = B;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Converts a standard MPIBAIJ into a MPIBAIJXSMM */
PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIBAIJXSMM(Mat A, MatType type, MatReuse reuse, Mat *newmat) {
    Mat          B = *newmat;
    Mat_MPIBAIJ *mpibaij;

    PetscFunctionBegin;
    if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
    }

    /* 1. Convert the local diagonal and off-diagonal blocks to SeqBAIJXSMM */
    mpibaij = (Mat_MPIBAIJ*)B->data;
    PetscCall(MatConvert(mpibaij->A, MATSEQBAIJXSMM, MAT_INPLACE_MATRIX, &mpibaij->A));
    PetscCall(MatConvert(mpibaij->B, MATSEQBAIJXSMM, MAT_INPLACE_MATRIX, &mpibaij->B));

    /* 2. Set the new type name */
    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJXSMM));

    /* 3. Compose the product dispatcher for MPI SpMM */
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijxsmm_mpidense_C", MatProductSetFromOptions_MPIBAIJXSMM));
    
    /* 4. Compose the conversion back to standard MPIBAIJ */
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijxsmm_mpibaij_C", MatConvert_MPIBAIJXSMM_MPIBAIJ));

    *newmat = B;
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Dispatcher for MPI Product. 
   In PETSc, MPIBAIJ * MPIDense usually defaults to a generic routine that calls 
   MatMatMult on the local A and B matrices. By setting this up, we ensure 
   the XSMM-accelerated local parts are used.
*/

static PetscErrorCode MatProductSetFromOptions_MPIBAIJXSMM(Mat C) {
    PetscFunctionBegin;
    /* For MPIBAIJXSMM, we can actually rely on the standard MPIBAIJ-Dense 
       symbolic/numeric logic because it internally calls MatMatMult() on 
       the sub-matrices (mpibaij->A and mpibaij->B). 
       Since we converted those to SeqBAIJXSMM, they will use your XSMM kernels.
    */
    PetscCall(MatProductSetFromOptions_MPIBAIJ(C));
    PetscFunctionReturn(PETSC_SUCCESS);
}

/* Factory */
PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJXSMM(Mat A) {
    PetscFunctionBegin;
    PetscCall(MatSetType(A, MATMPIBAIJ));
    PetscCall(MatConvert_MPIBAIJ_MPIBAIJXSMM(A, MATMPIBAIJXSMM, MAT_INPLACE_MATRIX, &A));
    PetscFunctionReturn(PETSC_SUCCESS);
}