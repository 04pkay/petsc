#include <../src/mat/impls/baij/mpi/mpibaij.h> /*I   "petscmat.h"   I*/

static PetscErrorCode       MatProductSetFromOptions_MPIBAIJXSMM(Mat C);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIBAIJ(Mat);

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJXSMM_MPIBAIJ(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat          B = *newmat;
  Mat_MPIBAIJ *mpibaij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  }

  mpibaij = (Mat_MPIBAIJ *)B->data;
  PetscCall(MatConvert(mpibaij->A, MATSEQBAIJ, MAT_INPLACE_MATRIX, &mpibaij->A));
  PetscCall(MatConvert(mpibaij->B, MATSEQBAIJ, MAT_INPLACE_MATRIX, &mpibaij->B));

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijxsmm_mpidense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijxsmm_mpibaij_C", NULL));

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJ));

  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPIBAIJXSMM(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat          B = *newmat;
  Mat_MPIBAIJ *mpibaij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  }

  /* Convert the local diagonal and off-diagonal blocks to SeqBAIJXSMM */
  mpibaij = (Mat_MPIBAIJ *)B->data;
  PetscCall(MatConvert(mpibaij->A, MATSEQBAIJXSMM, MAT_INPLACE_MATRIX, &mpibaij->A));
  PetscCall(MatConvert(mpibaij->B, MATSEQBAIJXSMM, MAT_INPLACE_MATRIX, &mpibaij->B));

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIBAIJXSMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpibaijxsmm_mpidense_C", MatProductSetFromOptions_MPIBAIJXSMM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpibaijxsmm_mpibaij_C", MatConvert_MPIBAIJXSMM_MPIBAIJ));

  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_MPIBAIJXSMM(Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatProductSetFromOptions_MPIBAIJ(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIBAIJXSMM(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(A, MATMPIBAIJ));
  PetscCall(MatConvert_MPIBAIJ_MPIBAIJXSMM(A, MATMPIBAIJXSMM, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
