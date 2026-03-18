// mpicc -o kron kron.c -I/Users/pascalkessler/petsc/include -I/Users/pascalkessler/petsc/arch-opt/include -L/Users/pascalkessler/petsc/arch-opt/lib -lpetsc

#include <petscmat.h>
#include <petscviewer.h>

static char help[] = "Reads AIJ matrix A, computes K = A (kron) B (random), saves K as BAIJ.\n\n";

int main(int argc, char **args)
{
  Mat             A, K;
  PetscViewer     viewer;
  char            infile[PETSC_MAX_PATH_LEN], outfile[PETSC_MAX_PATH_LEN];
  PetscInt        bs = 2, i, j, k, M, N, m, n, ncols;
  const PetscInt  *cols;
  const PetscScalar *vals;
  PetscScalar     *B_vals, *Work_block;
  PetscRandom     rand;
  PetscBool       flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  /* 1. Get arguments */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fin", infile, sizeof(infile), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify input file with -fin");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-fout", outfile, sizeof(outfile), &flg));
  if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Must specify output file with -fout");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));

  /* 2. Load Matrix A (AIJ) */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, infile, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  
  /* Get dimensions of A */
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));

  /* 3. Create Random Block B (bs x bs) */
  PetscCall(PetscMalloc2(bs*bs, &B_vals, bs*bs, &Work_block));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rand));
  for (i = 0; i < bs*bs; i++) {
    PetscCall(PetscRandomGetValue(rand, &B_vals[i]));
  }
  PetscCall(PetscRandomDestroy(&rand));

  /* 4. Create Matrix K (BAIJ) 
     The dimensions are scaled by bs, but the block-dimensions match A */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &K));
  PetscCall(MatSetSizes(K, m*bs, n*bs, M*bs, N*bs));
  PetscCall(MatSetType(K, MATBAIJ));
  PetscCall(MatSetBlockSize(K, bs));
  /* Note: For maximum performance on huge matrices, you should preallocate K 
     by copying the nnz structure from A. We skip this for 'minimal' code. */
  PetscCall(MatSetUp(K));

  /* 5. Perform Kronecker Product: K = A (x) B */
  PetscInt Istart, Iend;
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  for (i = Istart; i < Iend; i++) {
    PetscCall(MatGetRow(A, i, &ncols, &cols, &vals));
    for (j = 0; j < ncols; j++) {
      PetscScalar scalar_val = vals[j];
      
      /* Calculate the block: Work = scalar_val * B */
      for (k = 0; k < bs*bs; k++) {
        Work_block[k] = scalar_val * B_vals[k];
      }

      /* Insert block into K. 
         Note: MatSetValuesBlocked uses block indices. 
         Since A represents the block structure, we use i and cols[j] directly. */
      PetscCall(MatSetValuesBlocked(K, 1, &i, 1, &cols[j], Work_block, INSERT_VALUES));
    }
    PetscCall(MatRestoreRow(A, i, &ncols, &cols, &vals));
  }

  /* Assemble K */
  PetscCall(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));

  /* 6. Save Matrix K */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, outfile, FILE_MODE_WRITE, &viewer));
  PetscCall(MatView(K, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /* Cleanup */
  PetscCall(PetscFree2(B_vals, Work_block));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&K));
  PetscCall(PetscFinalize());
  return 0;
}