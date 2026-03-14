# Exercises

Small copy-paste queries for getting comfortable with `ndatafusion`.

These exercises use the `make_*` functions so they can be pasted directly into SQL with no
preloaded Arrow extension columns. If your data source already emits canonical Arrow values, you
can usually remove the constructor call and pass the column directly.

Use the same Rust harness from [README.md](README.md), replacing only the SQL string.

## 1. Warm Up: Dot Product

Two vectors. One scalar answer.

```sql
SELECT vector_dot(
    make_vector([3.0, 4.0], 2),
    make_vector([4.0, 0.0], 2)
) AS dot
```

Expected result:

```text
dot = 12.0
```

## 2. Unit-Normalize An Embedding

Normalize a vector and confirm its length is `1`.

```sql
SELECT vector_l2_norm(
    vector_normalize(make_vector([3.0, 4.0], 2))
) AS unit_norm
```

Expected result:

```text
unit_norm = 1.0
```

## 3. Compare Two Embeddings

Cosine similarity and cosine distance are row-wise embedding primitives.

```sql
SELECT
    vector_cosine_similarity(
        make_vector([1.0, 0.0, 0.0], 3),
        make_vector([0.0, 1.0, 0.0], 3)
    ) AS similarity,
    vector_cosine_distance(
        make_vector([1.0, 0.0, 0.0], 3),
        make_vector([0.0, 1.0, 0.0], 3)
    ) AS distance
```

Expected result:

```text
similarity = 0.0
distance   = 1.0
```

## 4. Matrix Times Vector

Take a small dense matrix and multiply it by a vector.

```sql
SELECT matrix_matvec(
    make_matrix([2.0, 0.0, 0.0, 1.0], 2, 2),
    make_vector([4.0, 3.0], 2)
) AS product
```

Expected result:

```text
product = [8.0, 3.0]
```

## 5. Determinant In One Line

```sql
SELECT matrix_determinant(
    make_matrix([9.0, 0.0, 0.0, 4.0], 2, 2)
) AS det
```

Expected result:

```text
det = 36.0
```

## 6. Peek Inside QR

Some UDFs return structs. This one gives `q`, `r`, and `rank`.

```sql
SELECT qr.rank
FROM (
    SELECT matrix_qr(
        make_matrix([1.0, 2.0, 3.0, 4.0], 2, 2)
    ) AS qr
) AS t
```

Expected result:

```text
rank = 2
```

## 7. Sparse Matrix Times Dense Vector

Build a CSR sparse batch from parts and multiply it by a dense vector.

```sql
SELECT sparse_matvec(
    make_csr_matrix_batch(
        [[2, 3]],
        [[0, 2, 3]],
        [[0, 2, 1]],
        [[1.0, 2.0, 3.0]]
    ),
    make_variable_tensor([[1.0, 2.0, 3.0]], [[3]], 1)
) AS result
```

Expected result:

```text
result = [7.0, 6.0]
```

## 8. Tensor Reduction

Reduce across the last axis.

```sql
SELECT tensor_sum_last_axis(
    make_tensor([1.0, 2.0, 3.0, 4.0], 2, 2)
) AS reduced
```

Expected result:

```text
reduced = [3.0, 7.0]
```

## 9. Tensor Axis Shuffle

Permute a `2 x 3` tensor into a `3 x 2` tensor.

```sql
SELECT tensor_permute_axes(
    make_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
    1,
    0
) AS permuted
```

Expected result:

```text
permuted = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
```

## 10. Tiny PCA Workflow

Fit PCA, then use the returned struct.

```sql
SELECT
    pca.explained_variance_ratio AS explained,
    matrix_pca_transform(matrix, pca) AS scores
FROM (
    SELECT
        make_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2) AS matrix,
        matrix_pca(make_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2)) AS pca
) AS t
```

This is a good first example of the “struct-returning UDF plus follow-on UDF” style in the
catalog.

## Direct Arrow Inputs

If your upstream source already produces the right Arrow contract, constructors are not required.

Examples:

- Dense vectors: `FixedSizeList<Float32|Float64>(D)` can go straight into `vector_dot`,
  `vector_l2_norm`, `vector_cosine_similarity`, and `vector_normalize`.
- Dense matrices and fixed-shape tensors: `arrow.fixed_shape_tensor` can go straight into matrix
  and tensor UDFs.
- Variable-shape tensors: `arrow.variable_shape_tensor` can go straight into the variable tensor
  UDF family.
- Sparse batches: `ndarrow.csr_matrix_batch` can go straight into `sparse_matvec`,
  `sparse_matmat_dense`, `sparse_matmat_sparse`, `sparse_transpose`, and `sparse_lu_solve`.
