use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{Float32Type, Float64Type, Int64Type};
use datafusion::arrow::array::{Array, Int64Array, ListArray, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use ndarray::{Ix1, Ix2};
use ndarrow::NdarrowElement;

use super::common::{expect_real_scalar_arg, expect_struct_arg, nullable_or};
use super::docs::sparse_doc;
use crate::error::{exec_error, plan_error};
use crate::metadata::{
    csr_matrix_batch_field, parse_csr_matrix_batch_field, parse_variable_shape_tensor_field,
    struct_field, variable_shape_tensor_field,
};
use crate::signatures::{
    ScalarCoercion, any_signature, coerce_scalar_arguments, named_user_defined_signature,
};

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparseFactorPairContract {
    value_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SparseLuFactorContract {
    value_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct JacobiPreconditionerContract {
    value_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IlukFactorContract {
    value_type: DataType,
}

type OwnedCsrBatchParts<T> = ([usize; 2], Vec<i32>, Vec<u32>, Vec<T>);
type OwnedSparseFactorPairs<T> =
    Vec<(nabled::linalg::sparse::CsrMatrix<T>, nabled::linalg::sparse::CsrMatrix<T>)>;

fn int64_list_field(name: &str, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, DataType::new_list(DataType::Int64, true), nullable))
}

fn parse_int64_list_field(field: &FieldRef, function_name: &str, position: usize) -> Result<()> {
    let DataType::List(inner) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} must store List<Int64> permutation vectors, found {}",
                field.data_type()
            ),
        ));
    };
    if inner.data_type() != &DataType::Int64 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} must store List<Int64> permutation vectors, found List<{}>",
                inner.data_type()
            ),
        ));
    }
    Ok(())
}

fn sparse_factor_pair_struct_field(
    name: &str,
    value_type: &DataType,
    nullable: bool,
) -> Result<FieldRef> {
    let lower = csr_matrix_batch_field("l", value_type, false)?;
    let upper = csr_matrix_batch_field("u", value_type, false)?;
    Ok(struct_field(name, vec![lower.as_ref().clone(), upper.as_ref().clone()], nullable))
}

fn sparse_lu_factor_struct_field(
    name: &str,
    value_type: &DataType,
    nullable: bool,
) -> Result<FieldRef> {
    let lower = csr_matrix_batch_field("l", value_type, false)?;
    let upper = csr_matrix_batch_field("u", value_type, false)?;
    let permutation = int64_list_field("permutation", false);
    Ok(struct_field(
        name,
        vec![lower.as_ref().clone(), upper.as_ref().clone(), permutation.as_ref().clone()],
        nullable,
    ))
}

fn jacobi_preconditioner_struct_field(
    name: &str,
    value_type: &DataType,
    nullable: bool,
) -> Result<FieldRef> {
    let inverse_diagonal =
        variable_shape_tensor_field("inverse_diagonal", value_type, 1, Some(&[None]), false)?;
    Ok(struct_field(name, vec![inverse_diagonal.as_ref().clone()], nullable))
}

fn iluk_factor_struct_field(name: &str, value_type: &DataType, nullable: bool) -> Result<FieldRef> {
    let lower = csr_matrix_batch_field("l", value_type, false)?;
    let upper = csr_matrix_batch_field("u", value_type, false)?;
    let level = Field::new("level_of_fill", DataType::Int64, false);
    Ok(struct_field(name, vec![lower.as_ref().clone(), upper.as_ref().clone(), level], nullable))
}

fn parse_sparse_factor_pair_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<SparseFactorPairContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a sparse factorization struct"),
        ));
    };
    if fields.len() != 2 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} sparse factorization struct must have 2 fields, found {}",
                fields.len()
            ),
        ));
    }
    if fields[0].name() != "l" || fields[1].name() != "u" {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} sparse factorization fields must be named l and u, found {} \
                 and {}",
                fields[0].name(),
                fields[1].name()
            ),
        ));
    }
    let lower_type =
        parse_csr_matrix_batch_field(&Arc::clone(&fields[0]), function_name, position)?;
    let upper_type =
        parse_csr_matrix_batch_field(&Arc::clone(&fields[1]), function_name, position)?;
    if lower_type != upper_type {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} factorization value type mismatch: l {lower_type}, u \
                 {upper_type}",
            ),
        ));
    }
    Ok(SparseFactorPairContract { value_type: lower_type })
}

fn parse_sparse_lu_factor_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<SparseLuFactorContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a sparse LU factorization struct"),
        ));
    };
    if fields.len() != 3 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} sparse LU factorization struct must have 3 fields, found {}",
                fields.len()
            ),
        ));
    }
    if fields[0].name() != "l" || fields[1].name() != "u" || fields[2].name() != "permutation" {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} sparse LU factorization fields must be named l, u, \
                 permutation"
            ),
        ));
    }
    let lower_type =
        parse_csr_matrix_batch_field(&Arc::clone(&fields[0]), function_name, position)?;
    let upper_type =
        parse_csr_matrix_batch_field(&Arc::clone(&fields[1]), function_name, position)?;
    parse_int64_list_field(&Arc::clone(&fields[2]), function_name, position)?;
    if lower_type != upper_type {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} factorization value type mismatch: l {lower_type}, u \
                 {upper_type}",
            ),
        ));
    }
    Ok(SparseLuFactorContract { value_type: lower_type })
}

fn parse_jacobi_preconditioner_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<JacobiPreconditionerContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a Jacobi preconditioner struct"),
        ));
    };
    if fields.len() != 1 || fields[0].name() != "inverse_diagonal" {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} Jacobi preconditioner struct must contain inverse_diagonal"
            ),
        ));
    }
    let contract =
        parse_variable_shape_tensor_field(&Arc::clone(&fields[0]), function_name, position)?;
    if contract.dimensions != 1 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} inverse_diagonal must be a rank-1 variable-shape tensor \
                 batch, found rank {}",
                contract.dimensions
            ),
        ));
    }
    Ok(JacobiPreconditionerContract { value_type: contract.value_type })
}

fn parse_iluk_factor_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<IlukFactorContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be an ILU(k) factorization struct"),
        ));
    };
    if fields.len() != 3 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} ILU(k) factorization struct must have 3 fields, found {}",
                fields.len()
            ),
        ));
    }
    if fields[0].name() != "l" || fields[1].name() != "u" || fields[2].name() != "level_of_fill" {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} ILU(k) factorization fields must be named l, u, level_of_fill"
            ),
        ));
    }
    let lower_type =
        parse_csr_matrix_batch_field(&Arc::clone(&fields[0]), function_name, position)?;
    let upper_type =
        parse_csr_matrix_batch_field(&Arc::clone(&fields[1]), function_name, position)?;
    if fields[2].data_type() != &DataType::Int64 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} level_of_fill field must use Int64, found {}",
                fields[2].data_type()
            ),
        ));
    }
    if lower_type != upper_type {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} factorization value type mismatch: l {lower_type}, u \
                 {upper_type}",
            ),
        ));
    }
    Ok(IlukFactorContract { value_type: lower_type })
}

fn owned_csr_from_view<T, R, C>(
    view: &nabled::linalg::sparse::CsrMatrixView<'_, R, T, C>,
) -> Result<nabled::linalg::sparse::CsrMatrix<T>>
where
    T: nabled::core::prelude::NabledReal + Copy,
    R: nabled::linalg::sparse::CsrIndex,
    C: nabled::linalg::sparse::CsrIndex,
{
    let mut indptr = Vec::with_capacity(view.row_ptrs.len());
    for &index in view.row_ptrs {
        indptr.push(index.to_usize().map_err(|error| exec_error("sparse_factorization", error))?);
    }
    let mut indices = Vec::with_capacity(view.col_indices.len());
    for &index in view.col_indices {
        indices.push(index.to_usize().map_err(|error| exec_error("sparse_factorization", error))?);
    }
    Ok(nabled::linalg::sparse::CsrMatrix {
        nrows: view.nrows,
        ncols: view.ncols,
        indptr,
        indices,
        data: view.values.to_vec(),
    })
}

fn csr_matrix_to_batch_parts<T: nabled::core::prelude::NabledReal>(
    matrix: nabled::linalg::sparse::CsrMatrix<T>,
    function_name: &str,
) -> Result<OwnedCsrBatchParts<T>> {
    let row_ptrs = matrix
        .indptr
        .into_iter()
        .map(|index| {
            i32::try_from(index).map_err(|_| {
                exec_error(function_name, format!("row pointer {index} exceeds i32 limits"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let col_indices = matrix
        .indices
        .into_iter()
        .map(|index| {
            u32::try_from(index).map_err(|_| {
                exec_error(function_name, format!("column index {index} exceeds u32 limits"))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(([matrix.nrows, matrix.ncols], row_ptrs, col_indices, matrix.data))
}

fn build_csr_batch_array<T: nabled::core::prelude::NabledReal + NdarrowElement>(
    name: &str,
    matrices: Vec<nabled::linalg::sparse::CsrMatrix<T>>,
) -> Result<(FieldRef, Arc<StructArray>)> {
    let mut shapes = Vec::with_capacity(matrices.len());
    let mut row_ptrs = Vec::with_capacity(matrices.len());
    let mut col_indices = Vec::with_capacity(matrices.len());
    let mut values = Vec::with_capacity(matrices.len());
    for matrix in matrices {
        let (shape, row_ptr, cols, row_values) = csr_matrix_to_batch_parts(matrix, name)?;
        shapes.push(shape);
        row_ptrs.push(row_ptr);
        col_indices.push(cols);
        values.push(row_values);
    }
    let (field, array) =
        ndarrow::csr_batch_to_extension_array(name, shapes, row_ptrs, col_indices, values)
            .map_err(|error| exec_error(name, error))?;
    Ok((Arc::new(field), Arc::new(array)))
}

fn build_rank1_variable_batch<T: NdarrowElement>(
    name: &str,
    rows: Vec<ndarray::Array1<T>>,
) -> Result<(FieldRef, Arc<StructArray>)> {
    let arrays = rows.into_iter().map(ndarray::ArrayBase::into_dyn).collect::<Vec<_>>();
    let (field, array) = ndarrow::arrays_to_variable_shape_tensor(name, arrays, Some(vec![None]))
        .map_err(|error| exec_error(name, error))?;
    Ok((Arc::new(field), Arc::new(array)))
}

fn permutation_list_array(rows: Vec<Vec<i64>>) -> Arc<ListArray> {
    Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
    ))
}

fn build_sparse_lu_factorization_output<T>(
    function_name: &str,
    factorizations: Vec<nabled::linalg::sparse::SparseLUFactorization<T>>,
) -> Result<ColumnarValue>
where
    T: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let mut lower = Vec::with_capacity(factorizations.len());
    let mut upper = Vec::with_capacity(factorizations.len());
    let mut permutations = Vec::with_capacity(factorizations.len());
    for factorization in factorizations {
        lower.push(factorization.l);
        upper.push(factorization.u);
        let permutation = factorization
            .permutation
            .into_iter()
            .map(|value| {
                i64::try_from(value).map_err(|_| {
                    exec_error(
                        function_name,
                        format!("permutation index {value} exceeds i64 limits"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        permutations.push(permutation);
    }
    let (lower_field, lower_array) = build_csr_batch_array("l", lower)?;
    let (upper_field, upper_array) = build_csr_batch_array("u", upper)?;
    let permutation_field = int64_list_field("permutation", false);
    let permutation_array = permutation_list_array(permutations);
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![lower_field, upper_field, permutation_field].into(),
        vec![lower_array, upper_array, permutation_array],
        None,
    ))))
}

fn build_sparse_factor_pair_output<T>(
    _function_name: &str,
    lower_name: &str,
    upper_name: &str,
    factors: Vec<(nabled::linalg::sparse::CsrMatrix<T>, nabled::linalg::sparse::CsrMatrix<T>)>,
) -> Result<ColumnarValue>
where
    T: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let mut lower = Vec::with_capacity(factors.len());
    let mut upper = Vec::with_capacity(factors.len());
    for (l, u) in factors {
        lower.push(l);
        upper.push(u);
    }
    let (lower_field, lower_array) = build_csr_batch_array(lower_name, lower)?;
    let (upper_field, upper_array) = build_csr_batch_array(upper_name, upper)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![lower_field, upper_field].into(),
        vec![lower_array, upper_array],
        None,
    ))))
}

fn build_jacobi_preconditioner_output<T>(
    preconditioners: Vec<nabled::linalg::sparse::JacobiPreconditioner<T>>,
) -> Result<ColumnarValue>
where
    T: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let rows = preconditioners
        .into_iter()
        .map(|preconditioner| preconditioner.inverse_diagonal)
        .collect::<Vec<_>>();
    let (field, array) = build_rank1_variable_batch("inverse_diagonal", rows)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(vec![field].into(), vec![array], None))))
}

fn build_iluk_factor_output<T>(
    function_name: &str,
    factorizations: Vec<nabled::linalg::sparse::ILUKFactorization<T>>,
) -> Result<ColumnarValue>
where
    T: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let mut lower = Vec::with_capacity(factorizations.len());
    let mut upper = Vec::with_capacity(factorizations.len());
    let mut levels = Vec::with_capacity(factorizations.len());
    for factorization in factorizations {
        lower.push(factorization.l);
        upper.push(factorization.u);
        levels.push(
            i64::try_from(factorization.level_of_fill)
                .map_err(|_| exec_error(function_name, "level_of_fill exceeds i64 limits"))?,
        );
    }
    let (lower_field, lower_array) = build_csr_batch_array("l", lower)?;
    let (upper_field, upper_array) = build_csr_batch_array("u", upper)?;
    let level_field = Arc::new(Field::new("level_of_fill", DataType::Int64, false));
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![lower_field, upper_field, level_field].into(),
        vec![lower_array, upper_array, Arc::new(Int64Array::from(levels))],
        None,
    ))))
}

fn sparse_matrix_vector_output_field(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
    factorization_type: &DataType,
    rhs_position: usize,
) -> Result<FieldRef> {
    let rhs = parse_variable_shape_tensor_field(
        &args.arg_fields[rhs_position - 1],
        function_name,
        rhs_position,
    )?;
    if &rhs.value_type != factorization_type {
        return Err(plan_error(
            function_name,
            format!(
                "value type mismatch: factorization {}, rhs {}",
                factorization_type, rhs.value_type
            ),
        ));
    }
    if rhs.dimensions != 1 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {rhs_position} must be a batch of rank-1 dense vectors, found rank {}",
                rhs.dimensions
            ),
        ));
    }
    variable_shape_tensor_field(
        function_name,
        &rhs.value_type,
        1,
        Some(&[None]),
        nullable_or(args.arg_fields),
    )
}

fn sparse_matrix_matrix_output_field(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
    factorization_type: &DataType,
    rhs_position: usize,
) -> Result<FieldRef> {
    let rhs = parse_variable_shape_tensor_field(
        &args.arg_fields[rhs_position - 1],
        function_name,
        rhs_position,
    )?;
    if &rhs.value_type != factorization_type {
        return Err(plan_error(
            function_name,
            format!(
                "value type mismatch: factorization {}, rhs {}",
                factorization_type, rhs.value_type
            ),
        ));
    }
    if rhs.dimensions != 2 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {rhs_position} must be a batch of rank-2 dense matrices, found rank {}",
                rhs.dimensions
            ),
        ));
    }
    variable_shape_tensor_field(
        function_name,
        &rhs.value_type,
        2,
        Some(&[None, None]),
        nullable_or(args.arg_fields),
    )
}

fn owned_sparse_lu_factorizations<T>(
    field: &FieldRef,
    struct_array: &StructArray,
    function_name: &str,
) -> Result<Vec<nabled::linalg::sparse::SparseLUFactorization<T::Native>>>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let DataType::Struct(fields) = field.data_type() else {
        return Err(exec_error(function_name, "expected sparse LU factorization struct field"));
    };
    let lower = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| exec_error(function_name, "expected StructArray storage for l"))?;
    let upper = struct_array
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| exec_error(function_name, "expected StructArray storage for u"))?;
    let permutation =
        struct_array.column(2).as_any().downcast_ref::<ListArray>().ok_or_else(|| {
            exec_error(function_name, "expected ListArray storage for permutation")
        })?;
    let mut lower_iter = ndarrow::csr_matrix_batch_iter::<T>(fields[0].as_ref(), lower)
        .map_err(|error| exec_error(function_name, error))?;
    let mut upper_iter = ndarrow::csr_matrix_batch_iter::<T>(fields[1].as_ref(), upper)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(struct_array.len());
    for row in 0..struct_array.len() {
        let (_, lower_view) = lower_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "lower factor iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let (_, upper_view) = upper_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "upper factor iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let permutation_values = permutation.value(row);
        let permutation_view = permutation_values
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| exec_error(function_name, "expected Int64Array permutation values"))?;
        let lower_view = nabled::linalg::sparse::CsrMatrixView::new(
            lower_view.nrows,
            lower_view.ncols,
            lower_view.row_ptrs,
            lower_view.col_indices,
            lower_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        let upper_view = nabled::linalg::sparse::CsrMatrixView::new(
            upper_view.nrows,
            upper_view.ncols,
            upper_view.row_ptrs,
            upper_view.col_indices,
            upper_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        let permutation = permutation_view
            .values()
            .iter()
            .copied()
            .map(|value| {
                usize::try_from(value).map_err(|_| {
                    exec_error(
                        function_name,
                        format!("permutation index must be non-negative, found {value}"),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        outputs.push(nabled::linalg::sparse::SparseLUFactorization {
            l: owned_csr_from_view(&lower_view)?,
            u: owned_csr_from_view(&upper_view)?,
            permutation,
        });
    }
    Ok(outputs)
}

fn owned_jacobi_preconditioners<T>(
    field: &FieldRef,
    struct_array: &StructArray,
    function_name: &str,
) -> Result<Vec<nabled::linalg::sparse::JacobiPreconditioner<T::Native>>>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let DataType::Struct(fields) = field.data_type() else {
        return Err(exec_error(function_name, "expected Jacobi preconditioner struct field"));
    };
    let inverse_diagonal =
        struct_array.column(0).as_any().downcast_ref::<StructArray>().ok_or_else(|| {
            exec_error(function_name, "expected StructArray storage for inverse_diagonal")
        })?;
    let mut iter = ndarrow::variable_shape_tensor_iter::<T>(fields[0].as_ref(), inverse_diagonal)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(struct_array.len());
    for _ in 0..struct_array.len() {
        let (_, inverse_diagonal_view) = iter
            .next()
            .ok_or_else(|| exec_error(function_name, "inverse_diagonal iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let inverse_diagonal_view = inverse_diagonal_view
            .into_dimensionality::<Ix1>()
            .map_err(|error| exec_error(function_name, error))?;
        outputs.push(nabled::linalg::sparse::JacobiPreconditioner {
            inverse_diagonal: inverse_diagonal_view.to_owned(),
        });
    }
    Ok(outputs)
}

fn owned_sparse_factor_pairs<T>(
    field: &FieldRef,
    struct_array: &StructArray,
    function_name: &str,
) -> Result<OwnedSparseFactorPairs<T::Native>>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let DataType::Struct(fields) = field.data_type() else {
        return Err(exec_error(function_name, "expected sparse factorization struct field"));
    };
    let lower = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| exec_error(function_name, "expected StructArray storage for l"))?;
    let upper = struct_array
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| exec_error(function_name, "expected StructArray storage for u"))?;
    let mut lower_iter = ndarrow::csr_matrix_batch_iter::<T>(fields[0].as_ref(), lower)
        .map_err(|error| exec_error(function_name, error))?;
    let mut upper_iter = ndarrow::csr_matrix_batch_iter::<T>(fields[1].as_ref(), upper)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(struct_array.len());
    for _ in 0..struct_array.len() {
        let (_, lower_view) = lower_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "lower factor iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let (_, upper_view) = upper_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "upper factor iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let lower_view = nabled::linalg::sparse::CsrMatrixView::new(
            lower_view.nrows,
            lower_view.ncols,
            lower_view.row_ptrs,
            lower_view.col_indices,
            lower_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        let upper_view = nabled::linalg::sparse::CsrMatrixView::new(
            upper_view.nrows,
            upper_view.ncols,
            upper_view.row_ptrs,
            upper_view.col_indices,
            upper_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        outputs.push((owned_csr_from_view(&lower_view)?, owned_csr_from_view(&upper_view)?));
    }
    Ok(outputs)
}

fn owned_iluk_factorizations<T>(
    field: &FieldRef,
    struct_array: &StructArray,
    function_name: &str,
) -> Result<Vec<nabled::linalg::sparse::ILUKFactorization<T::Native>>>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let DataType::Struct(fields) = field.data_type() else {
        return Err(exec_error(function_name, "expected ILU(k) factorization struct field"));
    };
    let lower = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| exec_error(function_name, "expected StructArray storage for l"))?;
    let upper = struct_array
        .column(1)
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| exec_error(function_name, "expected StructArray storage for u"))?;
    let levels = struct_array.column(2).as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        exec_error(function_name, "expected Int64Array storage for level_of_fill")
    })?;
    let mut lower_iter = ndarrow::csr_matrix_batch_iter::<T>(fields[0].as_ref(), lower)
        .map_err(|error| exec_error(function_name, error))?;
    let mut upper_iter = ndarrow::csr_matrix_batch_iter::<T>(fields[1].as_ref(), upper)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(struct_array.len());
    for row in 0..struct_array.len() {
        let (_, lower_view) = lower_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "lower factor iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let (_, upper_view) = upper_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "upper factor iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let Some(level_value) = levels.value(row).try_into().ok() else {
            return Err(exec_error(function_name, "level_of_fill exceeds usize limits"));
        };
        let lower_view = nabled::linalg::sparse::CsrMatrixView::new(
            lower_view.nrows,
            lower_view.ncols,
            lower_view.row_ptrs,
            lower_view.col_indices,
            lower_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        let upper_view = nabled::linalg::sparse::CsrMatrixView::new(
            upper_view.nrows,
            upper_view.ncols,
            upper_view.row_ptrs,
            upper_view.col_indices,
            upper_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        outputs.push(nabled::linalg::sparse::ILUKFactorization {
            l:             owned_csr_from_view(&lower_view)?,
            u:             owned_csr_from_view(&upper_view)?,
            level_of_fill: level_value,
        });
    }
    Ok(outputs)
}

fn invoke_apply_iluk_preconditioner_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let factors = expect_struct_arg(args, 1, function_name)?;
    let rhs = expect_struct_arg(args, 2, function_name)?;
    if factors.len() != rhs.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} factorizations vs {} dense vectors",
                factors.len(),
                rhs.len()
            ),
        ));
    }
    let factor_rows = owned_iluk_factorizations::<T>(&args.arg_fields[0], factors, function_name)?;
    let mut rhs_iter = ndarrow::variable_shape_tensor_iter::<T>(args.arg_fields[1].as_ref(), rhs)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(rhs.len());
    for factorization in factor_rows {
        let (_, rhs_view) = rhs_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "dense vector iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let rhs_view = rhs_view
            .into_dimensionality::<Ix1>()
            .map_err(|error| exec_error(function_name, error))?;
        outputs.push(
            nabled::linalg::sparse::apply_iluk_preconditioner(&factorization, &rhs_view)
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
        function_name,
        outputs.into_iter().map(ndarray::ArrayBase::into_dyn).collect(),
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_sparse_lu_factor_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let mut factorizations = Vec::with_capacity(matrices.len());
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
        .map_err(|error| exec_error(function_name, error))?
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        factorizations.push(
            nabled::linalg::sparse::sparse_lu_factor_view(&matrix_view)
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    build_sparse_lu_factorization_output(function_name, factorizations)
}

fn invoke_jacobi_preconditioner_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let mut preconditioners = Vec::with_capacity(matrices.len());
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
        .map_err(|error| exec_error(function_name, error))?
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        preconditioners.push(
            nabled::linalg::sparse::jacobi_preconditioner_view(&matrix_view)
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    build_jacobi_preconditioner_output(preconditioners)
}

fn invoke_apply_jacobi_preconditioner_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let preconditioners = expect_struct_arg(args, 1, function_name)?;
    let rhs = expect_struct_arg(args, 2, function_name)?;
    if preconditioners.len() != rhs.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} preconditioners vs {} dense vectors",
                preconditioners.len(),
                rhs.len()
            ),
        ));
    }
    let factorization_rows =
        owned_jacobi_preconditioners::<T>(&args.arg_fields[0], preconditioners, function_name)?;
    let mut rhs_iter = ndarrow::variable_shape_tensor_iter::<T>(args.arg_fields[1].as_ref(), rhs)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(rhs.len());
    for preconditioner in factorization_rows {
        let (_, rhs_view) = rhs_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "dense vector iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let rhs_view = rhs_view
            .into_dimensionality::<Ix1>()
            .map_err(|error| exec_error(function_name, error))?;
        outputs.push(
            nabled::linalg::sparse::apply_jacobi_preconditioner(&preconditioner, &rhs_view)
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
        function_name,
        outputs.into_iter().map(ndarray::ArrayBase::into_dyn).collect(),
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_ilut_factor_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    drop_tolerance: T::Native,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let max_fill = super::common::expect_usize_scalar_arg(args, 3, function_name)?;
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let mut factors = Vec::with_capacity(matrices.len());
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
        .map_err(|error| exec_error(function_name, error))?
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        let factorization =
            nabled::linalg::sparse::ilut_factor_view(&matrix_view, drop_tolerance, max_fill)
                .map_err(|error| exec_error(function_name, error))?;
        factors.push((factorization.l, factorization.u));
    }
    build_sparse_factor_pair_output(function_name, "l", "u", factors)
}

fn float32_scalar_from_f64(function_name: &str, value: f64, name: &str) -> Result<f32> {
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(exec_error(
            function_name,
            format!("{name} could not be represented as Float32"),
        ));
    }
    <f32 as num_traits::FromPrimitive>::from_f64(value).ok_or_else(|| {
        exec_error(function_name, format!("{name} could not be represented as Float32"))
    })
}

fn invoke_iluk_factor_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let level_of_fill = super::common::expect_usize_scalar_arg(args, 2, function_name)?;
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let mut factors = Vec::with_capacity(matrices.len());
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
        .map_err(|error| exec_error(function_name, error))?
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        factors.push(
            nabled::linalg::sparse::iluk_factor_view(&matrix_view, level_of_fill)
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    build_iluk_factor_output(function_name, factors)
}

fn invoke_apply_sparse_factor_pair_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    apply: impl Fn(
        &(
            nabled::linalg::sparse::CsrMatrix<T::Native>,
            nabled::linalg::sparse::CsrMatrix<T::Native>,
        ),
        &ndarray::ArrayView1<'_, T::Native>,
    ) -> std::result::Result<
        ndarray::Array1<T::Native>,
        nabled::linalg::sparse::SparseError,
    >,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let factors = expect_struct_arg(args, 1, function_name)?;
    let rhs = expect_struct_arg(args, 2, function_name)?;
    if factors.len() != rhs.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} factorizations vs {} dense vectors",
                factors.len(),
                rhs.len()
            ),
        ));
    }
    let factor_rows = owned_sparse_factor_pairs::<T>(&args.arg_fields[0], factors, function_name)?;
    let mut rhs_iter = ndarrow::variable_shape_tensor_iter::<T>(args.arg_fields[1].as_ref(), rhs)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(rhs.len());
    for factorization in factor_rows {
        let (_, rhs_view) = rhs_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "dense vector iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let rhs_view = rhs_view
            .into_dimensionality::<Ix1>()
            .map_err(|error| exec_error(function_name, error))?;
        outputs.push(
            apply(&factorization, &rhs_view).map_err(|error| exec_error(function_name, error))?,
        );
    }
    let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
        function_name,
        outputs.into_iter().map(ndarray::ArrayBase::into_dyn).collect(),
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_sparse_lu_solve_with_factorization_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let rhs = expect_struct_arg(args, 2, function_name)?;
    let factorizations = expect_struct_arg(args, 3, function_name)?;
    if matrices.len() != rhs.len() || matrices.len() != factorizations.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} matrices, {} rhs vectors, {} factorizations",
                matrices.len(),
                rhs.len(),
                factorizations.len()
            ),
        ));
    }
    let factor_rows =
        owned_sparse_lu_factorizations::<T>(&args.arg_fields[2], factorizations, function_name)?;
    let mut rhs_iter = ndarrow::variable_shape_tensor_iter::<T>(args.arg_fields[1].as_ref(), rhs)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(rhs.len());
    for (matrix_row, factorization) in
        ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
            .map_err(|error| exec_error(function_name, error))?
            .zip(factor_rows.into_iter())
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let (_, rhs_view) = rhs_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "dense vector iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let rhs_view = rhs_view
            .into_dimensionality::<Ix1>()
            .map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        outputs.push(
            nabled::linalg::sparse::sparse_lu_solve_with_factorization_view(
                &matrix_view,
                &rhs_view,
                &factorization,
            )
            .map_err(|error| exec_error(function_name, error))?,
        );
    }
    let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
        function_name,
        outputs.into_iter().map(ndarray::ArrayBase::into_dyn).collect(),
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_sparse_lu_solve_multiple_with_factorization_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + NdarrowElement,
{
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let rhs = expect_struct_arg(args, 2, function_name)?;
    let factorizations = expect_struct_arg(args, 3, function_name)?;
    if matrices.len() != rhs.len() || matrices.len() != factorizations.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} matrices, {} rhs matrices, {} factorizations",
                matrices.len(),
                rhs.len(),
                factorizations.len()
            ),
        ));
    }
    let factor_rows =
        owned_sparse_lu_factorizations::<T>(&args.arg_fields[2], factorizations, function_name)?;
    let mut rhs_iter = ndarrow::variable_shape_tensor_iter::<T>(args.arg_fields[1].as_ref(), rhs)
        .map_err(|error| exec_error(function_name, error))?;
    let mut outputs = Vec::with_capacity(rhs.len());
    for (matrix_row, factorization) in
        ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
            .map_err(|error| exec_error(function_name, error))?
            .zip(factor_rows.into_iter())
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let (_, rhs_view) = rhs_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "dense matrix iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let rhs_view = rhs_view
            .into_dimensionality::<Ix2>()
            .map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        outputs.push(
            nabled::linalg::sparse::sparse_lu_solve_multiple_with_factorization_view(
                &matrix_view,
                &rhs_view,
                &factorization,
            )
            .map_err(|error| exec_error(function_name, error))?,
        );
    }
    let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
        function_name,
        outputs.into_iter().map(ndarray::ArrayBase::into_dyn).collect(),
        Some(vec![None, None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseLuFactor {
    signature: Signature,
}

impl SparseLuFactor {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for SparseLuFactor {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_lu_factor" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        sparse_lu_factor_struct_field(self.name(), &value_type, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => invoke_sparse_lu_factor_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_sparse_lu_factor_typed::<Float64Type>(&args, self.name()),
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Build a reusable sparse LU factorization for each CSR matrix in the batch.",
                "sparse_lu_factor(sparse_batch)",
            )
            .with_argument("sparse_batch", "Canonical ndarrow.csr_matrix_batch column.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseLuSolveWithFactorization {
    signature: Signature,
}

impl SparseLuSolveWithFactorization {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for SparseLuSolveWithFactorization {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_lu_solve_with_factorization" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let factorization = parse_sparse_lu_factor_field(&args.arg_fields[2], self.name(), 3)?;
        if matrix_type != factorization.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {matrix_type}, factorization {}",
                    factorization.value_type
                ),
            ));
        }
        sparse_matrix_vector_output_field(&args, self.name(), &matrix_type, 2)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let factorization = parse_sparse_lu_factor_field(&args.arg_fields[2], self.name(), 3)?;
        if matrix_type != factorization.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {matrix_type}, factorization {}",
                    factorization.value_type
                ),
            ));
        }
        match matrix_type {
            DataType::Float32 => {
                invoke_sparse_lu_solve_with_factorization_typed::<Float32Type>(&args, self.name())
            }
            DataType::Float64 => {
                invoke_sparse_lu_solve_with_factorization_typed::<Float64Type>(&args, self.name())
            }
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Solve each CSR sparse linear system with a previously computed LU factorization.",
                "sparse_lu_solve_with_factorization(sparse_batch, rhs_batch, factorization)",
            )
            .with_argument("sparse_batch", "Canonical ndarrow.csr_matrix_batch column.")
            .with_argument(
                "rhs_batch",
                "Canonical rank-1 variable-shape tensor batch of dense vectors.",
            )
            .with_argument("factorization", "Struct returned by sparse_lu_factor.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseLuSolveMultipleWithFactorization {
    signature: Signature,
}

impl SparseLuSolveMultipleWithFactorization {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for SparseLuSolveMultipleWithFactorization {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_lu_solve_multiple_with_factorization" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let factorization = parse_sparse_lu_factor_field(&args.arg_fields[2], self.name(), 3)?;
        if matrix_type != factorization.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {matrix_type}, factorization {}",
                    factorization.value_type
                ),
            ));
        }
        sparse_matrix_matrix_output_field(&args, self.name(), &matrix_type, 2)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let factorization = parse_sparse_lu_factor_field(&args.arg_fields[2], self.name(), 3)?;
        if matrix_type != factorization.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {matrix_type}, factorization {}",
                    factorization.value_type
                ),
            ));
        }
        match matrix_type {
            DataType::Float32 => invoke_sparse_lu_solve_multiple_with_factorization_typed::<
                Float32Type,
            >(&args, self.name()),
            DataType::Float64 => invoke_sparse_lu_solve_multiple_with_factorization_typed::<
                Float64Type,
            >(&args, self.name()),
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Solve each CSR sparse linear system with multiple dense right-hand sides using a \
                 previously computed LU factorization.",
                "sparse_lu_solve_multiple_with_factorization(sparse_batch, rhs_batch, \
                 factorization)",
            )
            .with_argument("sparse_batch", "Canonical ndarrow.csr_matrix_batch column.")
            .with_argument(
                "rhs_batch",
                "Canonical rank-2 variable-shape tensor batch of dense matrices.",
            )
            .with_argument("factorization", "Struct returned by sparse_lu_factor.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseJacobiPreconditioner {
    signature: Signature,
}

impl SparseJacobiPreconditioner {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for SparseJacobiPreconditioner {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_jacobi_preconditioner" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        jacobi_preconditioner_struct_field(self.name(), &value_type, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => {
                invoke_jacobi_preconditioner_typed::<Float32Type>(&args, self.name())
            }
            DataType::Float64 => {
                invoke_jacobi_preconditioner_typed::<Float64Type>(&args, self.name())
            }
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Build a Jacobi preconditioner for each CSR sparse matrix in the batch.",
                "sparse_jacobi_preconditioner(sparse_batch)",
            )
            .with_argument("sparse_batch", "Canonical ndarrow.csr_matrix_batch column.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseApplyJacobiPreconditioner {
    signature: Signature,
}

impl SparseApplyJacobiPreconditioner {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseApplyJacobiPreconditioner {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_apply_jacobi_preconditioner" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let preconditioner =
            parse_jacobi_preconditioner_field(&args.arg_fields[0], self.name(), 1)?;
        sparse_matrix_vector_output_field(&args, self.name(), &preconditioner.value_type, 2)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let preconditioner =
            parse_jacobi_preconditioner_field(&args.arg_fields[0], self.name(), 1)?;
        match preconditioner.value_type {
            DataType::Float32 => {
                invoke_apply_jacobi_preconditioner_typed::<Float32Type>(&args, self.name())
            }
            DataType::Float64 => {
                invoke_apply_jacobi_preconditioner_typed::<Float64Type>(&args, self.name())
            }
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Apply a previously computed Jacobi preconditioner to each dense vector in the \
                 batch.",
                "sparse_apply_jacobi_preconditioner(preconditioner, rhs_batch)",
            )
            .with_argument("preconditioner", "Struct returned by sparse_jacobi_preconditioner.")
            .with_argument(
                "rhs_batch",
                "Canonical rank-1 variable-shape tensor batch of dense vectors.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseIlutFactor {
    signature: Signature,
}

impl SparseIlutFactor {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["matrix", "drop_tolerance", "max_fill"]) }
    }
}

impl ScalarUDFImpl for SparseIlutFactor {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_ilut_factor" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[
            (2, ScalarCoercion::Real),
            (3, ScalarCoercion::Integer),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        sparse_factor_pair_struct_field(self.name(), &value_type, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let drop_tolerance = expect_real_scalar_arg(&args, 2, self.name())?;
        match parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => invoke_ilut_factor_typed::<Float32Type>(
                &args,
                self.name(),
                float32_scalar_from_f64(self.name(), drop_tolerance, "drop_tolerance")?,
            ),
            DataType::Float64 => {
                invoke_ilut_factor_typed::<Float64Type>(&args, self.name(), drop_tolerance)
            }
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Build an ILUT sparse factorization for each CSR matrix in the batch.",
                "sparse_ilut_factor(sparse_batch, drop_tolerance => 1e-8, max_fill => 16)",
            )
            .with_argument("sparse_batch", "Canonical ndarrow.csr_matrix_batch column.")
            .with_argument("drop_tolerance", "Drop threshold for retaining off-diagonal entries.")
            .with_argument("max_fill", "Maximum retained off-diagonal entries per row.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseIlukFactor {
    signature: Signature,
}

impl SparseIlukFactor {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["matrix", "level_of_fill"]) }
    }
}

impl ScalarUDFImpl for SparseIlukFactor {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_iluk_factor" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[(2, ScalarCoercion::Integer)])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        iluk_factor_struct_field(self.name(), &value_type, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => invoke_iluk_factor_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_iluk_factor_typed::<Float64Type>(&args, self.name()),
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Build an ILU(k) sparse factorization for each CSR matrix in the batch.",
                "sparse_iluk_factor(sparse_batch, level_of_fill => 1)",
            )
            .with_argument("sparse_batch", "Canonical ndarrow.csr_matrix_batch column.")
            .with_argument("level_of_fill", "Requested level of fill during ILU(k) construction.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseApplyIlutPreconditioner {
    signature: Signature,
}

impl SparseApplyIlutPreconditioner {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseApplyIlutPreconditioner {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_apply_ilut_preconditioner" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let factorization = parse_sparse_factor_pair_field(&args.arg_fields[0], self.name(), 1)?;
        sparse_matrix_vector_output_field(&args, self.name(), &factorization.value_type, 2)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let factorization = parse_sparse_factor_pair_field(&args.arg_fields[0], self.name(), 1)?;
        match factorization.value_type {
            DataType::Float32 => invoke_apply_sparse_factor_pair_typed::<Float32Type>(
                &args,
                self.name(),
                |(l, u), rhs| {
                    nabled::linalg::sparse::apply_ilut_preconditioner(
                        &nabled::linalg::sparse::ILUTFactorization { l: l.clone(), u: u.clone() },
                        rhs,
                    )
                },
            ),
            DataType::Float64 => invoke_apply_sparse_factor_pair_typed::<Float64Type>(
                &args,
                self.name(),
                |(l, u), rhs| {
                    nabled::linalg::sparse::apply_ilut_preconditioner(
                        &nabled::linalg::sparse::ILUTFactorization { l: l.clone(), u: u.clone() },
                        rhs,
                    )
                },
            ),
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Apply an ILUT preconditioner to each dense vector in the batch.",
                "sparse_apply_ilut_preconditioner(factorization, rhs_batch)",
            )
            .with_argument("factorization", "Struct returned by sparse_ilut_factor.")
            .with_argument(
                "rhs_batch",
                "Canonical rank-1 variable-shape tensor batch of dense vectors.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseApplyIlukPreconditioner {
    signature: Signature,
}

impl SparseApplyIlukPreconditioner {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseApplyIlukPreconditioner {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_apply_iluk_preconditioner" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let factorization = parse_iluk_factor_field(&args.arg_fields[0], self.name(), 1)?;
        sparse_matrix_vector_output_field(&args, self.name(), &factorization.value_type, 2)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let factorization = parse_iluk_factor_field(&args.arg_fields[0], self.name(), 1)?;
        match factorization.value_type {
            DataType::Float32 => {
                invoke_apply_iluk_preconditioner_typed::<Float32Type>(&args, self.name())
            }
            DataType::Float64 => {
                invoke_apply_iluk_preconditioner_typed::<Float64Type>(&args, self.name())
            }
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Apply an ILU(k) preconditioner to each dense vector in the batch.",
                "sparse_apply_iluk_preconditioner(factorization, rhs_batch)",
            )
            .with_argument("factorization", "Struct returned by sparse_iluk_factor.")
            .with_argument(
                "rhs_batch",
                "Canonical rank-1 variable-shape tensor batch of dense vectors.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn sparse_lu_factor_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseLuFactor::new()))
}

#[must_use]
pub fn sparse_lu_solve_with_factorization_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseLuSolveWithFactorization::new()))
}

#[must_use]
pub fn sparse_lu_solve_multiple_with_factorization_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseLuSolveMultipleWithFactorization::new()))
}

#[must_use]
pub fn sparse_jacobi_preconditioner_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseJacobiPreconditioner::new()))
}

#[must_use]
pub fn sparse_apply_jacobi_preconditioner_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseApplyJacobiPreconditioner::new()))
}

#[must_use]
pub fn sparse_ilut_factor_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseIlutFactor::new()))
}

#[must_use]
pub fn sparse_iluk_factor_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseIlukFactor::new()))
}

#[must_use]
pub fn sparse_apply_ilut_preconditioner_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseApplyIlutPreconditioner::new()))
}

#[must_use]
pub fn sparse_apply_iluk_preconditioner_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseApplyIlukPreconditioner::new()))
}

#[cfg(test)]
mod tests {
    use datafusion::logical_expr::ReturnFieldArgs;
    use ndarray::Array1;

    use super::*;

    fn csr_matrix_f64(
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
    ) -> nabled::linalg::sparse::CsrMatrix<f64> {
        nabled::linalg::sparse::CsrMatrix { nrows, ncols, indptr, indices, data }
    }

    #[test]
    fn sparse_factorization_struct_fields_and_parsers_validate_contracts() {
        let pair = sparse_factor_pair_struct_field("factor", &DataType::Float64, false)
            .expect("factor field");
        let pair_contract =
            parse_sparse_factor_pair_field(&pair, "sparse_apply_ilut_preconditioner", 1)
                .expect("pair contract");
        assert_eq!(pair_contract, SparseFactorPairContract { value_type: DataType::Float64 });

        let lu = sparse_lu_factor_struct_field("factor", &DataType::Float32, false)
            .expect("lu factor field");
        let lu_contract =
            parse_sparse_lu_factor_field(&lu, "sparse_lu_solve_with_factorization", 1)
                .expect("lu contract");
        assert_eq!(lu_contract, SparseLuFactorContract { value_type: DataType::Float32 });

        let jacobi = jacobi_preconditioner_struct_field("jacobi", &DataType::Float64, false)
            .expect("jacobi field");
        let jacobi_contract =
            parse_jacobi_preconditioner_field(&jacobi, "sparse_apply_jacobi_preconditioner", 1)
                .expect("jacobi contract");
        assert_eq!(jacobi_contract, JacobiPreconditionerContract { value_type: DataType::Float64 });

        let iluk = iluk_factor_struct_field("iluk", &DataType::Float64, false).expect("iluk");
        let iluk_contract = parse_iluk_factor_field(&iluk, "sparse_apply_iluk_preconditioner", 1)
            .expect("iluk contract");
        assert_eq!(iluk_contract, IlukFactorContract { value_type: DataType::Float64 });
    }

    #[test]
    fn sparse_factorization_helpers_validate_input_contracts() {
        let int_list = int64_list_field("permutation", false);
        assert!(parse_int64_list_field(&int_list, "sparse_lu_factor", 1).is_ok());
        let wrong =
            Arc::new(Field::new("permutation", DataType::new_list(DataType::Float64, true), false));
        assert!(parse_int64_list_field(&wrong, "sparse_lu_factor", 1).is_err());

        let float32 = float32_scalar_from_f64("sparse_ilut_factor", 1.0e-6, "drop_tolerance")
            .expect("float32 conversion");
        assert!(float32 > 0.0);
        assert!(
            float32_scalar_from_f64("sparse_ilut_factor", f64::INFINITY, "drop_tolerance").is_err()
        );

        let wrong_pair = struct_field(
            "factor",
            vec![
                Field::new("x", DataType::Float64, false),
                Field::new("y", DataType::Float64, false),
            ],
            false,
        );
        assert!(
            parse_sparse_factor_pair_field(&wrong_pair, "sparse_apply_ilut_preconditioner", 1)
                .is_err()
        );

        let wrong_lu = struct_field(
            "factor",
            vec![
                csr_matrix_batch_field("l", &DataType::Float64, false).expect("l").as_ref().clone(),
                csr_matrix_batch_field("u", &DataType::Float64, false).expect("u").as_ref().clone(),
            ],
            false,
        );
        assert!(
            parse_sparse_lu_factor_field(&wrong_lu, "sparse_lu_solve_with_factorization", 1)
                .is_err()
        );

        let wrong_jacobi = struct_field(
            "jacobi",
            vec![
                variable_shape_tensor_field("inverse_diagonal", &DataType::Float64, 2, None, false)
                    .expect("inverse_diagonal")
                    .as_ref()
                    .clone(),
            ],
            false,
        );
        assert!(
            parse_jacobi_preconditioner_field(
                &wrong_jacobi,
                "sparse_apply_jacobi_preconditioner",
                1,
            )
            .is_err()
        );

        let wrong_iluk = struct_field(
            "iluk",
            vec![
                csr_matrix_batch_field("l", &DataType::Float64, false).expect("l").as_ref().clone(),
                csr_matrix_batch_field("u", &DataType::Float64, false).expect("u").as_ref().clone(),
                Field::new("level_of_fill", DataType::Float64, false),
            ],
            false,
        );
        assert!(
            parse_iluk_factor_field(&wrong_iluk, "sparse_apply_iluk_preconditioner", 1).is_err()
        );
    }

    #[test]
    fn sparse_factorization_output_field_helpers_validate_rank_and_types() {
        let factor_field = sparse_factor_pair_struct_field("factor", &DataType::Float64, false)
            .expect("factor field");
        let rhs_vectors =
            variable_shape_tensor_field("rhs", &DataType::Float64, 1, Some(&[None]), false)
                .expect("rhs vectors");
        let scalar_arguments: [Option<&datafusion::common::ScalarValue>; 0] = [];
        let vector_args = ReturnFieldArgs {
            arg_fields:       &[Arc::clone(&factor_field), Arc::clone(&rhs_vectors)],
            scalar_arguments: &scalar_arguments,
        };
        assert!(
            sparse_matrix_vector_output_field(
                &vector_args,
                "sparse_apply_ilut_preconditioner",
                &DataType::Float64,
                2,
            )
            .is_ok()
        );

        let rhs_matrices =
            variable_shape_tensor_field("rhs", &DataType::Float64, 2, Some(&[None, None]), false)
                .expect("rhs matrices");
        let matrix_args = ReturnFieldArgs {
            arg_fields:       &[Arc::clone(&factor_field), Arc::clone(&rhs_matrices)],
            scalar_arguments: &scalar_arguments,
        };
        assert!(
            sparse_matrix_matrix_output_field(
                &matrix_args,
                "sparse_lu_solve_multiple_with_factorization",
                &DataType::Float64,
                2,
            )
            .is_ok()
        );

        let rhs_matrices_float32 =
            variable_shape_tensor_field("rhs", &DataType::Float32, 2, Some(&[None, None]), false)
                .expect("rhs matrices");
        let wrong_type_args = ReturnFieldArgs {
            arg_fields:       &[
                sparse_factor_pair_struct_field("factor", &DataType::Float32, false)
                    .expect("factor field"),
                Arc::clone(&rhs_matrices_float32),
            ],
            scalar_arguments: &scalar_arguments,
        };
        assert!(
            sparse_matrix_matrix_output_field(
                &wrong_type_args,
                "sparse_lu_solve_multiple_with_factorization",
                &DataType::Float64,
                2,
            )
            .is_err()
        );

        let wrong_rank_args = ReturnFieldArgs {
            arg_fields:       &[Arc::clone(&factor_field), rhs_matrices_float32],
            scalar_arguments: &scalar_arguments,
        };
        assert!(
            sparse_matrix_vector_output_field(
                &wrong_rank_args,
                "sparse_apply_ilut_preconditioner",
                &DataType::Float32,
                2,
            )
            .is_err()
        );
    }

    #[test]
    fn sparse_factorization_roundtrip_helpers_cover_owned_and_output_paths() {
        let lower = csr_matrix_f64(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);
        let upper = csr_matrix_f64(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![4.0, 1.0, 3.0]);

        let pair_output =
            build_sparse_factor_pair_output("sparse_apply_ilut_preconditioner", "l", "u", vec![(
                lower.clone(),
                upper.clone(),
            )])
            .expect("pair output");
        let pair_field = sparse_factor_pair_struct_field("factor", &DataType::Float64, false)
            .expect("pair field");
        let pair_array = match pair_output {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        };
        let pair_array =
            pair_array.as_any().downcast_ref::<StructArray>().expect("pair struct array");
        let pair_roundtrip =
            owned_sparse_factor_pairs::<Float64Type>(&pair_field, pair_array, "roundtrip")
                .expect("pair roundtrip");
        assert_eq!(pair_roundtrip.len(), 1);
        assert_eq!(pair_roundtrip[0].0.nrows, 2);
        assert_eq!(pair_roundtrip[0].1.ncols, 2);

        let lu_output = build_sparse_lu_factorization_output("sparse_lu_factor", vec![
            nabled::linalg::sparse::SparseLUFactorization {
                l:           lower.clone(),
                u:           upper.clone(),
                permutation: vec![0, 1],
            },
        ])
        .expect("lu output");
        let lu_field =
            sparse_lu_factor_struct_field("factor", &DataType::Float64, false).expect("lu field");
        let lu_array = match lu_output {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        };
        let lu_array = lu_array.as_any().downcast_ref::<StructArray>().expect("lu struct array");
        let lu_roundtrip =
            owned_sparse_lu_factorizations::<Float64Type>(&lu_field, lu_array, "roundtrip")
                .expect("lu roundtrip");
        assert_eq!(lu_roundtrip.len(), 1);
        assert_eq!(lu_roundtrip[0].permutation, vec![0, 1]);

        let jacobi_output = build_jacobi_preconditioner_output(vec![
            nabled::linalg::sparse::JacobiPreconditioner {
                inverse_diagonal: Array1::from_vec(vec![0.25_f64, 1.0 / 3.0]),
            },
        ])
        .expect("jacobi output");
        let jacobi_field = jacobi_preconditioner_struct_field("jacobi", &DataType::Float64, false)
            .expect("jacobi field");
        let jacobi_array = match jacobi_output {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        };
        let jacobi_array =
            jacobi_array.as_any().downcast_ref::<StructArray>().expect("jacobi struct array");
        let jacobi_roundtrip =
            owned_jacobi_preconditioners::<Float64Type>(&jacobi_field, jacobi_array, "roundtrip")
                .expect("jacobi roundtrip");
        assert_eq!(jacobi_roundtrip[0].inverse_diagonal.len(), 2);

        let iluk_output = build_iluk_factor_output("sparse_iluk_factor", vec![
            nabled::linalg::sparse::ILUKFactorization {
                l:             lower,
                u:             upper,
                level_of_fill: 1,
            },
        ])
        .expect("iluk output");
        let iluk_field =
            iluk_factor_struct_field("iluk", &DataType::Float64, false).expect("iluk field");
        let iluk_array = match iluk_output {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        };
        let iluk_array =
            iluk_array.as_any().downcast_ref::<StructArray>().expect("iluk struct array");
        let iluk_roundtrip =
            owned_iluk_factorizations::<Float64Type>(&iluk_field, iluk_array, "roundtrip")
                .expect("iluk roundtrip");
        assert_eq!(iluk_roundtrip[0].level_of_fill, 1);
    }
}
