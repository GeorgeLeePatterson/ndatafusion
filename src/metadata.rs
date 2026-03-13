use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::extension::{ExtensionType, FixedShapeTensor, VariableShapeTensor};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use ndarrow::CsrMatrixBatchExtension;
use serde::{Deserialize, Serialize};

use crate::error::{plan_error, type_mismatch};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariableShapeTensorContract {
    pub(crate) dimensions:    usize,
    pub(crate) uniform_shape: Option<Vec<Option<i32>>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FixedShapeTensorWireMetadata {
    shape: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct VariableShapeTensorWireMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    uniform_shape: Option<Vec<Option<i32>>>,
}

pub(crate) fn float64_scalar_field(name: &str, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, DataType::Float64, nullable))
}

pub(crate) fn vector_field(
    name: &str,
    len: usize,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let value_length = i32::try_from(len)
        .map_err(|_| plan_error(name, format!("vector length {len} exceeds Arrow i32 limits")))?;
    Ok(Arc::new(Field::new(
        name,
        DataType::new_fixed_size_list(DataType::Float64, value_length, false),
        nullable,
    )))
}

pub(crate) fn struct_field(name: &str, fields: Vec<Field>, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, DataType::Struct(fields.into()), nullable))
}

pub(crate) fn field_like(name: &str, template: &FieldRef, nullable: bool) -> FieldRef {
    Arc::new(
        Field::new(name, template.data_type().clone(), nullable)
            .with_metadata(template.metadata().clone()),
    )
}

pub(crate) fn fixed_shape_tensor_field(
    name: &str,
    tensor_shape: &[usize],
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let list_size = tensor_shape.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            plan_error(name, format!("tensor shape product overflow for {tensor_shape:?}"))
        })
    })?;
    let list_size = i32::try_from(list_size).map_err(|_| {
        plan_error(
            name,
            format!("tensor element count exceeds Arrow i32 limits for {tensor_shape:?}"),
        )
    })?;

    let extension = FixedShapeTensor::try_new(DataType::Float64, tensor_shape.to_vec(), None, None)
        .map_err(|error| plan_error(name, error))?;
    let data_type = DataType::new_fixed_size_list(DataType::Float64, list_size, false);
    extension.supports_data_type(&data_type).map_err(|error| plan_error(name, error))?;
    let metadata =
        serde_json::to_string(&FixedShapeTensorWireMetadata { shape: tensor_shape.to_vec() })
            .map_err(|error| plan_error(name, error))?;
    let mut field_metadata = HashMap::new();
    drop(
        field_metadata.insert("ARROW:extension:name".to_owned(), FixedShapeTensor::NAME.to_owned()),
    );
    drop(field_metadata.insert("ARROW:extension:metadata".to_owned(), metadata));
    let mut field = Field::new(name, data_type, nullable);
    field = field.with_metadata(field_metadata);
    Ok(Arc::new(field))
}

pub(crate) fn variable_shape_tensor_field(
    name: &str,
    dimensions: usize,
    uniform_shape: Option<&[Option<i32>]>,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let dimensions_i32 = i32::try_from(dimensions).map_err(|_| {
        plan_error(name, format!("tensor rank {dimensions} exceeds Arrow i32 limits"))
    })?;
    let extension = VariableShapeTensor::try_new(
        DataType::Float64,
        dimensions,
        None,
        None,
        uniform_shape.map(ToOwned::to_owned),
    )
    .map_err(|error| plan_error(name, error))?;
    let data_type = DataType::Struct(
        vec![
            Field::new("data", DataType::new_list(DataType::Float64, false), false),
            Field::new(
                "shape",
                DataType::new_fixed_size_list(DataType::Int32, dimensions_i32, false),
                false,
            ),
        ]
        .into(),
    );
    extension.supports_data_type(&data_type).map_err(|error| plan_error(name, error))?;
    let metadata = serde_json::to_string(&VariableShapeTensorWireMetadata {
        uniform_shape: uniform_shape.map(ToOwned::to_owned),
    })
    .map_err(|error| plan_error(name, error))?;
    let mut field_metadata = HashMap::new();
    drop(
        field_metadata
            .insert("ARROW:extension:name".to_owned(), VariableShapeTensor::NAME.to_owned()),
    );
    drop(field_metadata.insert("ARROW:extension:metadata".to_owned(), metadata));
    let mut field = Field::new(name, data_type, nullable);
    field = field.with_metadata(field_metadata);
    Ok(Arc::new(field))
}

pub(crate) fn parse_float64_vector_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<usize> {
    match field.data_type() {
        DataType::FixedSizeList(item, len) if item.data_type() == &DataType::Float64 => {
            usize::try_from(*len).map_err(|_| {
                plan_error(
                    function_name,
                    format!("argument {position} has negative vector width {len}"),
                )
            })
        }
        actual => Err(type_mismatch(function_name, position, "FixedSizeList<Float64>(D)", actual)),
    }
}

pub(crate) fn parse_float64_tensor_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<Vec<usize>> {
    let extension = field
        .try_extension_type::<FixedShapeTensor>()
        .map_err(|error| plan_error(function_name, error))?;
    if extension.value_type() != &DataType::Float64 {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.fixed_shape_tensor<Float64>",
            field.data_type(),
        ));
    }

    let raw_metadata = field.extension_type_metadata().ok_or_else(|| {
        plan_error(function_name, format!("argument {position} is missing tensor metadata"))
    })?;
    let metadata: FixedShapeTensorWireMetadata =
        serde_json::from_str(raw_metadata).map_err(|error| plan_error(function_name, error))?;
    Ok(metadata.shape)
}

pub(crate) fn parse_float64_matrix_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<[usize; 2]> {
    let shape = parse_float64_tensor_batch_field(field, function_name, position)?;
    if shape.len() != 2 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} must be a batch of rank-2 matrices, found shape {shape:?}"
            ),
        ));
    }
    Ok([shape[0], shape[1]])
}

pub(crate) fn parse_float64_variable_shape_tensor_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<VariableShapeTensorContract> {
    let extension = field
        .try_extension_type::<VariableShapeTensor>()
        .map_err(|error| plan_error(function_name, error))?;
    if extension.value_type() != &DataType::Float64 {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.variable_shape_tensor<Float64>",
            field.data_type(),
        ));
    }
    Ok(VariableShapeTensorContract {
        dimensions:    extension.dimensions(),
        uniform_shape: extension.uniform_shapes().map(ToOwned::to_owned),
    })
}

pub(crate) fn require_float64_csr_matrix_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<()> {
    let extension = field
        .try_extension_type::<CsrMatrixBatchExtension>()
        .map_err(|error| plan_error(function_name, error))?;
    if extension.value_type() != &DataType::Float64 {
        return Err(type_mismatch(
            function_name,
            position,
            "ndarrow.csr_matrix_batch<Float64>",
            field.data_type(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use datafusion::arrow::datatypes::{DataType, Field};
    use ndarray::{Array1, Array2};

    use super::{
        field_like, fixed_shape_tensor_field, float64_scalar_field,
        parse_float64_matrix_batch_field, parse_float64_tensor_batch_field,
        parse_float64_variable_shape_tensor_field, parse_float64_vector_field,
        require_float64_csr_matrix_batch_field, struct_field, variable_shape_tensor_field,
        vector_field,
    };

    #[test]
    fn field_builders_create_expected_shapes() {
        let scalar = float64_scalar_field("score", true);
        let vector = vector_field("vector", 3, false).expect("vector field");
        let structure = struct_field("pair", vec![Field::new("x", DataType::Float64, false)], true);
        let tensor = fixed_shape_tensor_field("tensor", &[2, 3], false).expect("tensor field");
        let variable =
            variable_shape_tensor_field("ragged", 2, Some(&[Some(2), None]), true).expect("field");

        assert_eq!(scalar.data_type(), &DataType::Float64);
        assert!(scalar.is_nullable());
        assert_eq!(vector.data_type(), &DataType::new_fixed_size_list(DataType::Float64, 3, false));
        assert!(matches!(structure.data_type(), DataType::Struct(_)));
        assert_eq!(
            tensor.extension_type_name().expect("tensor extension name"),
            "arrow.fixed_shape_tensor"
        );
        assert_eq!(
            variable.extension_type_name().expect("variable extension name"),
            "arrow.variable_shape_tensor"
        );
    }

    #[test]
    fn field_like_preserves_type_and_metadata_with_new_name() {
        let tensor = fixed_shape_tensor_field("tensor", &[2, 3], false).expect("tensor field");
        let renamed = field_like("renamed", &tensor, true);

        assert_eq!(renamed.name(), "renamed");
        assert!(renamed.is_nullable());
        assert_eq!(renamed.data_type(), tensor.data_type());
        assert_eq!(renamed.metadata(), tensor.metadata());
    }

    #[test]
    fn field_builders_reject_overflowing_contracts() {
        let vector_error = vector_field("vector", usize::MAX, false).expect_err("vector overflow");
        let tensor_overflow = fixed_shape_tensor_field("tensor", &[usize::MAX, 2], false)
            .expect_err("shape overflow");
        let tensor_limit = fixed_shape_tensor_field("tensor", &[65_536, 65_536], false)
            .expect_err("tensor i32 limit");
        let variable =
            variable_shape_tensor_field("ragged", usize::MAX, None, false).expect_err("rank limit");

        assert!(vector_error.to_string().contains("exceeds Arrow i32 limits"));
        assert!(tensor_overflow.to_string().contains("tensor shape product overflow"));
        assert!(tensor_limit.to_string().contains("tensor element count exceeds Arrow i32 limits"));
        assert!(variable.to_string().contains("tensor rank"));
    }

    #[test]
    fn parse_helpers_accept_expected_contracts() {
        let vector = vector_field("vector", 3, false).expect("vector field");
        let tensor = fixed_shape_tensor_field("tensor", &[2, 3], false).expect("tensor field");
        let ragged =
            variable_shape_tensor_field("ragged", 1, Some(&[None]), true).expect("variable field");
        let (csr_field, _csr_array) = ndarrow::csr_batch_to_extension_array(
            "sparse",
            vec![[2, 3], [1, 2]],
            vec![vec![0, 1, 2], vec![0, 1]],
            vec![vec![0, 2], vec![1]],
            vec![vec![1.0, 2.0], vec![3.0]],
        )
        .expect("csr field");

        assert_eq!(parse_float64_vector_field(&vector, "vector_dot", 1).expect("vector len"), 3);
        assert_eq!(
            parse_float64_tensor_batch_field(&tensor, "matrix_matmul", 1).expect("tensor shape"),
            vec![2, 3]
        );
        assert_eq!(
            parse_float64_matrix_batch_field(&tensor, "matrix_matmul", 1).expect("matrix shape"),
            [2, 3]
        );
        let ragged = parse_float64_variable_shape_tensor_field(&ragged, "sparse_matvec", 2)
            .expect("ragged contract");
        assert_eq!(ragged.dimensions, 1);
        assert_eq!(ragged.uniform_shape, Some(vec![None]));
        require_float64_csr_matrix_batch_field(&Arc::new(csr_field), "sparse_matvec", 1)
            .expect("csr batch field");
    }

    #[test]
    fn parse_helpers_reject_mismatches_and_missing_metadata() {
        let scalar = float64_scalar_field("scalar", false);
        let vector = vector_field("vector", 4, false).expect("vector field");
        let rank_three =
            fixed_shape_tensor_field("tensor", &[2, 3, 4], false).expect("rank-3 tensor");
        let mut tensor_metadata = HashMap::new();
        drop(
            tensor_metadata
                .insert("ARROW:extension:name".to_owned(), "arrow.fixed_shape_tensor".to_owned()),
        );
        let missing_tensor_metadata = Arc::new(
            Field::new("tensor", DataType::new_fixed_size_list(DataType::Float64, 4, false), false)
                .with_metadata(tensor_metadata),
        );
        let mut variable_metadata = HashMap::new();
        drop(
            variable_metadata.insert(
                "ARROW:extension:name".to_owned(),
                "arrow.variable_shape_tensor".to_owned(),
            ),
        );
        let missing_variable_metadata = Arc::new(
            Field::new(
                "ragged",
                DataType::Struct(
                    vec![
                        Field::new("data", DataType::new_list(DataType::Float64, false), false),
                        Field::new(
                            "shape",
                            DataType::new_fixed_size_list(DataType::Int32, 1, false),
                            false,
                        ),
                    ]
                    .into(),
                ),
                false,
            )
            .with_metadata(variable_metadata),
        );

        let scalar_error =
            parse_float64_vector_field(&scalar, "vector_dot", 1).expect_err("vector type mismatch");
        let tensor_error = parse_float64_tensor_batch_field(&vector, "matrix_matmul", 1)
            .expect_err("tensor type mismatch");
        let rank_error = parse_float64_matrix_batch_field(&rank_three, "matrix_matmul", 1)
            .expect_err("matrix rank mismatch");
        let missing_fixed_error =
            parse_float64_tensor_batch_field(&missing_tensor_metadata, "matrix_matmul", 1)
                .expect_err("missing tensor metadata");
        let missing_variable_error = parse_float64_variable_shape_tensor_field(
            &missing_variable_metadata,
            "sparse_matvec",
            2,
        )
        .expect_err("missing variable metadata");
        let csr_error = require_float64_csr_matrix_batch_field(&scalar, "sparse_matvec", 1)
            .expect_err("csr mismatch");

        assert!(scalar_error.to_string().contains("expected FixedSizeList<Float64>(D)"));
        assert!(tensor_error.to_string().contains("matrix_matmul"));
        assert!(rank_error.to_string().contains("batch of rank-2 matrices"));
        assert!(missing_fixed_error.to_string().contains("matrix_matmul"));
        assert!(missing_variable_error.to_string().contains("sparse_matvec"));
        assert!(csr_error.to_string().contains("sparse_matvec"));
    }

    #[test]
    fn parse_helpers_reject_non_float64_extension_value_types() {
        let (int_tensor_field, _array) = ndarrow::arrayd_to_fixed_shape_tensor(
            "tensor",
            Array2::<f32>::zeros((2, 2)).into_dyn(),
        )
        .expect("f32 tensor");
        let (int_ragged_field, _array) = ndarrow::arrays_to_variable_shape_tensor(
            "ragged",
            vec![Array1::<f32>::zeros(2).into_dyn()],
            Some(vec![None]),
        )
        .expect("f32 ragged tensor");

        let tensor_error =
            parse_float64_tensor_batch_field(&Arc::new(int_tensor_field), "matrix_matmul", 1)
                .expect_err("tensor value type mismatch");
        let ragged_error = parse_float64_variable_shape_tensor_field(
            &Arc::new(int_ragged_field),
            "sparse_matvec",
            2,
        )
        .expect_err("ragged value type mismatch");

        assert!(tensor_error.to_string().contains("expected arrow.fixed_shape_tensor<Float64>"));
        assert!(ragged_error.to_string().contains("expected arrow.variable_shape_tensor<Float64>"));
    }
}
