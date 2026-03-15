use std::collections::HashMap;
use std::sync::Arc;

use arrow_schema::extension::{ExtensionType, FixedShapeTensor, VariableShapeTensor};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use ndarrow::{Complex64Extension, CsrMatrixBatchExtension};
use serde::{Deserialize, Serialize};

use crate::error::{plan_error, type_mismatch};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorContract {
    pub(crate) value_type: DataType,
    pub(crate) len:        usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorBatchContract {
    pub(crate) value_type: DataType,
    pub(crate) shape:      Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MatrixBatchContract {
    pub(crate) value_type: DataType,
    pub(crate) rows:       usize,
    pub(crate) cols:       usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VariableShapeTensorContract {
    pub(crate) value_type:    DataType,
    pub(crate) dimensions:    usize,
    pub(crate) uniform_shape: Option<Vec<Option<i32>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComplexVectorContract {
    pub(crate) len: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComplexTensorBatchContract {
    pub(crate) shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComplexMatrixBatchContract {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComplexVariableShapeTensorContract {
    pub(crate) dimensions:    usize,
    pub(crate) uniform_shape: Option<Vec<Option<i32>>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct FixedShapeTensorWireMetadata {
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
struct VariableShapeTensorWireMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    uniform_shape: Option<Vec<Option<i32>>>,
}

fn supported_numeric_type(
    value_type: &DataType,
    function_name: &str,
    position: usize,
    expected: &str,
) -> datafusion::common::Result<DataType> {
    match value_type {
        DataType::Float32 | DataType::Float64 => Ok(value_type.clone()),
        actual => Err(type_mismatch(function_name, position, expected, actual)),
    }
}

pub(crate) fn scalar_field(name: &str, value_type: &DataType, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, value_type.clone(), nullable))
}

pub(crate) fn complex_scalar_field(
    name: &str,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let mut field =
        Field::new(name, DataType::new_fixed_size_list(DataType::Float64, 2, false), nullable);
    field
        .try_with_extension_type(Complex64Extension)
        .map_err(|error| plan_error(name, error))?;
    Ok(Arc::new(field))
}

pub(crate) fn vector_field(
    name: &str,
    value_type: &DataType,
    len: usize,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let value_length = i32::try_from(len)
        .map_err(|_| plan_error(name, format!("vector length {len} exceeds Arrow i32 limits")))?;
    Ok(Arc::new(Field::new(
        name,
        DataType::new_fixed_size_list(value_type.clone(), value_length, false),
        nullable,
    )))
}

pub(crate) fn complex_vector_field(
    name: &str,
    len: usize,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let value_length = i32::try_from(len)
        .map_err(|_| plan_error(name, format!("vector length {len} exceeds Arrow i32 limits")))?;
    Ok(Arc::new(Field::new(
        name,
        DataType::FixedSizeList(complex_scalar_field("item", false)?, value_length),
        nullable,
    )))
}

pub(crate) fn complex_fixed_shape_tensor_field(
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

    let item = complex_scalar_field("item", false)?;
    let extension =
        FixedShapeTensor::try_new(item.data_type().clone(), tensor_shape.to_vec(), None, None)
            .map_err(|error| plan_error(name, error))?;
    let data_type = DataType::FixedSizeList(field_like("item", &item, false), list_size);
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
    value_type: &DataType,
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

    let extension =
        FixedShapeTensor::try_new(value_type.clone(), tensor_shape.to_vec(), None, None)
            .map_err(|error| plan_error(name, error))?;
    let data_type = DataType::new_fixed_size_list(value_type.clone(), list_size, false);
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
    value_type: &DataType,
    dimensions: usize,
    uniform_shape: Option<&[Option<i32>]>,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let dimensions_i32 = i32::try_from(dimensions).map_err(|_| {
        plan_error(name, format!("tensor rank {dimensions} exceeds Arrow i32 limits"))
    })?;
    let extension = VariableShapeTensor::try_new(
        value_type.clone(),
        dimensions,
        None,
        None,
        uniform_shape.map(ToOwned::to_owned),
    )
    .map_err(|error| plan_error(name, error))?;
    let data_type = DataType::Struct(
        vec![
            Field::new("data", DataType::new_list(value_type.clone(), false), false),
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

pub(crate) fn complex_variable_shape_tensor_field(
    name: &str,
    dimensions: usize,
    uniform_shape: Option<&[Option<i32>]>,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let dimensions_i32 = i32::try_from(dimensions).map_err(|_| {
        plan_error(name, format!("tensor rank {dimensions} exceeds Arrow i32 limits"))
    })?;
    let item = complex_scalar_field("item", false)?;
    let extension = VariableShapeTensor::try_new(
        item.data_type().clone(),
        dimensions,
        None,
        None,
        uniform_shape.map(ToOwned::to_owned),
    )
    .map_err(|error| plan_error(name, error))?;
    let data_type = DataType::Struct(
        vec![
            Field::new("data", DataType::List(field_like("item", &item, false)), false),
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

pub(crate) fn csr_matrix_batch_field(
    name: &str,
    value_type: &DataType,
    nullable: bool,
) -> datafusion::common::Result<FieldRef> {
    let data_type = DataType::Struct(
        vec![
            Field::new("shape", DataType::new_fixed_size_list(DataType::Int32, 2, false), false),
            Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false),
            Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false),
            Field::new("values", DataType::new_list(value_type.clone(), false), false),
        ]
        .into(),
    );
    let extension = CsrMatrixBatchExtension::try_new(&data_type, ())
        .map_err(|error| plan_error(name, error))?;
    let mut field = Field::new(name, data_type, nullable);
    field.try_with_extension_type(extension).map_err(|error| plan_error(name, error))?;
    Ok(Arc::new(field))
}

pub(crate) fn parse_vector_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<VectorContract> {
    match field.data_type() {
        DataType::FixedSizeList(item, len) => {
            let value_type = supported_numeric_type(
                item.data_type(),
                function_name,
                position,
                "FixedSizeList<Float32|Float64>(D)",
            )?;
            let len = usize::try_from(*len).map_err(|_| {
                plan_error(
                    function_name,
                    format!("argument {position} has negative vector width {len}"),
                )
            })?;
            Ok(VectorContract { value_type, len })
        }
        actual => {
            Err(type_mismatch(function_name, position, "FixedSizeList<Float32|Float64>(D)", actual))
        }
    }
}

pub(crate) fn parse_complex_vector_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<(FieldRef, ComplexVectorContract)> {
    match field.data_type() {
        DataType::FixedSizeList(item, len) => {
            let _extension = item.try_extension_type::<Complex64Extension>().map_err(|_| {
                type_mismatch(
                    function_name,
                    position,
                    "FixedSizeList<ndarrow.complex64>(D)",
                    field.data_type(),
                )
            })?;
            let len = usize::try_from(*len).map_err(|_| {
                plan_error(
                    function_name,
                    format!("argument {position} has negative vector width {len}"),
                )
            })?;
            Ok((Arc::clone(item), ComplexVectorContract { len }))
        }
        actual => Err(type_mismatch(
            function_name,
            position,
            "FixedSizeList<ndarrow.complex64>(D)",
            actual,
        )),
    }
}

pub(crate) fn parse_complex_tensor_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<ComplexTensorBatchContract> {
    let _extension = field
        .try_extension_type::<FixedShapeTensor>()
        .map_err(|error| plan_error(function_name, error))?;
    let DataType::FixedSizeList(item, _len) = field.data_type() else {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.fixed_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        ));
    };
    let _complex = item.try_extension_type::<Complex64Extension>().map_err(|_| {
        type_mismatch(
            function_name,
            position,
            "arrow.fixed_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        )
    })?;

    let raw_metadata = field.extension_type_metadata().ok_or_else(|| {
        plan_error(function_name, format!("argument {position} is missing tensor metadata"))
    })?;
    let metadata: FixedShapeTensorWireMetadata =
        serde_json::from_str(raw_metadata).map_err(|error| plan_error(function_name, error))?;
    Ok(ComplexTensorBatchContract { shape: metadata.shape })
}

pub(crate) fn parse_complex_matrix_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<ComplexMatrixBatchContract> {
    let contract = parse_complex_tensor_batch_field(field, function_name, position)?;
    if contract.shape.len() != 2 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} must be a batch of rank-2 complex matrices, found shape {:?}",
                contract.shape
            ),
        ));
    }
    Ok(ComplexMatrixBatchContract { rows: contract.shape[0], cols: contract.shape[1] })
}

pub(crate) fn parse_complex_variable_shape_tensor_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<ComplexVariableShapeTensorContract> {
    if field.extension_type_name() != Some(VariableShapeTensor::NAME) {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.variable_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        ));
    }
    let DataType::Struct(fields) = field.data_type() else {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.variable_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        ));
    };
    let Some(data_field) = fields.first() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} variable tensor is missing its data field"),
        ));
    };
    let DataType::List(item) = data_field.data_type() else {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.variable_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        ));
    };
    let _complex = item.try_extension_type::<Complex64Extension>().map_err(|_| {
        type_mismatch(
            function_name,
            position,
            "arrow.variable_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        )
    })?;
    let Some(shape_field) = fields.get(1) else {
        return Err(plan_error(
            function_name,
            format!("argument {position} variable tensor is missing its shape field"),
        ));
    };
    let DataType::FixedSizeList(_, dimensions_i32) = shape_field.data_type() else {
        return Err(type_mismatch(
            function_name,
            position,
            "arrow.variable_shape_tensor<ndarrow.complex64>",
            field.data_type(),
        ));
    };
    let dimensions = usize::try_from(*dimensions_i32).map_err(|_| {
        plan_error(
            function_name,
            format!("argument {position} variable tensor has negative rank {dimensions_i32}"),
        )
    })?;
    let uniform_shape = match field.extension_type_metadata() {
        Some(raw_metadata) => {
            let metadata: VariableShapeTensorWireMetadata = serde_json::from_str(raw_metadata)
                .map_err(|error| plan_error(function_name, error))?;
            metadata.uniform_shape
        }
        None => None,
    };
    Ok(ComplexVariableShapeTensorContract { dimensions, uniform_shape })
}

pub(crate) fn parse_tensor_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<TensorBatchContract> {
    let extension = field
        .try_extension_type::<FixedShapeTensor>()
        .map_err(|error| plan_error(function_name, error))?;
    let value_type = supported_numeric_type(
        extension.value_type(),
        function_name,
        position,
        "arrow.fixed_shape_tensor<Float32|Float64>",
    )?;

    let raw_metadata = field.extension_type_metadata().ok_or_else(|| {
        plan_error(function_name, format!("argument {position} is missing tensor metadata"))
    })?;
    let metadata: FixedShapeTensorWireMetadata =
        serde_json::from_str(raw_metadata).map_err(|error| plan_error(function_name, error))?;
    Ok(TensorBatchContract { value_type, shape: metadata.shape })
}

pub(crate) fn parse_matrix_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<MatrixBatchContract> {
    let contract = parse_tensor_batch_field(field, function_name, position)?;
    if contract.shape.len() != 2 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} must be a batch of rank-2 matrices, found shape {:?}",
                contract.shape
            ),
        ));
    }
    Ok(MatrixBatchContract {
        value_type: contract.value_type,
        rows:       contract.shape[0],
        cols:       contract.shape[1],
    })
}

pub(crate) fn parse_variable_shape_tensor_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<VariableShapeTensorContract> {
    let extension = field
        .try_extension_type::<VariableShapeTensor>()
        .map_err(|error| plan_error(function_name, error))?;
    let value_type = supported_numeric_type(
        extension.value_type(),
        function_name,
        position,
        "arrow.variable_shape_tensor<Float32|Float64>",
    )?;
    Ok(VariableShapeTensorContract {
        value_type,
        dimensions: extension.dimensions(),
        uniform_shape: extension.uniform_shapes().map(ToOwned::to_owned),
    })
}

pub(crate) fn parse_csr_matrix_batch_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> datafusion::common::Result<DataType> {
    let extension = field
        .try_extension_type::<CsrMatrixBatchExtension>()
        .map_err(|error| plan_error(function_name, error))?;
    supported_numeric_type(
        extension.value_type(),
        function_name,
        position,
        "ndarrow.csr_matrix_batch<Float32|Float64>",
    )
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_schema::extension::ExtensionType;
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::{DataType, Field};
    use ndarray::{Array1, Array2, Array3};
    use ndarrow::{Complex64Extension, CsrMatrixBatchExtension};
    use num_complex::Complex64;

    use super::{
        ComplexVectorContract, FixedShapeTensorWireMetadata, VariableShapeTensorWireMetadata,
        complex_scalar_field, complex_vector_field, csr_matrix_batch_field, field_like,
        fixed_shape_tensor_field, parse_complex_vector_field, parse_csr_matrix_batch_field,
        parse_matrix_batch_field, parse_tensor_batch_field, parse_variable_shape_tensor_field,
        parse_vector_field, scalar_field, struct_field, variable_shape_tensor_field, vector_field,
    };

    #[test]
    fn field_builders_create_expected_shapes() {
        let scalar = scalar_field("score", &DataType::Float64, true);
        let complex_scalar = complex_scalar_field("complex", false).expect("complex scalar field");
        let vector = vector_field("vector", &DataType::Float32, 3, false).expect("vector field");
        let complex_vector =
            complex_vector_field("complex_vector", 2, true).expect("complex vector field");
        let structure = struct_field("pair", vec![Field::new("x", DataType::Float64, false)], true);
        let tensor = fixed_shape_tensor_field("tensor", &DataType::Float32, &[2, 3], false)
            .expect("tensor field");
        let variable = variable_shape_tensor_field(
            "ragged",
            &DataType::Float32,
            2,
            Some(&[Some(2), None]),
            true,
        )
        .expect("field");
        let sparse =
            csr_matrix_batch_field("sparse", &DataType::Float32, false).expect("sparse field");

        assert_eq!(scalar.data_type(), &DataType::Float64);
        assert!(scalar.is_nullable());
        assert_eq!(
            complex_scalar.extension_type_name().expect("complex extension"),
            Complex64Extension::NAME
        );
        assert_eq!(vector.data_type(), &DataType::new_fixed_size_list(DataType::Float32, 3, false));
        let DataType::FixedSizeList(item, len) = complex_vector.data_type() else {
            panic!("expected complex vector storage");
        };
        assert_eq!(*len, 2);
        assert_eq!(
            item.extension_type_name().expect("complex item extension"),
            Complex64Extension::NAME
        );
        assert!(complex_vector.is_nullable());
        assert!(matches!(structure.data_type(), DataType::Struct(_)));
        assert_eq!(
            tensor.extension_type_name().expect("tensor extension name"),
            "arrow.fixed_shape_tensor"
        );
        assert_eq!(
            variable.extension_type_name().expect("variable extension name"),
            "arrow.variable_shape_tensor"
        );
        assert_eq!(
            sparse.extension_type_name().expect("sparse extension name"),
            CsrMatrixBatchExtension::NAME
        );
    }

    #[test]
    fn field_like_preserves_type_and_metadata_with_new_name() {
        let tensor = fixed_shape_tensor_field("tensor", &DataType::Float64, &[2, 3], false)
            .expect("tensor field");
        let renamed = field_like("renamed", &tensor, true);

        assert_eq!(renamed.name(), "renamed");
        assert!(renamed.is_nullable());
        assert_eq!(renamed.data_type(), tensor.data_type());
        assert_eq!(renamed.metadata(), tensor.metadata());
    }

    #[test]
    fn field_builders_reject_overflowing_contracts() {
        let vector_error = vector_field("vector", &DataType::Float64, usize::MAX, false)
            .expect_err("vector overflow");
        let tensor_overflow =
            fixed_shape_tensor_field("tensor", &DataType::Float64, &[usize::MAX, 2], false)
                .expect_err("shape overflow");
        let tensor_limit =
            fixed_shape_tensor_field("tensor", &DataType::Float64, &[65_536, 65_536], false)
                .expect_err("tensor i32 limit");
        let variable =
            variable_shape_tensor_field("ragged", &DataType::Float64, usize::MAX, None, false)
                .expect_err("rank limit");

        assert!(vector_error.to_string().contains("exceeds Arrow i32 limits"));
        assert!(tensor_overflow.to_string().contains("tensor shape product overflow"));
        assert!(tensor_limit.to_string().contains("tensor element count exceeds Arrow i32 limits"));
        assert!(variable.to_string().contains("tensor rank"));
    }

    #[test]
    fn parse_helpers_accept_expected_contracts() {
        let vector = vector_field("vector", &DataType::Float64, 3, false).expect("vector field");
        let complex_vectors = {
            let array = Array2::from_shape_vec((2, 3), vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, -2.0),
            ])
            .expect("complex matrix");
            Arc::new(Field::new(
                "complex_vectors",
                ndarrow::array2_complex64_to_fixed_size_list(array)
                    .expect("complex outbound")
                    .data_type()
                    .clone(),
                false,
            ))
        };
        let tensor = fixed_shape_tensor_field("tensor", &DataType::Float64, &[2, 3], false)
            .expect("tensor field");
        let ragged =
            variable_shape_tensor_field("ragged", &DataType::Float64, 1, Some(&[None]), true)
                .expect("variable field");
        let (csr_field, _csr_array) = ndarrow::csr_batch_to_extension_array(
            "sparse",
            vec![[2, 3], [1, 2]],
            vec![vec![0, 1, 2], vec![0, 1]],
            vec![vec![0, 2], vec![1]],
            vec![vec![1.0, 2.0], vec![3.0]],
        )
        .expect("csr field");

        let vector_contract = parse_vector_field(&vector, "vector_dot", 1).expect("vector len");
        assert_eq!(vector_contract.value_type, DataType::Float64);
        assert_eq!(vector_contract.len, 3);
        let (complex_item_field, complex_contract) =
            parse_complex_vector_field(&complex_vectors, "vector_dot_hermitian", 1)
                .expect("complex vector len");
        assert_eq!(complex_item_field.extension_type_name(), Some(Complex64Extension::NAME));
        assert_eq!(complex_contract, ComplexVectorContract { len: 3 });
        let tensor_contract =
            parse_tensor_batch_field(&tensor, "matrix_matmul", 1).expect("tensor shape");
        assert_eq!(tensor_contract.value_type, DataType::Float64);
        assert_eq!(tensor_contract.shape, vec![2, 3]);
        let matrix_contract =
            parse_matrix_batch_field(&tensor, "matrix_matmul", 1).expect("matrix shape");
        assert_eq!(matrix_contract.value_type, DataType::Float64);
        assert_eq!([matrix_contract.rows, matrix_contract.cols], [2, 3]);
        let ragged = parse_variable_shape_tensor_field(&ragged, "sparse_matvec", 2)
            .expect("ragged contract");
        assert_eq!(ragged.value_type, DataType::Float64);
        assert_eq!(ragged.dimensions, 1);
        assert_eq!(ragged.uniform_shape, Some(vec![None]));
        assert_eq!(
            parse_csr_matrix_batch_field(&Arc::new(csr_field), "sparse_matvec", 1)
                .expect("csr batch field"),
            DataType::Float64
        );
    }

    #[test]
    fn parse_helpers_reject_mismatches_and_missing_metadata() {
        let scalar = scalar_field("scalar", &DataType::Float64, false);
        let vector = vector_field("vector", &DataType::Float64, 4, false).expect("vector field");
        let plain_nested_complex = Arc::new(Field::new(
            "complex_vectors",
            DataType::new_fixed_size_list(
                DataType::new_fixed_size_list(DataType::Float64, 2, false),
                3,
                false,
            ),
            false,
        ));
        let rank_three = fixed_shape_tensor_field("tensor", &DataType::Float64, &[2, 3, 4], false)
            .expect("rank-3 tensor");
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
            parse_vector_field(&scalar, "vector_dot", 1).expect_err("vector type mismatch");
        let complex_error =
            parse_complex_vector_field(&plain_nested_complex, "vector_dot_hermitian", 1)
                .expect_err("complex vector type mismatch");
        let tensor_error = parse_tensor_batch_field(&vector, "matrix_matmul", 1)
            .expect_err("tensor type mismatch");
        let rank_error = parse_matrix_batch_field(&rank_three, "matrix_matmul", 1)
            .expect_err("matrix rank mismatch");
        let missing_fixed_error =
            parse_tensor_batch_field(&missing_tensor_metadata, "matrix_matmul", 1)
                .expect_err("missing tensor metadata");
        let missing_variable_error =
            parse_variable_shape_tensor_field(&missing_variable_metadata, "sparse_matvec", 2)
                .expect_err("missing variable metadata");
        let csr_error =
            parse_csr_matrix_batch_field(&scalar, "sparse_matvec", 1).expect_err("csr mismatch");

        assert!(scalar_error.to_string().contains("expected FixedSizeList<Float32|Float64>(D)"));
        assert!(complex_error.to_string().contains("expected FixedSizeList<ndarrow.complex64>(D)"));
        assert!(tensor_error.to_string().contains("matrix_matmul"));
        assert!(rank_error.to_string().contains("batch of rank-2 matrices"));
        assert!(missing_fixed_error.to_string().contains("matrix_matmul"));
        assert!(missing_variable_error.to_string().contains("sparse_matvec"));
        assert!(csr_error.to_string().contains("sparse_matvec"));
    }

    #[test]
    fn parse_helpers_accept_float32_extension_value_types() {
        let (tensor_field, _array) = ndarrow::arrayd_to_fixed_shape_tensor(
            "tensor",
            Array3::<f32>::zeros((1, 2, 2)).into_dyn(),
        )
        .expect("f32 tensor");
        let (ragged_field, _array) = ndarrow::arrays_to_variable_shape_tensor(
            "ragged",
            vec![Array1::<f32>::zeros(2).into_dyn()],
            Some(vec![None]),
        )
        .expect("f32 ragged tensor");

        let tensor_contract = parse_tensor_batch_field(&Arc::new(tensor_field), "matrix_matmul", 1)
            .expect("f32 tensor");
        let ragged_contract =
            parse_variable_shape_tensor_field(&Arc::new(ragged_field), "sparse_matvec", 2)
                .expect("f32 ragged tensor");

        assert_eq!(tensor_contract.value_type, DataType::Float32);
        assert_eq!(tensor_contract.shape, vec![2, 2]);
        assert_eq!(ragged_contract.value_type, DataType::Float32);
        assert_eq!(ragged_contract.uniform_shape, Some(vec![None]));
    }

    #[test]
    fn parse_helpers_reject_non_float_extension_value_types() {
        let mut tensor_metadata = HashMap::new();
        drop(
            tensor_metadata
                .insert("ARROW:extension:name".to_owned(), "arrow.fixed_shape_tensor".to_owned()),
        );
        drop(
            tensor_metadata.insert(
                "ARROW:extension:metadata".to_owned(),
                serde_json::to_string(&FixedShapeTensorWireMetadata { shape: vec![2, 2] })
                    .expect("tensor metadata"),
            ),
        );
        let int_tensor_field = Arc::new(
            Field::new("tensor", DataType::new_fixed_size_list(DataType::Int32, 4, false), false)
                .with_metadata(tensor_metadata),
        );

        let mut ragged_metadata = HashMap::new();
        drop(
            ragged_metadata.insert(
                "ARROW:extension:name".to_owned(),
                "arrow.variable_shape_tensor".to_owned(),
            ),
        );
        drop(
            ragged_metadata.insert(
                "ARROW:extension:metadata".to_owned(),
                serde_json::to_string(&VariableShapeTensorWireMetadata {
                    uniform_shape: Some(vec![None]),
                })
                .expect("ragged metadata"),
            ),
        );
        let int_ragged_field = Arc::new(
            Field::new(
                "ragged",
                DataType::Struct(
                    vec![
                        Field::new("data", DataType::new_list(DataType::Int32, false), false),
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
            .with_metadata(ragged_metadata),
        );

        let tensor_error = parse_tensor_batch_field(&int_tensor_field, "matrix_matmul", 1)
            .expect_err("tensor value type mismatch");
        let ragged_error = parse_variable_shape_tensor_field(&int_ragged_field, "sparse_matvec", 2)
            .expect_err("ragged value type mismatch");

        assert!(
            tensor_error.to_string().contains("expected arrow.fixed_shape_tensor<Float32|Float64>")
        );
        assert!(
            ragged_error
                .to_string()
                .contains("expected arrow.variable_shape_tensor<Float32|Float64>")
        );
    }
}
