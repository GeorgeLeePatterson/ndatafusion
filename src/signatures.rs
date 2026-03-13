use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::logical_expr::{Signature, Volatility};

const FIXED_SIZE_LIST_WILDCARD: i32 = i32::MIN;

pub(crate) fn any_signature(arg_count: usize) -> Signature {
    Signature::any(arg_count, Volatility::Immutable)
}

pub(crate) fn float64_vector_signature(arg_count: usize) -> Signature {
    let vector_type = DataType::FixedSizeList(
        Arc::new(Field::new_list_field(DataType::Float64, false)),
        FIXED_SIZE_LIST_WILDCARD,
    );
    Signature::exact(vec![vector_type; arg_count], Volatility::Immutable)
}
