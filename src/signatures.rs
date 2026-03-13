use datafusion::logical_expr::{Signature, Volatility};

pub(crate) fn any_signature(arg_count: usize) -> Signature {
    Signature::any(arg_count, Volatility::Immutable)
}
