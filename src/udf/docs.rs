use datafusion::logical_expr::{DocSection, Documentation, DocumentationBuilder};

const CONSTRUCTORS_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Constructors",
    description: Some("Canonical numerical constructors for DataFusion SQL."),
};

const VECTOR_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Vector Functions",
    description: Some("Dense vector operations over canonical fixed-size vector batches."),
};

const MATRIX_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Matrix Functions",
    description: Some("Dense matrix operations and configurable matrix functions."),
};

const DECOMPOSITION_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Matrix Decompositions",
    description: Some("Matrix decompositions and related helpers."),
};

const ITERATIVE_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Iterative Solvers",
    description: Some("Iterative linear solvers over canonical matrix and vector batches."),
};

const SPARSE_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Sparse Functions",
    description: Some("Sparse CSR operations over canonical ndarrow sparse batches."),
};

const TENSOR_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion Tensor Functions",
    description: Some("Fixed-shape and variable-shape tensor operations over canonical batches."),
};

const ML_SECTION: DocSection = DocSection {
    include:     true,
    label:       "NDataFusion ML And Statistics",
    description: Some(
        "Statistical and machine learning functions over canonical numerical batches.",
    ),
};

pub(crate) fn constructor_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(CONSTRUCTORS_SECTION, description, syntax_example)
}

pub(crate) fn vector_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(VECTOR_SECTION, description, syntax_example)
}

pub(crate) fn matrix_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(MATRIX_SECTION, description, syntax_example)
}

pub(crate) fn decomposition_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(DECOMPOSITION_SECTION, description, syntax_example)
}

pub(crate) fn iterative_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(ITERATIVE_SECTION, description, syntax_example)
}

pub(crate) fn sparse_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(SPARSE_SECTION, description, syntax_example)
}

pub(crate) fn tensor_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(TENSOR_SECTION, description, syntax_example)
}

pub(crate) fn ml_doc(
    description: impl Into<String>,
    syntax_example: impl Into<String>,
) -> DocumentationBuilder {
    Documentation::builder(ML_SECTION, description, syntax_example)
}
