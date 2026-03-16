use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayRef, StructArray};
use datafusion::arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::{Session, TableFunctionImpl};
use datafusion::common::{DFSchema, Result, plan_datafusion_err};
use datafusion::datasource::TableProvider;
use datafusion::datasource::memory::MemorySourceConfig;
use datafusion::logical_expr::expr_rewriter::normalize_col;
use datafusion::logical_expr::utils::columnize_expr;
use datafusion::logical_expr::{
    EmptyRelation, Expr, ExprSchemable, LogicalPlan, Projection, TableType,
};
use datafusion::physical_plan::ExecutionPlan;

#[derive(Debug)]
pub(crate) struct UnpackStructTableFunction;

impl TableFunctionImpl for UnpackStructTableFunction {
    fn call(&self, exprs: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let [expr] = exprs else {
            return Err(plan_datafusion_err!(
                "unpack_struct requires exactly one scalar struct-valued expression"
            ));
        };

        let data_type = expr.get_type(&DFSchema::empty())?;
        let DataType::Struct(fields) = data_type else {
            return Err(plan_datafusion_err!(
                "unpack_struct requires a struct-valued expression, found {data_type}"
            ));
        };

        let schema = Arc::new(Schema::new(
            fields.iter().map(|field| field.as_ref().clone()).collect::<Vec<_>>(),
        ));
        Ok(Arc::new(UnpackStructTable { schema, expr: expr.clone() }))
    }
}

#[derive(Debug)]
struct UnpackStructTable {
    schema: SchemaRef,
    expr:   Expr,
}

#[async_trait]
impl TableProvider for UnpackStructTable {
    fn as_any(&self) -> &dyn std::any::Any { self }

    fn schema(&self) -> SchemaRef { Arc::clone(&self.schema) }

    fn table_type(&self) -> TableType { TableType::Temporary }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = LogicalPlan::EmptyRelation(EmptyRelation {
            produce_one_row: true,
            schema:          Arc::new(DFSchema::empty()),
        });
        let projected_expr = columnize_expr(normalize_col(self.expr.clone(), &plan)?, &plan)?;
        let logical_plan = Projection::try_new(vec![projected_expr], Arc::new(plan))
            .map(LogicalPlan::Projection)?;
        let physical_plan = state.create_physical_plan(&logical_plan).await?;
        let task_ctx = datafusion::execution::TaskContext::from(state);
        let batches = datafusion::physical_plan::collect(physical_plan, Arc::new(task_ctx)).await?;
        let Some(batch) = batches.first() else {
            return Err(plan_datafusion_err!("unpack_struct expression produced no rows"));
        };
        let struct_array =
            batch.column(0).as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                plan_datafusion_err!("unpack_struct expression did not evaluate to a StructArray")
            })?;
        let output = RecordBatch::try_new(
            Arc::clone(&self.schema),
            struct_array.columns().iter().map(Arc::clone).collect::<Vec<ArrayRef>>(),
        )?;
        Ok(MemorySourceConfig::try_new_exec(
            &[vec![output]],
            Arc::clone(&self.schema),
            projection.cloned(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{ArrayRef, Float64Array, StructArray};
    use datafusion::arrow::datatypes::{DataType, Field};
    use datafusion::catalog::TableFunctionImpl;
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::{Expr, TableType};
    use datafusion::prelude::SessionContext;

    use super::{UnpackStructTable, UnpackStructTableFunction};

    fn struct_literal_expr() -> Expr {
        let struct_array = StructArray::new(
            vec![
                Arc::new(Field::new("sign", DataType::Float64, false)),
                Arc::new(Field::new("log_abs", DataType::Float64, false)),
            ]
            .into(),
            vec![
                Arc::new(Float64Array::from(vec![1.0])) as ArrayRef,
                Arc::new(Float64Array::from(vec![3.5])) as ArrayRef,
            ],
            None,
        );
        Expr::Literal(ScalarValue::Struct(Arc::new(struct_array)), None)
    }

    #[test]
    fn unpack_struct_rejects_wrong_arity_and_non_struct_inputs() {
        let function = UnpackStructTableFunction;

        assert!(function.call(&[]).is_err());
        assert!(function.call(&[Expr::Literal(ScalarValue::Int64(Some(1)), None)]).is_err());
    }

    #[tokio::test]
    async fn unpack_struct_scans_struct_literal_into_columns() {
        let function = UnpackStructTableFunction;
        let provider = function.call(&[struct_literal_expr()]).expect("table provider");
        assert_eq!(provider.schema().fields().len(), 2);
        assert_eq!(provider.table_type(), TableType::Temporary);

        let ctx = SessionContext::new();
        let state = ctx.state();
        let projection = vec![1];
        let exec = provider.scan(&state, Some(&projection), &[], None).await.expect("scan");
        let batches =
            datafusion::physical_plan::collect(exec, ctx.task_ctx()).await.expect("collect");
        let output = &batches[0];
        assert_eq!(output.num_columns(), 1);
        assert_eq!(output.schema().field(0).name(), "log_abs");
        let values = output.column(0).as_any().downcast_ref::<Float64Array>().expect("log_abs");
        assert!((values.value(0) - 3.5).abs() < f64::EPSILON);
        assert!(provider.as_any().downcast_ref::<UnpackStructTable>().is_some());
    }
}
