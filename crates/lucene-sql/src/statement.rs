//! What a statement says before it runs: the table it reads.

use datafusion::sql::sqlparser::ast::{
    ObjectNamePart, Query, Select, SetExpr, Statement, TableFactor,
};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use datafusion::sql::sqlparser::tokenizer::{Token, Tokenizer};

use crate::error::{Error, Result};

/// The name of the table `sql` reads, as the engine resolves it (unquoted
/// names fold to lower case).
///
/// Only a query over one table has one; a join, a subquery in `FROM` or a
/// statement that is not a query is [`Error::Unsupported`], so the caller can
/// resolve the table before it opens anything.
///
/// # Errors
/// A statement that does not parse, or does not read exactly one table.
pub fn table_name(sql: &str) -> Result<String> {
    check_operators(sql)?;
    let mut statements = Parser::parse_sql(&GenericDialect {}, sql)
        .map_err(|e| Error::unsupported(format!("sql: {e}")))?;
    if statements.len() != 1 {
        return Err(Error::unsupported("exactly one statement is expected"));
    }
    let Statement::Query(query) = statements.remove(0) else {
        return Err(Error::unsupported("only queries run: this is not a SELECT"));
    };
    let mut found = Vec::new();
    collect_query(&query, &mut found)?;
    found.dedup();
    match found.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(Error::unsupported("the statement reads no table")),
        _ => Err(Error::unsupported("a statement over several tables")),
    }
}

/// The most binary operators (`AND`, `OR`, arithmetic, `||`) one statement may chain.
/// The parser, the planner and the tree's drop all recurse over a chain's length, so
/// one statement of thousands of terms would overflow the stack of the thread running it.
const MAX_OPERATORS: usize = 1000;

/// Refuses a statement that chains more operators than [`MAX_OPERATORS`], before
/// anything recursive sees it.
///
/// # Errors
/// A statement with too many operators, or one that does not tokenize.
pub fn check_operators(sql: &str) -> Result<()> {
    let tokens = Tokenizer::new(&GenericDialect {}, sql)
        .tokenize()
        .map_err(|e| Error::unsupported(format!("sql: {e}")))?;
    let operators = tokens
        .iter()
        .filter(|t| match t {
            Token::Word(w) => {
                w.quote_style.is_none()
                    && (w.value.eq_ignore_ascii_case("and") || w.value.eq_ignore_ascii_case("or"))
            }
            Token::Plus
            | Token::Minus
            | Token::Mul
            | Token::Div
            | Token::Mod
            | Token::StringConcat => true,
            _ => false,
        })
        .count();
    if operators > MAX_OPERATORS {
        return Err(Error::unsupported(format!(
            "sql: a statement may chain at most {MAX_OPERATORS} operators"
        )));
    }
    Ok(())
}

fn collect_query(query: &Query, found: &mut Vec<String>) -> Result<()> {
    if query.with.is_some() {
        return Err(Error::unsupported("WITH is not supported"));
    }
    collect_set(&query.body, found)
}

fn collect_set(set: &SetExpr, found: &mut Vec<String>) -> Result<()> {
    match set {
        SetExpr::Select(select) => collect_select(select, found),
        SetExpr::Query(q) => collect_query(q, found),
        SetExpr::SetOperation { left, right, .. } => {
            collect_set(left, found)?;
            collect_set(right, found)
        }
        _ => Err(Error::unsupported("this form of query is not supported")),
    }
}

fn collect_select(select: &Select, found: &mut Vec<String>) -> Result<()> {
    for item in &select.from {
        if !item.joins.is_empty() {
            return Err(Error::unsupported("a join is not supported"));
        }
        match &item.relation {
            TableFactor::Table { name, .. } => {
                let [ObjectNamePart::Identifier(id)] = name.0.as_slice() else {
                    return Err(Error::unsupported(format!(
                        "`{name}`: only a table name is supported, not a schema-qualified one"
                    )));
                };
                found.push(if id.quote_style.is_some() {
                    id.value.clone()
                } else {
                    id.value.to_lowercase()
                });
            }
            TableFactor::Derived { subquery, .. } => collect_query(subquery, found)?,
            _ => return Err(Error::unsupported("this FROM item is not supported")),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_the_one_table_a_query_reads() {
        assert_eq!(table_name("SELECT * FROM Logs").unwrap(), "logs");
        assert_eq!(
            table_name("select a from \"My-Index\" where a = 'x'").unwrap(),
            "My-Index"
        );
        assert_eq!(
            table_name("SELECT k, count(*) FROM (SELECT k FROM t) GROUP BY k").unwrap(),
            "t"
        );
        assert_eq!(
            table_name("SELECT a FROM t UNION ALL SELECT a FROM t").unwrap(),
            "t"
        );
    }

    #[test]
    fn a_statement_of_thousands_of_terms_is_refused_not_planned() {
        let long = format!("SELECT a FROM t WHERE a = 1{}", " OR a = 0".repeat(20_000));
        assert!(table_name(&long).is_err());
        let ok = format!("SELECT a FROM t WHERE a = 1{}", " OR a = 0".repeat(400));
        assert_eq!(table_name(&ok).unwrap(), "t");
        let ins = format!("SELECT a FROM t WHERE a IN (1{})", ",1".repeat(20_000));
        assert_eq!(table_name(&ins).unwrap(), "t");
    }

    #[test]
    fn refuses_what_is_not_a_query_over_one_table() {
        for bad in [
            "SELECT 1",
            "SELECT * FROM a, b",
            "SELECT * FROM a JOIN b ON a.x = b.x",
            "SELECT * FROM s.t",
            "DELETE FROM t",
            "SELECT 1; SELECT 2",
            "SELEC",
            "WITH c AS (SELECT 1) SELECT * FROM c",
        ] {
            assert!(table_name(bad).is_err(), "{bad}");
        }
    }
}
