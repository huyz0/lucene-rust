//! TermWeight and PhraseWeight.scorerSupplier for the SimpleText format.
//! Borrows the native text postings and skip reader through TermPostings.

use super::{similarity, LeafContext, Mode, Result, TermForm, TermLeg, TermQuery};

/// Java: PhraseQuery.rewrite and PhraseWeight.getPhraseMatcher, over native
/// SimpleText positional enums. Uses the shared exact/sloppy matchers and
/// ExactPhraseMatcher.mergeImpacts, including all four text skip levels.
pub(super) fn phrase<'a>(
    ctx: &LeafContext<'a>,
    fields: Option<&'a lucene_codecs::simpletext::fields_reader::SimpleTextFieldsReader>,
    query: &crate::query::PhraseQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<super::BoxScorer<'a>>> {
    use crate::bulk_scorer::PhrasePostings;
    use crate::exec::{
        leaf::TermScorer,
        phrase::{term_positions_cost, PhraseScorer, PhraseTerm},
    };
    use lucene_codecs::simpletext::{self as text, fields_reader::SimpleTextImpactsEnum};

    if query.terms.is_empty() {
        return Ok(None);
    }
    if let [only] = query.terms.as_slice() {
        let term_query = TermQuery::new(&query.field, only.clone());
        return Ok(match term(ctx, fields, &term_query, None, boost, mode)? {
            TermForm::Leg(leg) => Some(Box::new(TermScorer::new(
                *leg,
                mode == Mode::TopScores && top_level,
            ))),
            _ => None,
        });
    }
    let convert = crate::directory_reader::Error::SimpleText;
    let terms = fields
        .map(|fields| fields.terms(&query.field))
        .transpose()
        .map_err(convert)?
        .flatten();
    crate::phrase_positions_check(terms.map(|terms| terms.info().index_options), query)?;
    let Some(terms) = terms else {
        return Ok(None);
    };
    let mut postings = Vec::with_capacity(query.terms.len());
    let mut statistics = Vec::with_capacity(query.terms.len());
    let mut collection = None;
    let mut match_cost = 0.0f32;
    for (slot, term) in query.terms.iter().enumerate() {
        let mut cursor = terms.iterator();
        // SimpleText's own fallible lookup, rather than BlockTree's retired one.
        if !text::fields_reader::SimpleTextTermsEnum::seek_exact(&mut cursor, term)
            .map_err(convert)?
        {
            return Ok(None);
        }
        let cost = i64::from(cursor.doc_freq());
        let stats = ctx
            .global
            .and_then(|g| g.term(&query.field, term))
            .copied()
            .unwrap_or(crate::CollectionStats {
                doc_freq: cost,
                doc_count: i64::from(terms.doc_count()),
                total_term_freq: cursor.total_term_freq(),
                max_doc: i64::from(ctx.max_doc.unwrap_or(0)),
                sum_total_term_freq: terms.sum_total_term_freq(),
                sum_doc_freq: terms.sum_doc_freq(),
            });
        collection.get_or_insert_with(|| stats.collection_statistics());
        statistics.push(stats.term_statistics(term));
        match_cost += term_positions_cost(cost, cursor.total_term_freq());
        let cursor = if mode == Mode::TopScores {
            cursor.impacts(text::POSITIONS).map_err(convert)?
        } else {
            SimpleTextImpactsEnum::Slow(cursor.postings(text::POSITIONS).map_err(convert)?)
        };
        postings.push(PhraseTerm {
            cursor: PhrasePostings::text(cursor)?,
            slot,
            cost,
        });
    }
    let scorer = crate::exec::extended::sim_scorer(
        ctx,
        &query.field,
        boost,
        &collection.expect("a phrase has at least two terms"),
        &statistics,
    );
    let positions = query.positions();
    crate::exec::phrase::sort_exact_phrase_terms(&mut postings, query, &positions);
    let first = positions[0];
    let offsets: Vec<_> = positions
        .into_iter()
        .map(|position| position.wrapping_sub(first))
        .collect();
    Ok(Some(Box::new(
        PhraseScorer::new(
            postings,
            0.0,
            query.slop,
            crate::sloppy_phrase::PhraseRepeats::for_phrase_positions(&query.terms, &offsets),
            ctx.norms
                .and_then(|norms| norms.get(&query.field))
                .map(|n| n.cursor()),
            match_cost,
            mode == Mode::TopScores,
            mode.needs_scores(),
        )
        .with_sim_scorer(scorer)
        .with_offsets(offsets),
    )))
}

/// Java: TermWeight.scorerSupplier over SimpleTextTermsEnum.impacts/postings.
/// Borrows the core's cached terms, keeping the native skip list and line
/// cursor alive for the same scorer tree used by the binary format.
pub(super) fn term<'a>(
    ctx: &LeafContext<'a>,
    fields: Option<&'a lucene_codecs::simpletext::fields_reader::SimpleTextFieldsReader>,
    t: &TermQuery,
    constant: Option<f32>,
    boost: f32,
    mode: Mode,
) -> Result<TermForm<'a>> {
    use lucene_codecs::simpletext::{self as text, fields_reader::SimpleTextImpactsEnum};
    let convert = crate::directory_reader::Error::SimpleText;
    let Some(terms) = fields
        .map(|fields| fields.terms(&t.field))
        .transpose()
        .map_err(convert)?
        .flatten()
    else {
        return Ok(TermForm::Absent);
    };
    let mut cursor = terms.iterator();
    // SimpleText's own fallible lookup, rather than BlockTree's retired one.
    if !text::fields_reader::SimpleTextTermsEnum::seek_exact(&mut cursor, &t.term)
        .map_err(convert)?
    {
        return Ok(TermForm::Absent);
    }
    let cost = i64::from(cursor.doc_freq());
    let scoring = mode.needs_scores() && constant.is_none();
    let flags = if scoring { text::FREQS } else { text::NONE };
    let postings = if scoring && mode == Mode::TopScores {
        cursor.impacts(flags).map_err(convert)?
    } else {
        SimpleTextImpactsEnum::Slow(cursor.postings(flags).map_err(convert)?)
    };
    let postings = crate::bulk_scorer::TermPostings::text(postings, terms.has_freqs())?;
    if !scoring {
        return Ok(TermForm::Leg(Box::new(match constant {
            Some(score) if mode.needs_scores() => TermLeg::constant(postings, cost, score),
            _ => TermLeg::filter(postings, cost),
        })));
    }
    let stats = ctx
        .global
        .and_then(|g| g.term(&t.field, &t.term))
        .copied()
        .unwrap_or(crate::CollectionStats {
            doc_freq: cost,
            doc_count: i64::from(terms.doc_count()),
            total_term_freq: cursor.total_term_freq(),
            max_doc: i64::from(ctx.max_doc.unwrap_or(0)),
            sum_total_term_freq: terms.sum_total_term_freq(),
            sum_doc_freq: terms.sum_doc_freq(),
        });
    let norms = ctx
        .norms
        .and_then(|norms| norms.get(&t.field))
        .map(|n| n.cursor());
    let leg = if let Some(sim) = ctx.similarity {
        let scorer = sim.scorer(
            &t.field,
            boost,
            &stats.collection_statistics(),
            &[stats.term_statistics(&t.term)],
        );
        TermLeg::scoring_sim(postings, scorer, norms, cost)
    } else {
        TermLeg::scoring(
            postings,
            boost * similarity::idf(stats.doc_freq, stats.doc_count),
            norms,
            cost,
            // SimpleText's SlowImpactsEnum and tail keep Java's dummy
            // (Integer.MAX_VALUE, 1), including fields without frequencies.
            f32::MAX,
        )
    };
    Ok(TermForm::Leg(Box::new(leg)))
}

#[cfg(test)]
mod tests;
