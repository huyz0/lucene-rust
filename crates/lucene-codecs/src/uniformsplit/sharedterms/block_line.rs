//! Port of `STBlockLine` and its `Serializer`, and of
//! `FieldMetadataTermState`: a line's term states, one per field that has
//! the line's term.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use super::super::field_metadata::FieldMetadata;
use super::super::{
    corrupt, read_vint, BlockHeader, DeltaBaseTermStateSerializer, Result, UniformSplitTermState,
};
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::postings::TermMetadata;

/// `FieldMetadataTermState`: one field's term state on a line being
/// written; `field` indexes the writer's field list (Java: the field's
/// `FieldMetadata`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldMetadataTermState {
    /// The field, as an index into the writer's fields.
    pub field: usize,
    /// `docFreq`.
    pub doc_freq: i32,
    /// `totalTermFreq` (`-1` for a field without frequencies).
    pub total_term_freq: i64,
    /// The postings writer's term metadata.
    pub meta: TermMetadata,
}

/// `STBlockLine.Serializer.writeLineTermStates(termStatesOutput, line,
/// encoder)`: `states` in field-number order, never empty.
pub(super) fn write_line_term_states(
    out: &mut Vec<u8>,
    serializer: &mut DeltaBaseTermStateSerializer,
    states: &[FieldMetadataTermState],
    fields: &[FieldMetadata],
) {
    let write_state = |out: &mut Vec<u8>,
                       serializer: &mut DeltaBaseTermStateSerializer,
                       state: &FieldMetadataTermState| {
        let info = &fields[state.field].field_info;
        serializer.write_term_state(
            out,
            info.index_options,
            info.store_payloads,
            state.doc_freq,
            state.total_term_freq,
            &state.meta,
        );
    };
    if let [state] = states {
        // A field number is non-negative: its negation does not overflow.
        out.write_zint(fields[state.field].field_info.number.wrapping_neg());
        write_state(out, serializer, state);
        return;
    }
    // At most the group's field count, an `i32`.
    out.write_zint(states.len() as i32);
    for state in states {
        out.write_vint(fields[state.field].field_info.number);
    }
    for state in states {
        write_state(out, serializer, state);
    }
}

/// The segment's `FieldInfos`, by number, as `STBlockLine.Serializer`
/// reads a line's term states with them.
#[derive(Debug)]
pub(crate) struct SharedFieldInfos {
    /// The fields, sorted by number.
    by_number: Vec<FieldInfo>,
}

/// `ByteArrayDataInput.readZInt` at `*pos`.
fn read_zint(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_zint()?;
    *pos = input.position();
    Ok(v)
}

impl SharedFieldInfos {
    /// The fields of `infos`.
    pub(crate) fn new(infos: &FieldInfos) -> Self {
        let mut by_number = infos.fields.clone();
        by_number.sort_by_key(|info| info.number);
        SharedFieldInfos { by_number }
    }

    /// `fieldInfos.fieldInfo(number)`.
    pub(super) fn field(&self, number: i32) -> Result<&FieldInfo> {
        self.by_number
            .binary_search_by_key(&number, |info| info.number)
            .map(|i| &self.by_number[i])
            .map_err(|_| corrupt(format!("unknown field number {number} in a block line")))
    }

    /// `termStateSerializer.readTermState(base*FP, termStatesInput,
    /// fieldInfos.fieldInfo(number), reuse)`.
    fn read_state(
        &self,
        number: i32,
        buf: &[u8],
        pos: &mut usize,
        header: &BlockHeader,
    ) -> Result<UniformSplitTermState> {
        let info = self.field(number)?;
        DeltaBaseTermStateSerializer::read_term_state(
            header.base_docs_fp,
            header.base_positions_fp,
            header.base_payloads_fp,
            buf,
            pos,
            info.index_options,
            info.store_payloads,
        )
    }

    /// `readTermStateForField(fieldId, termStatesInput, termStateSerializer,
    /// blockHeader, fieldInfos, reuse)`: field `field_id`'s term state on
    /// the line at `*pos`, `None` when the line's term is not the field's.
    ///
    /// # Errors
    /// `Corrupted` for a truncated line or a field number `FieldInfos`
    /// lacks.
    pub(crate) fn read_term_state_for_field(
        &self,
        field_id: i32,
        buf: &[u8],
        pos: &mut usize,
        header: &BlockHeader,
    ) -> Result<Option<UniformSplitTermState>> {
        let num_fields = read_zint(buf, pos)?;
        if num_fields <= 0 {
            // `i32::MIN` stays `i32::MIN`, as Java's negation: no field's.
            let read_field_id = num_fields.wrapping_neg();
            if field_id == read_field_id {
                return self.read_state(read_field_id, buf, pos, header).map(Some);
            }
            return Ok(None);
        }
        // Each id reads at least one byte, so a corrupt count stops at the
        // block's end.
        let ids_start = *pos;
        let mut is_field_in_list = false;
        for _ in 0..num_fields {
            let read_field_id = read_vint(buf, pos)?;
            if !is_field_in_list && read_field_id > field_id {
                return Ok(None);
            }
            is_field_in_list |= read_field_id == field_id;
        }
        if is_field_in_list {
            // The ids again (Java: `readFieldIds` kept them in an array).
            let mut ids_pos = ids_start;
            for _ in 0..num_fields {
                let read_field_id = read_vint(buf, &mut ids_pos)?;
                let state = self.read_state(read_field_id, buf, pos, header)?;
                if read_field_id == field_id {
                    return Ok(Some(state));
                }
            }
        }
        Ok(None)
    }

    /// `readFieldTermStatesMap(termStatesInput, termStateSerializer,
    /// blockHeader, fieldInfos, fieldTermStatesMap)`: every field's term
    /// state on the line at `*pos`, as `(fieldNumber, state)` in the line's
    /// order.
    ///
    /// # Errors
    /// As [`Self::read_term_state_for_field`].
    pub(crate) fn read_field_term_states(
        &self,
        buf: &[u8],
        pos: &mut usize,
        header: &BlockHeader,
        out: &mut Vec<(i32, UniformSplitTermState)>,
    ) -> Result<()> {
        out.clear();
        let num_fields = read_zint(buf, pos)?;
        if num_fields <= 0 {
            let field_id = num_fields.wrapping_neg();
            out.push((field_id, self.read_state(field_id, buf, pos, header)?));
            return Ok(());
        }
        // `readFieldIds`: skipped here, read again below.
        let mut ids_pos = *pos;
        for _ in 0..num_fields {
            read_vint(buf, pos)?;
        }
        for _ in 0..num_fields {
            let field_id = read_vint(buf, &mut ids_pos)?;
            out.push((field_id, self.read_state(field_id, buf, pos, header)?));
        }
        Ok(())
    }
}
