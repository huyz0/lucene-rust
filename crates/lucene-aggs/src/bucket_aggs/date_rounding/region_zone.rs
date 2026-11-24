//! Rounding in an IANA region zone (`Europe/Paris`, `America/Sao_Paulo`): the
//! `JavaTime*Rounding` classes of `Rounding.java`, which are the reference the
//! prepared (array, `LocalTimeOffset`) variants must agree with.
//!
//! A region zone's offset changes, so a bucket is not a fixed number of
//! milliseconds wide: a day is 23 or 25 hours long across a daylight-saving
//! change, and a midnight inside a gap starts the day at the gap's end.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock, PoisonError};

use jiff::tz::TimeZone;
use jiff::Timestamp;

use super::{DateUnit, RoundingKind};

/// A zone's rules, kept for the life of the process: a handle is `Copy`, and
/// the table holds one entry for each id ever asked for, which is at most the
/// few hundred in the tz database.
struct ZoneData {
    id: String,
    tz: TimeZone,
}

/// An IANA region zone: its id and its transition rules.
#[derive(Clone, Copy)]
pub struct RegionZone(&'static ZoneData);

impl PartialEq for RegionZone {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for RegionZone {}

impl std::fmt::Debug for RegionZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegionZone({})", self.0.id)
    }
}

/// The first and last milliseconds a zone's rules answer for; a date outside
/// takes the rules of the nearest one inside.
const MIN_MS: i64 = -377_705_116_800_000;
const MAX_MS: i64 = 253_402_207_199_999;

fn timestamp(ms: i64) -> Timestamp {
    Timestamp::from_millisecond(ms.clamp(MIN_MS, MAX_MS)).unwrap_or(Timestamp::UNIX_EPOCH)
}

impl RegionZone {
    /// The zone `id` names, spelled as in the tz database
    /// (`ZoneId.of` is case-sensitive); `None` for anything else, among them
    /// a fixed offset (`+02:00`, `UTC+2`), which a [`super::DateRounding`]
    /// holds as `zone_ms`.
    pub fn named(id: &str) -> Option<Self> {
        static ZONES: OnceLock<Mutex<HashMap<String, &'static ZoneData>>> = OnceLock::new();
        let table = ZONES.get_or_init(Default::default);
        let mut table = table.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(found) = table.get(id) {
            return Some(Self(found));
        }
        let tz = TimeZone::get(id).ok()?;
        if tz.iana_name() != Some(id) {
            return None;
        }
        let data: &'static ZoneData = Box::leak(Box::new(ZoneData {
            id: id.to_owned(),
            tz,
        }));
        table.insert(id.to_owned(), data);
        Some(Self(data))
    }

    /// The zone's id.
    pub fn id(&self) -> &str {
        &self.0.id
    }

    /// `ZoneRules.getOffset(Instant)`, in milliseconds.
    pub fn offset_ms(&self, utc: i64) -> i64 {
        i64::from(self.0.tz.to_offset(timestamp(utc)).seconds()) * 1000
    }

    /// `ZoneRules.previousTransition(Instant)`: the latest offset change
    /// strictly before `utc`, in epoch milliseconds.
    fn previous_transition(&self, utc: i64) -> Option<i64> {
        if utc <= MIN_MS {
            return None;
        }
        self.0
            .tz
            .preceding(timestamp(utc))
            .map(|t| t.timestamp().as_millisecond())
            .find(|&t| self.offset_ms(t) != self.offset_ms(t - 1))
    }

    /// The offset changes within a day either side of `local`, as
    /// `(instant, offset before, offset after)`.
    fn transitions_near(&self, local: i64) -> Vec<(i64, i64, i64)> {
        const DAY: i64 = 86_400_000;
        let from = local.saturating_sub(2 * DAY);
        let to = local.saturating_add(2 * DAY);
        self.0
            .tz
            .following(timestamp(from))
            .map(|t| t.timestamp().as_millisecond())
            .take_while(|&t| t <= to)
            .map(|t| (t, self.offset_ms(t - 1), self.offset_ms(t)))
            .filter(|&(_, before, after)| before != after)
            .collect()
    }

    /// `ZoneRules.getValidOffsets(LocalDateTime)` in milliseconds, the
    /// earliest instant first: none inside a gap, two inside an overlap.
    fn valid_offsets(&self, local: i64) -> ([i64; 2], usize) {
        let mut found = [0i64; 2];
        let mut n = 0usize;
        // The offset in force before, at and after the local time, which are
        // all the offsets a transition within a day of it can have changed to.
        let mut candidates = [
            self.offset_ms(local.saturating_sub(86_400_000 + 18 * 3_600_000)),
            self.offset_ms(local),
            self.offset_ms(local.saturating_add(86_400_000 + 18 * 3_600_000)),
        ];
        // The earliest instant is the largest offset.
        candidates.sort_unstable_by(|a, b| b.cmp(a));
        let mut last = None;
        for o in candidates {
            if last == Some(o) {
                continue;
            }
            last = Some(o);
            if self.offset_ms(local.saturating_sub(o)) == o {
                if let Some(slot) = found.get_mut(n) {
                    *slot = o;
                    n += 1;
                }
            }
        }
        (found, n)
    }

    /// `ZoneRules.getTransition(LocalDateTime)` for a local time in a gap:
    /// the instant the gap starts.
    fn gap_start(&self, local: i64) -> i64 {
        for (at, before, after) in self.transitions_near(local) {
            let gap_from = at.saturating_add(before);
            let gap_to = at.saturating_add(after);
            if local >= gap_from && local < gap_to {
                return at;
            }
        }
        local.saturating_sub(self.offset_ms(local))
    }

    /// `firstTimeOnDay`: the first instant of the local day that starts at
    /// `local_midnight`.
    fn first_time_on_day(&self, local_midnight: i64) -> i64 {
        let (offsets, n) = self.valid_offsets(local_midnight);
        if n > 0 {
            local_midnight.saturating_sub(offsets[0])
        } else {
            self.gap_start(local_midnight)
        }
    }

    /// `truncateAsLocalTime`: the latest instant no later than `utc` at which
    /// the local time was the start of the unit, `None` where that start
    /// fell in a gap.
    fn truncate_as_local_time(&self, unit: DateUnit, utc: i64) -> Option<i64> {
        let local = utc.saturating_add(self.offset_ms(utc));
        let truncated = unit.round_floor(local);
        let (offsets, n) = self.valid_offsets(truncated);
        offsets
            .iter()
            .take(n)
            .rev()
            .map(|&o| truncated.saturating_sub(o))
            .find(|&at| at <= utc)
    }

    /// `JavaTimeToMidnightRounding.round` and `JavaTimeNotToMidnightRounding.round`.
    fn round_unit(&self, unit: DateUnit, utc: i64) -> i64 {
        if unit_rounds_to_midnight(unit) {
            let local = utc.saturating_add(self.offset_ms(utc));
            return self.first_time_on_day(unit.round_floor(local));
        }
        let mut instant = utc;
        loop {
            let truncated = self.truncate_as_local_time(unit, instant);
            let Some(previous) = self.previous_transition(instant) else {
                return truncated.unwrap_or(instant);
            };
            if truncated.is_some_and(|t| previous <= t) {
                return truncated.unwrap_or(instant);
            }
            // A transition between the time and its truncation: round the
            // moment before the transition down instead.
            instant = previous - 1;
        }
    }

    /// `JavaTimeToMidnightRounding.nextRoundingValue` and
    /// `AbstractNotToMidnightRounding.nextRoundingValue`; `None` past the
    /// range.
    fn next_unit(&self, unit: DateUnit, utc: i64) -> Option<i64> {
        if unit_rounds_to_midnight(unit) {
            let local = utc.checked_add(self.offset_ms(utc))?;
            let earlier = unit.round_floor(local);
            return Some(self.first_time_on_day(unit.next_start(earlier)?));
        }
        let step = unit_millis(unit);
        let after_one = self.round_unit(unit, utc.checked_add(step)?);
        if utc < after_one {
            Some(after_one)
        } else {
            Some(self.round_unit(unit, utc.checked_add(step.checked_mul(2)?)?))
        }
    }

    /// `JavaTimeRounding.round`: a fixed interval counted in local time.
    fn round_interval(&self, interval: i64, utc: i64) -> i64 {
        let mut utc = utc;
        loop {
            let local = utc.saturating_add(self.offset_ms(utc));
            let key = if local < 0 {
                local.saturating_sub(interval).saturating_add(1) / interval
            } else {
                local / interval
            };
            let rounded_local = key.saturating_mul(interval);
            let (offsets, n) = self.valid_offsets(rounded_local);
            if n == 0 {
                // The rounded time is inside a gap: the gap's start.
                return self.gap_start(rounded_local);
            }
            let previous = self.previous_transition(utc.saturating_add(1));
            let mut crossed = None;
            for &o in offsets.iter().take(n).rev() {
                let at = rounded_local.saturating_sub(o);
                if previous.is_some_and(|p| at < p) {
                    // Rounding down across the transition can give the wrong
                    // answer: round the moment before the transition instead.
                    crossed = previous;
                    break;
                }
                if at <= utc {
                    return at;
                }
            }
            match crossed {
                Some(p) => utc = p - 1,
                None => return offsets[0].wrapping_neg().saturating_add(rounded_local),
            }
        }
    }

    /// `JavaTimeRounding.nextRoundingValue`: `round` as a black box, searched
    /// by halving the step.
    fn next_interval(&self, interval: i64, utc: i64) -> i64 {
        let previous = self.round_interval(interval, utc);
        let mut increment = interval;
        let mut from = previous;
        for _ in 1..100 {
            from = from.saturating_add(increment);
            let rounded = self.round_interval(interval, from);
            if rounded <= previous {
                if increment < 0 {
                    increment = -increment / 2;
                }
                continue;
            }
            if self.round_interval(interval, rounded.saturating_sub(1)) > previous {
                if increment > 0 {
                    increment = -increment / 2;
                }
                continue;
            }
            return rounded;
        }
        self.round_interval(interval, from)
    }

    /// `Prepared.round` of a rounding of `kind` in this zone.
    pub(super) fn round(&self, kind: RoundingKind, utc: i64) -> i64 {
        match kind {
            RoundingKind::Unit(unit) => self.round_unit(unit, utc),
            RoundingKind::Interval(interval) => self.round_interval(interval, utc),
        }
    }

    /// `Prepared.nextRoundingValue` of a rounding of `kind` in this zone.
    pub(super) fn next(&self, kind: RoundingKind, utc: i64) -> Option<i64> {
        match kind {
            RoundingKind::Unit(unit) => self.next_unit(unit, utc),
            RoundingKind::Interval(interval) => Some(self.next_interval(interval, utc)),
        }
    }
}

/// `unitRoundsToMidnight`: the unit's base unit is longer than an hour.
fn unit_rounds_to_midnight(unit: DateUnit) -> bool {
    !matches!(unit, DateUnit::Hour | DateUnit::Minute | DateUnit::Second)
}

/// The base unit's length for a unit that does not round to midnight.
fn unit_millis(unit: DateUnit) -> i64 {
    match unit {
        DateUnit::Hour => 3_600_000,
        DateUnit::Minute => 60_000,
        _ => 1_000,
    }
}
