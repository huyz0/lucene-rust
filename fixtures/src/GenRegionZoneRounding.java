import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.DayOfWeek;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.ZoneId;
import java.time.ZoneOffset;
import java.time.temporal.ChronoField;
import java.time.temporal.TemporalQueries;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.time.zone.ZoneOffsetTransition;
import java.time.zone.ZoneRules;

/**
 * Ground truth for {@code lucene-aggs}' region-zone {@code date_histogram} rounding: the
 * {@code JavaTimeToMidnightRounding}, {@code JavaTimeNotToMidnightRounding} and
 * {@code JavaTimeRounding} classes of OpenSearch's {@code Rounding.java}, copied with their
 * logic unchanged (only {@code java.time} is used), behind {@code OffsetRounding}. Each row is
 * {@code zone kind offset utc round next}: the bucket of {@code utc} and the start of the next one.
 */
public class GenRegionZoneRounding {
  static final String[] ZONES = {
    "Europe/Paris", "America/New_York", "America/Sao_Paulo", "Australia/Lord_Howe",
    "Asia/Kolkata", "America/St_Johns", "Pacific/Apia", "Africa/Casablanca",
    "Europe/Dublin", "America/Havana", "Pacific/Chatham"
  };
  static final String[] UNITS = {"second", "minute", "hour", "day", "week", "month", "quarter", "year"};
  static final long[] INTERVALS = {3_600_000L, 5_400_000L, 86_400_000L, 7 * 3_600_000L};

  final ZoneId timeZone;
  final String unit; // null for an interval
  final long interval;

  GenRegionZoneRounding(ZoneId timeZone, String unit, long interval) {
    this.timeZone = timeZone;
    this.unit = unit;
    this.interval = interval;
  }

  boolean roundsToMidnight() {
    return !(unit.equals("hour") || unit.equals("minute") || unit.equals("second"));
  }

  long unitMillis() {
    return switch (unit) {
      case "hour" -> 3_600_000L;
      case "minute" -> 60_000L;
      default -> 1_000L;
    };
  }

  LocalDateTime truncateLocalDateTime(LocalDateTime l) {
    return switch (unit) {
      case "second" -> l.withNano(0);
      case "minute" -> LocalDateTime.of(l.getYear(), l.getMonthValue(), l.getDayOfMonth(), l.getHour(), l.getMinute(), 0, 0);
      case "hour" -> LocalDateTime.of(l.getYear(), l.getMonth(), l.getDayOfMonth(), l.getHour(), 0, 0);
      case "day" -> l.query(TemporalQueries.localDate()).atStartOfDay();
      case "week" -> LocalDateTime.of(l.toLocalDate(), LocalTime.MIDNIGHT).with(ChronoField.DAY_OF_WEEK, 1);
      case "month" -> LocalDateTime.of(l.getYear(), l.getMonthValue(), 1, 0, 0);
      case "quarter" -> LocalDateTime.of(l.getYear(), l.getMonth().firstMonthOfQuarter(), 1, 0, 0);
      default -> LocalDateTime.of(LocalDate.of(l.getYear(), 1, 1), LocalTime.MIDNIGHT);
    };
  }

  // ---- JavaTimeToMidnightRounding / JavaTimeNotToMidnightRounding ----
  long unitRound(long utcMillis) {
    if (roundsToMidnight()) {
      LocalDateTime localDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(utcMillis), timeZone);
      return firstTimeOnDay(truncateLocalDateTime(localDateTime));
    }
    Instant instant = Instant.ofEpochMilli(utcMillis);
    final ZoneRules rules = timeZone.getRules();
    while (true) {
      final Instant truncatedTime = truncateAsLocalTime(instant, rules);
      final ZoneOffsetTransition previousTransition = rules.previousTransition(instant);
      if (previousTransition == null) {
        return truncatedTime.toEpochMilli();
      }
      Instant previousTransitionInstant = previousTransition.getInstant();
      if (truncatedTime != null && previousTransitionInstant.compareTo(truncatedTime) < 1) {
        return truncatedTime.toEpochMilli();
      }
      instant = previousTransitionInstant.minusNanos(1_000_000);
    }
  }

  long unitNext(long utcMillis) {
    if (roundsToMidnight()) {
      LocalDateTime localDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(utcMillis), timeZone);
      LocalDateTime earlier = truncateLocalDateTime(localDateTime);
      LocalDateTime next =
          switch (unit) {
            case "day" -> earlier.plusDays(1);
            case "week" -> earlier.plusDays(7);
            case "month" -> earlier.plusMonths(1);
            case "quarter" -> earlier.plusMonths(3);
            default -> earlier.plusYears(1);
          };
      return firstTimeOnDay(next);
    }
    long unitMillis = unitMillis();
    final long roundedAfterOneIncrement = unitRound(utcMillis + unitMillis);
    if (utcMillis < roundedAfterOneIncrement) {
      return roundedAfterOneIncrement;
    } else {
      return unitRound(utcMillis + 2 * unitMillis);
    }
  }

  long firstTimeOnDay(LocalDateTime localMidnight) {
    final List<ZoneOffset> currentOffsets = timeZone.getRules().getValidOffsets(localMidnight);
    if (currentOffsets.isEmpty() == false) {
      return localMidnight.atOffset(currentOffsets.get(0)).toInstant().toEpochMilli();
    } else {
      ZoneOffsetTransition t = timeZone.getRules().getTransition(localMidnight);
      return t.getInstant().toEpochMilli();
    }
  }

  Instant truncateAsLocalTime(Instant instant, final ZoneRules rules) {
    LocalDateTime localDateTime = LocalDateTime.ofInstant(instant, timeZone);
    final LocalDateTime truncated = truncateLocalDateTime(localDateTime);
    final List<ZoneOffset> currentOffsets = rules.getValidOffsets(truncated);
    if (currentOffsets.isEmpty() == false) {
      for (int offsetIndex = currentOffsets.size() - 1; offsetIndex >= 0; offsetIndex--) {
        final Instant result = truncated.atOffset(currentOffsets.get(offsetIndex)).toInstant();
        if (result.isAfter(instant) == false) {
          return result;
        }
      }
    }
    return null;
  }

  // ---- JavaTimeRounding (a fixed interval) ----
  long roundKey(long value, long interval) {
    if (value < 0) {
      return (value - interval + 1) / interval;
    } else {
      return value / interval;
    }
  }

  long intervalRound(long utcMillis) {
    final Instant utcInstant = Instant.ofEpochMilli(utcMillis);
    final long localMillis = utcMillis + timeZone.getRules().getOffset(utcInstant).getTotalSeconds() * 1000L;
    final long roundedMillis = roundKey(localMillis, interval) * interval;
    final LocalDateTime roundedLocalDateTime = LocalDateTime.ofInstant(Instant.ofEpochMilli(roundedMillis), ZoneOffset.UTC);
    final List<ZoneOffset> currentOffsets = timeZone.getRules().getValidOffsets(roundedLocalDateTime);
    if (currentOffsets.isEmpty() == false) {
      final ZoneOffsetTransition previousTransition = timeZone.getRules().previousTransition(utcInstant.plusMillis(1));
      for (int offsetIndex = currentOffsets.size() - 1; 0 <= offsetIndex; offsetIndex--) {
        final java.time.OffsetDateTime offsetTime = roundedLocalDateTime.atOffset(currentOffsets.get(offsetIndex));
        final Instant offsetInstant = offsetTime.toInstant();
        if (previousTransition != null && offsetInstant.isBefore(previousTransition.getInstant())) {
          return intervalRound(previousTransition.getInstant().toEpochMilli() - 1);
        }
        if (utcInstant.isBefore(offsetTime.toInstant()) == false) {
          return offsetInstant.toEpochMilli();
        }
      }
      return roundedLocalDateTime.atOffset(currentOffsets.get(0)).toInstant().toEpochMilli();
    } else {
      ZoneOffsetTransition zoneOffsetTransition = timeZone.getRules().getTransition(roundedLocalDateTime);
      return zoneOffsetTransition.getInstant().toEpochMilli();
    }
  }

  long intervalNext(long utcMillis) {
    long prevRound = intervalRound(utcMillis);
    long increment = interval;
    long from = prevRound;
    int iterations = 0;
    while (++iterations < 100) {
      from += increment;
      long rounded = intervalRound(from);
      boolean highEnough = rounded > prevRound;
      if (false == highEnough) {
        if (increment < 0) {
          increment = -increment / 2;
        }
        continue;
      }
      long roundedRoundedDown = intervalRound(rounded - 1);
      boolean tooHigh = roundedRoundedDown > prevRound;
      if (tooHigh) {
        if (increment > 0) {
          increment = -increment / 2;
        }
        continue;
      }
      return rounded;
    }
    return intervalRound(from);
  }

  // ---- OffsetRounding ----
  long round(long utc, long offset) {
    return (unit != null ? unitRound(utc - offset) : intervalRound(utc - offset)) + offset;
  }

  long next(long utc, long offset) {
    return (unit != null ? unitNext(utc - offset) : intervalNext(utc - offset)) + offset;
  }

  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("region_zone_rounding");
    Files.createDirectories(out);
    List<String> rows = new ArrayList<>();
    Random random = new Random(0x5EEDL);
    for (String zone : ZONES) {
      ZoneId zoneId = ZoneId.of(zone);
      ZoneRules rules = zoneId.getRules();
      List<Long> points = new ArrayList<>();
      // Around transitions from 1990 to 2030, and the last ones before.
      long[] around = {-60 * 60_000L, -1L, 0L, 1L, 30 * 60_000L, 90 * 60_000L, 86_400_000L, -86_400_000L, 8 * 86_400_000L};
      Instant t = Instant.parse("1990-01-01T00:00:00Z");
      Instant end = Instant.parse("2030-01-01T00:00:00Z");
      int taken = 0;
      while (true) {
        ZoneOffsetTransition tr = rules.nextTransition(t);
        if (tr == null || tr.getInstant().isAfter(end)) {
          break;
        }
        t = tr.getInstant();
        taken++;
        if (taken % 5 != 0 && !zone.equals("Pacific/Apia") && !zone.equals("Australia/Lord_Howe")) {
          continue;
        }
        for (long d : around) {
          points.add(t.toEpochMilli() + d);
        }
      }
      // Transitions before 1990 (the old rules), and the far past.
      t = Instant.parse("1900-01-01T00:00:00Z");
      for (int i = 0; i < 4; i++) {
        ZoneOffsetTransition tr = rules.nextTransition(t);
        if (tr == null) break;
        t = tr.getInstant();
        for (long d : new long[] {-1L, 0L, 1L, 3_600_000L}) points.add(t.toEpochMilli() + d);
      }
      for (int i = 0; i < 24; i++) {
        points.add(-2_208_988_800_000L + (long) (random.nextDouble() * 6.0e12));
      }
      for (int u = 0; u < UNITS.length + INTERVALS.length; u++) {
        GenRegionZoneRounding r =
            u < UNITS.length
                ? new GenRegionZoneRounding(zoneId, UNITS[u], 0)
                : new GenRegionZoneRounding(zoneId, null, INTERVALS[u - UNITS.length]);
        String kind = u < UNITS.length ? UNITS[u] : "fixed:" + INTERVALS[u - UNITS.length];
        long[] offsets = (kind.equals("day") || kind.equals("hour") || kind.startsWith("fixed:86400000"))
            ? new long[] {0L, 2_700_000L}
            : new long[] {0L};
        for (long offset : offsets) {
          for (long utc : points) {
            try {
              rows.add(zone + " " + kind + " " + offset + " " + utc + " " + r.round(utc, offset) + " " + r.next(utc, offset));
            } catch (RuntimeException e) {
              // The Java class itself fails here (a null truncation before the first transition): no row.
              System.err.println("skipped " + zone + " " + kind + " " + utc + ": " + e);
            }
          }
        }
      }
    }
    Files.write(out.resolve("rounding.txt"), rows);
  }
}
