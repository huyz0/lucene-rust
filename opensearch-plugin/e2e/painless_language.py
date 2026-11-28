#!/usr/bin/env python3
"""Painless language and API corpus, answered by a plugin node and a stock node: each snippet runs as a
sort script (a native context) over a few documents and its value (or error type) must be identical.

    python3 opensearch-plugin/e2e/painless_language.py [--plugin URL] [--stock URL] [--index fz]

The snippets are typed arithmetic and casts, numeric printing, strings, regular expressions, collections
(and the order a HashMap or HashSet iterates in), lambdas, streams, Optional, StringBuilder, Character and
boxed types, exceptions, loops and `def` dispatch: the language surface the fuzzers do not generate. The
nodes need `script.disable_max_compilations_rate=true`. The index needs the documents `fuzz_diff.py --queries 0`
builds."""
import argparse
import json
import sys
import urllib.error
import urllib.request

SNIPPETS = r"""
return 7 / 2;
return -7 / 2;
return -7 % 3;
return 7 % -3;
return 7.5 % 2;
return -7.5 % 2;
return 1 / 0.0;
return -1 / 0.0;
return 0.0 / 0.0;
return Integer.MAX_VALUE + 1;
return Integer.MIN_VALUE - 1;
return Long.MAX_VALUE + 1L;
return Math.abs(Integer.MIN_VALUE);
return Math.abs(Long.MIN_VALUE);
return (int) 3.99e10;
return (int) -3.99e10;
return (long) 1e30;
return (int) Double.NaN;
return (byte) 200;
return (short) 70000;
return (char) 65;
return (int) 'a' + 1;
return 'a' + 1;
return (char) ('a' + 1);
return 1 << 33;
return 1L << 65;
return -16 >> 2;
return -16 >>> 28;
return -16L >>> 60;
return 5 & 3 | 8 ^ 2;
return ~5;
return 0.1 + 0.2;
return 0.1f + 0.2f;
return (float) 0.1 + 0.2;
return 100.0 / 3;
return 100.0f / 3;
return 1e7;
return 1.0e-5;
return 123456789.0 * 1000;
return 1.0E10f;
return (double) Float.MAX_VALUE;
return Double.MIN_VALUE;
return -0.0;
return 0.0 == -0.0;
return Double.compare(0.0, -0.0);
return Math.round(-2.5);
return Math.round(2.5);
return Math.round(-0.5);
return Math.round(2.5f);
return Math.rint(2.5);
return Math.floor(-1.5);
return Math.ceil(-1.5);
return Math.floorMod(-7, 3);
return Math.floorDiv(-7, 3);
return Math.pow(2, 0.5);
return Math.pow(-8, 1.0/3);
return Math.sqrt(-1);
return Math.cbrt(-27);
return Math.hypot(3, 4);
return Math.log10(1000);
return Math.atan2(1, 1);
return Math.signum(-3.2);
return Math.toDegrees(Math.PI);
return Math.max(1, 2.5);
return Math.min(-0.0, 0.0);
return Math.max(Long.MAX_VALUE, 1);
return Math.addExact(Integer.MAX_VALUE, 1);
return Math.toIntExact(1L << 40);
return Integer.parseInt("-123");
return Integer.parseInt("12x");
return Long.parseLong("9223372036854775808");
return Double.parseDouble("1e3");
return Double.parseDouble("abc");
return Integer.toBinaryString(10);
return Integer.toHexString(-1);
return Long.toString(255, 16);
return Integer.valueOf(127) == Integer.valueOf(127);
return Integer.valueOf(1000) == Integer.valueOf(1000);
return Integer.valueOf(1000).equals(Integer.valueOf(1000));
return Integer.compare(3, 5);
return Integer.bitCount(255);
return Long.numberOfTrailingZeros(1024L);
return Integer.reverse(1);
return Integer.MAX_VALUE;
return Double.toString(1.0E-4);
return Float.toString(1.1f);
return String.valueOf(1.0f / 3);
return String.valueOf((double) (1.0f / 3));
return Boolean.parseBoolean("TRUE");
return Character.isDigit('5');
return Character.toUpperCase('ä');
return Character.getNumericValue('z');
return (int) Character.forDigit(11, 16);
return "abc".charAt(1);
return "abc".charAt(5);
return "héllo".length();
return "😀x".length();
return "😀x".codePointAt(0);
return "abc".substring(1);
return "abc".substring(2, 1);
return "abc".indexOf("c", -5);
return "abcabc".lastIndexOf("b");
return "abc".compareTo("abd");
return "ABC".compareToIgnoreCase("abd");
return "abc".equalsIgnoreCase("ABC");
return "a,b,,c,,".split(",").length;
return "a,b,,c,,".splitOnToken(",").length;
return Arrays.toString("a1b22c".split("[0-9]+"));
return "a b  c".trim().replace(" ", "_");
return "  x ".strip() + "|";
return "abc".repeat(3);
return "Hello".toUpperCase().toLowerCase();
return "İ".toLowerCase().length();
return "straße".toUpperCase();
return "abc".contains("");
return "abc".startsWith("bc", 1);
return "abc".endsWith("");
return "a-b".replace('-', '+');
return "aaa".replaceAll("a*", "X");
return "aaa".replaceFirst("a", "$0$0");
return "abc".replaceAll("(b)", "[$1]");
return "x".concat("y");
return "abc".isEmpty();
return "".isEmpty();
return "abc".hashCode();
return "".hashCode();
return "Hello World".hashCode();
return "abc".toCharArray().length;
return new String(new char[] {'a', 'b'});
return "a\tb".length();
return "%d items".replace("%d", "3");
return String.format("%5.2f|%-5d|%05d|%x|%s", new def[] {3.14159, 42, 42, 255, "z"});
return String.format("%,d", new def[] {1234567});
return String.format("%e", new def[] {12345.678});
return String.format("%10s|%-10s|", new def[] {"a", "b"});
return String.format("%b %c %%", new def[] {true, 'x'});
return String.format("%.0f", new def[] {2.5});
return String.format("%.0f", new def[] {3.5});
return String.format("%08.3f", new def[] {-3.14159});
return String.format("%d", new def[] {"x"});
return String.join(",", new String[] {"a", "b"});
return String.join("-", List.of("a", "b", "c"));
return /a(b+)c/.matcher("xabbbcx").find();
return /a(b+)c/.matcher("xabbbcx").group(1);
return /a(b+)c/.matcher("xabbbcx").matches();
return "xabbbcx" ==~ /.*b+.*/;
return "xabbbcx" =~ /b+/;
return /(\d+)-(\d+)/.matcher("10-20").replaceAll("$2-$1");
return /(?<y>\d{4})-(?<m>\d\d)/.matcher("2024-05").replaceAll("${m}/${y}");
return /\bfoo\b/i.matcher("a FOO b").find();
return /[\p{L}]+/.matcher("héllo wörld").replaceAll("*");
return /a.c/s.matcher("a\nc").matches();
return /^b/m.matcher("a\nb").find();
return Pattern.compile("a|b").split("xaybz").length;
return Pattern.quote("a.b");
return Pattern.matches("a+", "aaa");
return /x*/.matcher("aaa").replaceAll("-");
return /(a)|(b)/.matcher("b").matches() ? 1 : 0;
def m = /(a)|(b)/.matcher("b"); m.matches(); return m.group(1);
def m = /(\w)(\d)?/.matcher("a"); m.matches(); return m.groupCount() + ":" + m.group(2);
def m = /\d+/.matcher("a1b22c333"); int c = 0; while (m.find()) { c += m.group().length(); } return c;
def m = /\d+/.matcher("a1b22c333"); m.find(); m.find(); return m.start() + "," + m.end();
List l = new ArrayList(); l.add(3); l.add(1); l.add(2); Collections.sort(l); return l;
List l = [3, 1, 2]; l.sort((a, b) -> b - a); return l;
List l = ["bb", "a", "ccc"]; l.sort((a, b) -> a.length() - b.length()); return l;
List l = [1, 2, 3, 4]; l.removeIf(x -> x % 2 == 0); return l;
List l = [1, 2, 3]; l.replaceAll(x -> x * x); return l;
List l = [1, 2, 3]; return l.stream().mapToInt(x -> x).sum();
List l = [1, 2, 3, 4]; return l.stream().filter(x -> x > 1).map(x -> x * 2).collect(Collectors.toList());
List l = [1, 2, 3, 4]; return l.stream().mapToDouble(x -> x).average().getAsDouble();
List l = []; return l.stream().mapToInt(x -> x).max().isPresent();
List l = ["x", "y"]; return l.stream().collect(Collectors.joining(","));
List l = [5, 3, 9]; return l.stream().sorted().collect(Collectors.toList());
List l = [5, 3, 9]; return l.stream().mapToInt(x -> x).boxed().collect(Collectors.toList());
List l = [1, 2, 2, 3]; return l.stream().distinct().count();
List l = [1, 2, 3]; return l.stream().anyMatch(x -> x > 2);
List l = [1, 2, 3]; return l.stream().reduce(0, (a, b) -> a + b);
List l = [1, 2, 3]; return l.stream().map(x -> x + "").collect(Collectors.joining("|", "<", ">"));
List l = [1, 2, 3]; l.add(1, 9); return l;
List l = [1, 2, 3]; l.remove(1); return l;
List l = [1, 2, 3]; l.remove(Integer.valueOf(3)); return l;
List l = [1, 2, 3]; return l.indexOf(4);
List l = [1, 2, 3]; return l.subList(1, 3);
List l = [1, 2, 3]; return l.get(3);
List l = [1, 2, 3]; return l.contains(2.0);
List l = [1, 2, 3]; return l.equals([1, 2, 3]);
List l = [1, 2, 3]; return l.hashCode();
List l = [[1, 2], [3]]; return l.toString();
List l = [null, 1]; return l.toString();
List l = new LinkedList(); l.add(1); l.add(0, 2); return l;
List l = List.of(1, 2); l.add(3); return l;
List l = [1, 2, 3]; Collections.reverse(l); return l;
List l = [1, 2, 3]; return Collections.max(l) + Collections.min(l);
List l = [3, 1, 2]; Collections.sort(l, (a, b) -> b - a); return l;
Map m = new HashMap(); m.put("b", 1); m.put("a", 2); m.put("c", 3); return m;
Map m = new HashMap(); for (int i = 0; i < 20; i++) { m.put("k" + i, i); } return m.keySet().toString();
Map m = new HashMap(); for (int i = 0; i < 40; i++) { m.put(i * 37, i); } return m.keySet().toString();
Map m = new HashMap(); for (int i = 0; i < 12; i++) { m.put("key" + i * 7, i); } return m.values().toString();
Map m = new HashMap(); m.put(1.5, 1); m.put(2L, 2); m.put('c', 3); m.put("s", 4); return m.keySet().toString();
Map m = new LinkedHashMap(); m.put("b", 1); m.put("a", 2); return m;
Map m = new TreeMap(); m.put("b", 1); m.put("a", 2); return m;
Map m = new HashMap(); m.put("a", 1); m.putIfAbsent("a", 2); m.putIfAbsent("b", 3); return m;
Map m = new HashMap(); m.merge("a", 1, (x, y) -> x + y); m.merge("a", 1, (x, y) -> x + y); return m;
Map m = new HashMap(); m.computeIfAbsent("a", k -> k + "!"); return m;
Map m = new HashMap(); m.put("a", 1); m.compute("a", (k, v) -> v == null ? 0 : v + 10); return m;
Map m = new HashMap(); m.put("a", 1); return m.getOrDefault("z", 5);
Map m = new HashMap(); m.put("a", 1); m.put("b", 2); int s = 0; for (def e : m.entrySet()) { s += e.getValue(); } return s;
Map m = new HashMap(); m.put("a", 1); m.put("b", 2); m.forEach((k, v) -> m.put(k, v * 2)); return m;
Map m = new HashMap(); m.put(null, 1); return m.get(null);
Map m = Map.of("a", 1); return m.get("a");
Map m = new HashMap(); m.put("a", 1); m.remove("a"); return m.isEmpty();
Map m = new HashMap(); m.put("a", [1, 2]); return m.toString();
Map m = ["x": 1, "y": 2]; return m.size();
Map m = [:]; return m.size();
Set s = new HashSet(); s.add("b"); s.add("a"); s.add("c"); return s;
Set s = new HashSet(); for (int i = 0; i < 30; i++) { s.add("v" + i); } return s.toString();
Set s = new HashSet(); for (int i = 0; i < 30; i++) { s.add(i * 101); } return s.toString();
Set s = new TreeSet(); s.add("b"); s.add("a"); return s;
Set s = new HashSet(); s.add(1); s.add(1L); return s.size();
Set s = new HashSet([1, 2]); s.retainAll([2, 3]); return s;
Deque d = new ArrayDeque(); d.push(1); d.push(2); return d.pop();
Queue q = new PriorityQueue(); q.add(3); q.add(1); q.add(2); return q.poll();
Iterator it = [1, 2].iterator(); it.next(); return it.hasNext();
int[] a = new int[] {3, 1, 2}; Arrays.sort(a); return a;
int[] a = {1, 2, 3}; return a.length;
int[][] a = new int[2][3]; return a[1].length;
int[] a = new int[2]; return a[2];
long[] a = {1L, 2L}; return Arrays.stream(a).sum();
double[] a = {1.5, 2.5}; return Arrays.stream(a).max().getAsDouble();
String[] a = {"b", "a"}; Arrays.sort(a); return Arrays.toString(a);
def a = new ArrayList(); a.add(1); return a.size();
def x = 5; x += 2.5; return x;
def x = 5; x /= 2; return x;
def x = 5L; x <<= 2; return x;
byte b = 10; b += 300; return b;
short s = 1; s *= 40000; return s;
char c = 'a'; c += 1; return c;
char c = 'a'; c++; return c;
int i = 5; i += 3.9; return i;
int i = 5; i -= 0.5; return i;
long l = 1; l <<= 40; return l;
int i = 2147483647; i++; return i;
int i = 0; i = i++ + ++i; return i;
int i = 3; return i-- - --i;
float f = 1.1f; f *= 2; return f;
float f = 16777216f; f++; return f;
double d = 0; d += 1 / 3; return d;
double d = 1; d /= 0; return d;
int x = 5; return x > 3 ? 1 : 2.0;
return true ? 1 : 2.0;
return true ? 'a' : 0;
return false ? 1 : 'b';
def a = null; return a == null;
def a = null; return a?.length();
String s = null; return s?.length();
String s = null; return s + "x";
String s = null; return s.length();
return null == null;
return 1 == 1.0;
return 'a' == 97;
return "a" == "a";
return new String("a") == "a";
return new String("a").equals("a");
return 10000000000L == 1e10;
return 16777217 == 16777217f;
return Integer.valueOf(5).equals(5L);
return (Object) 5 instanceof Integer;
return (Object) "s" instanceof CharSequence;
def x = 5; return x instanceof int;
def x = [1]; return x instanceof List;
Object o = 5; return o.getClass().getName();
Object o = 5L; return ((Long) o).intValue();
Object o = "s"; return (Integer) o;
long x = (long) 1e19; return x;
double d = 1e308 * 10; return d;
return Float.MAX_VALUE * 2;
return (float) 1e40;
return 5 / (1 - 1);
return 5L % 0;
return 5.0 % 0;
int s = 0; for (int i = 0; i < 10; i++) { if (i % 2 == 0) continue; if (i > 7) break; s += i; } return s;
int s = 0; int i = 0; while (true) { if (++i > 5) break; s += i; } return s;
int s = 0; int i = 0; do { s += i; i++; } while (i < 4); return s;
int s = 0; for (def x : [1, 2, 3]) { s += x; } return s;
int s = 0; for (int x : new int[] {1, 2, 3}) { s += x; } return s;
int s = 0; for (int i = 0, j = 10; i < j; i += 3, j -= 3) { s += i * j; } return s;
int f(int n) { return n <= 1 ? 1 : n * f(n - 1); } return f(10);
long fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); } return fib(15);
String g(def x) { return x + "!"; } return g(1) + g("a");
int f(int n) { return f(n + 1); } return f(0);
try { return 1 / 0; } catch (ArithmeticException e) { return e.getMessage(); }
try { int[] a = new int[1]; return a[2]; } catch (ArrayIndexOutOfBoundsException e) { return e.getMessage(); }
try { return Integer.parseInt("x"); } catch (NumberFormatException e) { return e.getMessage(); }
try { String s = null; return s.length(); } catch (NullPointerException e) { return "npe"; }
try { return "abc".substring(5); } catch (StringIndexOutOfBoundsException e) { return e.getMessage(); }
try { return [1].get(2); } catch (IndexOutOfBoundsException e) { return e.getMessage(); }
try { throw new IllegalArgumentException("boom"); } catch (IllegalArgumentException e) { return e.getMessage(); } finally { int x = 1; }
try { Object o = "s"; return (Integer) o; } catch (ClassCastException e) { return "cce"; }
try { return Math.toIntExact(1L << 40); } catch (ArithmeticException e) { return e.getMessage(); }
try { return new ArrayList().iterator().next(); } catch (NoSuchElementException e) { return "nse"; }
try { List l = List.of(1); l.add(2); return 1; } catch (UnsupportedOperationException e) { return "uoe"; }
try { return Optional.empty().get(); } catch (NoSuchElementException e) { return e.getMessage(); }
throw new RuntimeException("x");
throw new IllegalStateException();
Optional o = Optional.of(5); return o.map(x -> x + 1).get();
Optional o = Optional.empty(); return o.orElse(7);
Optional o = Optional.ofNullable(null); return o.isPresent();
Optional o = Optional.of(5); return o.filter(x -> x > 9).isPresent();
OptionalInt o = OptionalInt.of(3); return o.getAsInt();
StringBuilder sb = new StringBuilder(); sb.append(1).append('a').append("b").append(1.5).append(true).append((Object) null); return sb.toString();
StringBuilder sb = new StringBuilder("abc"); sb.insert(1, "X").reverse().setLength(2); return sb.toString();
StringBuilder sb = new StringBuilder("abc"); sb.deleteCharAt(1); sb.setCharAt(0, 'z'); return sb;
StringBuilder sb = new StringBuilder("abc"); return sb.indexOf("c") + sb.length();
StringBuilder sb = new StringBuilder("abc"); sb.replace(0, 2, "Q"); return sb.toString();
Locale l = Locale.ROOT; return "TITLE".toLowerCase(l);
return "i".toUpperCase(Locale.forLanguageTag("tr"));
return "I".toLowerCase(Locale.forLanguageTag("tr"));
return String.format(Locale.ROOT, "%,.2f", new def[] {1234.5});
return Objects.equals(null, null);
return Objects.hash(1, "a", 2.5);
return Objects.toString(null);
return Objects.requireNonNull("x");
return Objects.isNull(null);
return Arrays.asList(1, 2).size();
return List.of(1, 2, 3).stream().mapToInt(x -> x).summaryStatistics().getMax();
return new BigDecimal("1.10").add(new BigDecimal("2.205")).toString();
return Instant.ofEpochMilli(1700000000000L).toString();
return Instant.ofEpochMilli(1700000000123L).atZone(ZoneId.of("UTC")).getYear();
return ZonedDateTime.ofInstant(Instant.ofEpochMilli(0), ZoneId.of("UTC")).getDayOfWeek().toString();
return Instant.parse("2024-02-29T12:30:00Z").toEpochMilli();
return Instant.parse("2024-02-30T00:00:00Z");
return LocalDate.of(2024, 2, 29).plusYears(1).toString();
return Duration.ofSeconds(3661).toMinutes();
return ChronoUnit.DAYS.between(LocalDate.of(2024, 1, 1), LocalDate.of(2024, 3, 1));
return DateTimeFormatter.ISO_LOCAL_DATE.format(LocalDate.of(2024, 12, 5));
return ZonedDateTime.of(2024, 3, 10, 1, 30, 0, 0, ZoneId.of("America/New_York")).plusHours(1).toString();
return Random.class.getName();
return UUID.randomUUID().toString().length();
return doc['n'].size();
return doc['n'].size() == 0 ? -1 : doc['n'].value;
return doc['tag'].size() == 0 ? "none" : doc['tag'].value;
return doc['tag'].size() == 0 ? "none" : doc['tag'].value.toUpperCase() + doc['tag'].value.length();
return doc['price'].size() == 0 ? -1.0 : doc['price'].value * 3;
return doc['price'].size() == 0 ? -1.0 : (long) (doc['price'].value * 100) % 7;
return doc['f'].size() == 0 ? -1 : doc['f'].value;
return doc['f'].size() == 0 ? -1 : doc['f'].value + 0.1;
return doc['f'].size() == 0 ? -1 : (double) doc['f'].value;
return doc['f'].size() == 0 ? -1 : doc['f'].value / 3;
return doc['i'].size() == 0 ? -1 : doc['i'].value / 3;
return doc['n'].size() == 0 ? -1 : doc['n'].value / 3;
return doc['n'].size() == 0 ? -1 : doc['n'].value % 7;
return doc['n'].size() == 0 ? -1 : doc['n'].value * doc['n'].value * doc['n'].value;
return doc['n'].size() == 0 ? -1 : (int) doc['n'].value;
return doc['n'].size() == 0 ? -1 : doc['n'].value << 20;
return doc['n'].size() == 0 ? -1 : Long.toBinaryString(doc['n'].value);
return doc['tag'].size() == 0 ? -1 : doc['tag'].values.size();
return doc['tag'].size() == 0 ? "" : doc['tag'].value.substring(0, 1) + doc['tag'].value.charAt(0);
return doc.containsKey('n');
return doc.containsKey('nope');
return doc['nope'].size();
return doc['body'].size();
return params.x;
return params.get("x");
return params._source.n;
return _score;
return state;
return ctx;
return doc['ts'].size() == 0 ? -1 : doc['ts'].value.getYear();
return doc['ts'].size() == 0 ? -1 : doc['ts'].value.toInstant().toEpochMilli();
return doc['ts'].size() == 0 ? "" : doc['ts'].value.toString();
return doc['ts'].size() == 0 ? -1 : doc['ts'].value.getDayOfWeek().getValue();
return doc['ts'].size() == 0 ? -1 : doc['ts'].value.withZoneSameInstant(ZoneId.of("Asia/Tokyo")).getHour();
"""


def req(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    r = urllib.request.Request(base + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(r) as f:
            return f.status, json.load(f)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except ValueError:
            return e.code, {}


def wrap(source):
    """The snippet as a function, its value as a number: the hash of its text form, so a sort script (native)
    carries any result type and a difference in how a value prints shows."""
    return "def snippet() { %s } def v = snippet(); return v == null ? 0L : (long) String.valueOf(v).hashCode() * 31L + String.valueOf(v).length();" % source


def run(base, index, source):
    body = {"size": 4, "sort": [{"_script": {"type": "number", "order": "asc", "script": {"source": wrap(source), "params": {"x": 3}}}}, {"_id": "asc"}],
            "query": {"match_all": {}}, "_source": False}
    st, r = req(base, "POST", "/%s/_search" % index, body)
    if st != 200:
        err = r.get("error", {})
        shards = err.get("failed_shards") or [{}]
        reason = shards[0].get("reason") or err
        cause = reason.get("caused_by") or {}
        while cause.get("caused_by"):
            cause = cause["caused_by"]
        return ("error", reason.get("type"), cause.get("type") or reason.get("type"))
    return ("ok", [(h["_id"], h["sort"]) for h in r["hits"]["hits"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="http://localhost:9250")
    ap.add_argument("--stock", default="http://localhost:9251")
    ap.add_argument("--index", default="fz")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    snippets = [s.strip() for s in SNIPPETS.strip().splitlines() if s.strip() and args.only in s]
    bad = both_error = 0
    def fallbacks():
        st, stats = req(args.plugin, "GET", "/_plugins/accelerated/stats")
        return sum(stats.get("fallbacks", {}).values()), stats.get("script_errors", 0)

    native = 0
    for s in snippets:
        before = fallbacks()
        a = run(args.plugin, args.index, s)
        after = fallbacks()
        b = run(args.stock, args.index, s)
        if after[0] == before[0]:
            native += 1
        if a != b:
            bad += 1
            print("DIFF", "(native)" if after[0] == before[0] else "(fallback)", s, "\n   plugin:", str(a)[:300], "\n   stock: ", str(b)[:300])
        elif a[0] == "error":
            both_error += 1
    print("%d snippets (%d ran natively), %d differ, %d error on both" % (len(snippets), native, bad, both_error))
    sys.exit(1 if bad else 0)


main()
