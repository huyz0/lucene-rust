/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.lucenerust.opensearch.painless.PainlessIr;
import org.opensearch.common.settings.Settings;
import org.opensearch.painless.CompilerSettings;
import org.opensearch.painless.PainlessScriptEngine;
import org.opensearch.painless.action.PainlessExecuteAction.PainlessTestScript;
import org.opensearch.painless.lookup.PainlessLookup;
import org.opensearch.painless.spi.Allowlist;
import org.opensearch.script.ScriptException;

/**
 * Writes the differential corpus {@code lucene-painless} tests run: each script of Painless's
 * own test suite (extract.py's JSON lines), run by Painless and compiled to native IR, with
 * what Painless returned or threw.
 *
 * <p>Record layout (little-endian): source string, status {@code u8} ({@code 0} returned, then
 * the value; {@code 1} threw, then the exception class, its message ({@code u8} 1 then the string, or 0 for
 * none) and the script offset Painless points at ({@code i32}, -1 none); {@code 2} did not compile;
 * {@code 3} compiled but the IR layout refused it, then the reason), then for status 0 and 1
 * the IR as a length-prefixed blob.
 *
 * <pre>
 *   java -cp ... Corpus scripts.jsonl corpus.bin
 * </pre>
 */
public class Corpus {

    public static void main(String[] args) throws Exception {
        Map<org.opensearch.script.ScriptContext<?>, List<Allowlist>> contexts = new HashMap<>();
        contexts.put(PainlessTestScript.CONTEXT, Allowlist.BASE_ALLOWLISTS);
        PainlessScriptEngine engine = new PainlessScriptEngine(Settings.EMPTY, contexts);
        PainlessLookup lookup = engine.getContextsToLookups().get(PainlessTestScript.CONTEXT);
        Pattern srcPat = Pattern.compile("\"src\": (\"(?:[^\"\\\\]|\\\\.)*\")");
        int[] counts = new int[4];
        try (BufferedReader r = Files.newBufferedReader(Path.of(args[0]));
             DataOutputStream out = new DataOutputStream(new FileOutputStream(args[1]))) {
            out.write("PCO2".getBytes(StandardCharsets.US_ASCII));
            String line;
            while ((line = r.readLine()) != null) {
                Matcher m = srcPat.matcher(line);
                if (m.find() == false) {
                    continue;
                }
                String src = unjson(m.group(1));
                Rec rec = new Rec();
                rec.str(src);
                PainlessTestScript.Factory factory;
                try {
                    factory = engine.compile("corpus", src, PainlessTestScript.CONTEXT, Map.of());
                } catch (Throwable t) {
                    rec.u8(2);
                    counts[2]++;
                    rec.flush(out);
                    continue;
                }
                byte[] ir;
                try {
                    CompilerSettings settings = new CompilerSettings();
                    settings.setRegexesEnabled(CompilerSettings.RegexEnabled.TRUE);
                    ir = PainlessIr.compile(lookup, PainlessTestScript.class, src, settings).ir();
                } catch (PainlessIr.Unsupported u) {
                    rec.u8(3);
                    rec.str(u.getMessage());
                    counts[3]++;
                    rec.flush(out);
                    continue;
                }
                Object result;
                try {
                    result = factory.newInstance(new HashMap<>()).execute();
                    rec.u8(0);
                    rec.value(result, 0);
                    counts[0]++;
                } catch (Throwable t) {
                    Throwable cause = t instanceof ScriptException && t.getCause() != null ? t.getCause() : t;
                    ScriptException.Position pos = t instanceof ScriptException se ? se.getPos() : null;
                    rec.u8(1);
                    rec.str(cause.getClass().getName());
                    if (cause.getMessage() == null) {
                        rec.u8(0);
                    } else {
                        rec.u8(1);
                        rec.str(cause.getMessage());
                    }
                    rec.i32(pos == null ? -1 : pos.offset);
                    counts[1]++;
                }
                rec.blob(ir);
                rec.flush(out);
            }
        }
        System.err.printf("returned %d, threw %d, no compile %d, layout refused %d%n", counts[0], counts[1], counts[2], counts[3]);
    }

    static String unjson(String q) {
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i < q.length() - 1; i++) {
            char c = q.charAt(i);
            if (c != '\\') {
                sb.append(c);
                continue;
            }
            char e = q.charAt(++i);
            switch (e) {
                case 'n' -> sb.append('\n');
                case 't' -> sb.append('\t');
                case 'r' -> sb.append('\r');
                case 'b' -> sb.append('\b');
                case 'f' -> sb.append('\f');
                case 'u' -> {
                    sb.append((char) Integer.parseInt(q.substring(i + 1, i + 5), 16));
                    i += 4;
                }
                default -> sb.append(e);
            }
        }
        return sb.toString();
    }

    static final class Rec {
        final ByteArrayOutputStream buf = new ByteArrayOutputStream();

        void u8(int v) {
            buf.write(v);
        }

        void i32(int v) {
            buf.write(v);
            buf.write(v >>> 8);
            buf.write(v >>> 16);
            buf.write(v >>> 24);
        }

        void i64(long v) {
            i32((int) v);
            i32((int) (v >>> 32));
        }

        void str(String s) {
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            i32(b.length);
            buf.write(b, 0, b.length);
        }

        void blob(byte[] b) {
            i32(b.length);
            buf.write(b, 0, b.length);
        }

        /** A returned value: the runtime's `Value` encoding (tag, then payload). */
        void value(Object v, int depth) {
            if (depth > 32) {
                u8(14);
                str("deep");
                str("");
                return;
            }
            if (v == null) {
                u8(0);
            } else if (v instanceof Boolean b) {
                u8(1);
                u8(b ? 1 : 0);
            } else if (v instanceof Byte b) {
                u8(2);
                i32(b);
            } else if (v instanceof Short s) {
                u8(3);
                i32(s);
            } else if (v instanceof Character c) {
                u8(4);
                i32(c);
            } else if (v instanceof Integer i) {
                u8(5);
                i32(i);
            } else if (v instanceof Long l) {
                u8(6);
                i64(l);
            } else if (v instanceof Float f) {
                u8(7);
                i32(Float.floatToRawIntBits(f));
            } else if (v instanceof Double d) {
                u8(8);
                i64(Double.doubleToRawLongBits(d));
            } else if (v instanceof String s) {
                u8(9);
                str(s);
            } else if (v instanceof List<?> l) {
                u8(11);
                str(v.getClass().getName());
                i32(l.size());
                for (Object o : l) {
                    value(o, depth + 1);
                }
            } else if (v instanceof Map<?, ?> m) {
                u8(12);
                str(v.getClass().getName());
                i32(m.size());
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    value(e.getKey(), depth + 1);
                    value(e.getValue(), depth + 1);
                }
            } else if (v instanceof Collection<?> c) {
                u8(15);
                str(v.getClass().getName());
                i32(c.size());
                for (Object o : c) {
                    value(o, depth + 1);
                }
            } else if (v.getClass().isArray()) {
                u8(13);
                str(v.getClass().getName());
                int n = Array.getLength(v);
                i32(n);
                for (int i = 0; i < n; i++) {
                    value(Array.get(v, i), depth + 1);
                }
            } else {
                u8(14);
                str(v.getClass().getName());
                str(String.valueOf(v));
            }
        }

        void flush(DataOutputStream out) throws IOException {
            buf.writeTo(out);
        }
    }
}
