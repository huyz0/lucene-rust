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
package org.lucenerust.opensearch.knn.script;

import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import org.opensearch.index.mapper.MappedFieldType;
import org.lucenerust.opensearch.knn.KnnVectorFieldMapper.KnnVectorFieldType;

/**
 * A Painless {@code script_score} source made of one k-NN Painless function call ({@code
 * l2Squared}, {@code l1Norm}, {@code lInfNorm}, {@code innerProduct}, {@code cosineSimilarity})
 * inside {@code + - * /}, unary minus, parentheses and number literals, compiled to the typed
 * postfix program the native {@code knn_score} node runs.
 *
 * <p>Types follow Painless (Java) numeric promotion: the function is a {@code float}, an integer
 * literal an {@code int}, a decimal literal a {@code double} (a {@code f} suffix makes it a {@code
 * float}). Parts without the call are folded here, with Java's arithmetic, so the program only
 * holds {@code float} and {@code double} operations. Anything else -- statements, variables,
 * {@code _score}, other functions, a binary field -- is not compiled and the script runs in Java.
 */
public final class PainlessKnnScript {

    /** A compiled script: the native node's field, data type, query and formula. */
    public record Compiled(String field, byte dataType, float[] floats, byte[] bytes, byte function, float norm, byte[] program) {}

    private static final List<String> FUNCTIONS = List.of("l2Squared", "l1Norm", "lInfNorm", "innerProduct", "cosineSimilarity");

    /** The names Painless takes from the search lookup instead of the user's params. */
    private static final List<String> LOOKUP_NAMES = List.of("doc", "_doc", "_source", "_fields");

    private PainlessKnnScript() {}

    /**
     * {@code source} compiled against {@code params} and the index's field types, or null when it
     * is not this shape or Painless would throw on it (the script then runs in Java and throws as
     * stock does).
     */
    public static Compiled compile(String source, Map<String, Object> params, Function<String, MappedFieldType> fieldTypes) {
        try {
            Parser p = new Parser(source, params);
            p.skipSpace();
            if (p.word("return")) {
                p.skipSpace();
            }
            Node root = p.expr();
            p.skipSpace();
            p.eat(';');
            p.skipSpace();
            if (p.pos != source.length() || p.call == null || root.constant != null) {
                return null;
            }
            return p.call.compile(root, fieldTypes);
        } catch (NotThisShape e) {
            return null;
        }
    }

    private static final class NotThisShape extends Exception {
        NotThisShape() {
            super(null, null, false, false);
        }
    }

    private enum Ty {
        INT,
        FLOAT,
        DOUBLE;

        static Ty promote(Ty a, Ty b) {
            return a == DOUBLE || b == DOUBLE ? DOUBLE : a == FLOAT || b == FLOAT ? FLOAT : INT;
        }
    }

    /** An expression: a constant (folded) or an operation over the call. */
    private record Node(Ty ty, Number constant, char op, Node left, Node right) {
        static Node constant(Ty ty, Number value) {
            return new Node(ty, value, ' ', null, null);
        }
    }

    private record Call(byte function, String queryParam, String field, Number norm, Map<String, Object> params) {
        Compiled compile(Node root, Function<String, MappedFieldType> fieldTypes) throws NotThisShape {
            if (fieldTypes == null || (fieldTypes.apply(field) instanceof KnnVectorFieldType) == false) {
                throw new NotThisShape();
            }
            KnnVectorFieldType ft = (KnnVectorFieldType) fieldTypes.apply(field);
            String type = ft.config().dataTypeName();
            if (type.equals("binary")) {
                throw new NotThisShape();
            }
            boolean floatField = type.equals("float") || type.equals("half_float");
            byte dataType = (byte) (type.equals("float") ? 0 : type.equals("half_float") ? 2 : 1);
            if ((params.get(queryParam) instanceof List<?>) == false) {
                throw new NotThisShape();
            }
            List<?> list = (List<?>) params.get(queryParam);
            float[] floats = null;
            byte[] bytes = null;
            float norm = 0;
            try {
                if (function == 5) {
                    // cosineSimilarity with a norm reads the query as floats on any field.
                    if (floatField == false) {
                        throw new NotThisShape();
                    }
                    norm = this.norm.floatValue();
                }
                if (floatField) {
                    floats = KnnScoring.toFloats(list, type);
                    if (function >= 4) {
                        KnnScoring.checkNotZero(floats);
                    }
                } else {
                    bytes = KnnScoring.toBytes(list, type);
                    if (function >= 4) {
                        KnnScoring.checkNotZero(bytes);
                    }
                }
            } catch (RuntimeException e) {
                // Painless throws on every scored document: Java answers.
                throw new NotThisShape();
            }
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            int[] count = { 0, 0, 0 };
            emit(root, root.ty, out, count);
            if (count[0] > MAX_OPS || count[2] > MAX_DEPTH) {
                throw new NotThisShape();
            }
            ByteArrayOutputStream program = new ByteArrayOutputStream();
            writeInt(program, count[0]);
            program.writeBytes(out.toByteArray());
            return new Compiled(field, dataType, floats, bytes, function == 5 ? (byte) 5 : function, norm, program.toByteArray());
        }
    }

    /** The native evaluator's limits on a program's operations and stack depth, as in knn_painless.rs. */
    static final int MAX_OPS = 256;
    static final int MAX_DEPTH = 16;

    /** Pushes one value: counts the operation and tracks the stack depth in {@code count[1]} and its peak in {@code count[2]}. */
    private static void push(int[] count) {
        count[0]++;
        count[1]++;
        count[2] = Math.max(count[2], count[1]);
    }

    /**
     * Appends {@code n}'s operations, leaving a value of type {@code want} (FLOAT or DOUBLE). {@code count} holds the
     * operation count, the current stack depth and the peak depth.
     */
    private static void emit(Node n, Ty want, ByteArrayOutputStream out, int[] count) throws NotThisShape {
        if (n.constant != null) {
            push(count);
            if (want == Ty.FLOAT) {
                // An int or float constant in a float operation.
                if (n.ty == Ty.DOUBLE) {
                    throw new NotThisShape();
                }
                out.write(0);
                writeInt(out, Float.floatToRawIntBits(n.constant.floatValue()));
            } else {
                out.write(1);
                double d = n.ty == Ty.INT ? (double) n.constant.intValue() : n.ty == Ty.FLOAT ? (double) n.constant.floatValue() : n.constant.doubleValue();
                writeLong(out, Double.doubleToRawLongBits(d));
            }
            return;
        }
        if (n.ty == Ty.FLOAT && want == Ty.DOUBLE) {
            emit(n, Ty.FLOAT, out, count);
            count[0]++;
            out.write(3);
            return;
        }
        if (n.ty != want) {
            throw new NotThisShape();
        }
        int base = want == Ty.FLOAT ? 16 : 32;
        switch (n.op) {
            case 'v' -> {
                push(count);
                out.write(2);
            }
            case 'n' -> {
                emit(n.left, want, out, count);
                count[0]++;
                out.write(base + 4);
            }
            default -> {
                emit(n.left, want, out, count);
                emit(n.right, want, out, count);
                count[0]++;
                count[1]--;
                out.write(base + "+-*/".indexOf(n.op));
            }
        }
    }

    private static void writeInt(ByteArrayOutputStream out, int v) {
        out.write(v);
        out.write(v >>> 8);
        out.write(v >>> 16);
        out.write(v >>> 24);
    }

    private static void writeLong(ByteArrayOutputStream out, long v) {
        writeInt(out, (int) v);
        writeInt(out, (int) (v >>> 32));
    }

    private static final class Parser {
        final String s;
        final Map<String, Object> params;
        int pos;
        Call call;

        Parser(String s, Map<String, Object> params) {
            this.s = s;
            this.params = params;
        }

        void skipSpace() {
            while (pos < s.length() && Character.isWhitespace(s.charAt(pos))) {
                pos++;
            }
        }

        boolean peek(char c) {
            skipSpace();
            return pos < s.length() && s.charAt(pos) == c;
        }

        boolean eat(char c) {
            if (peek(c)) {
                pos++;
                return true;
            }
            return false;
        }

        void expect(char c) throws NotThisShape {
            if (eat(c) == false) {
                throw new NotThisShape();
            }
        }

        /** {@code w} as a whole word at the cursor, consumed. */
        boolean word(String w) {
            skipSpace();
            int end = pos + w.length();
            if (s.startsWith(w, pos) && (end == s.length() || Character.isJavaIdentifierPart(s.charAt(end)) == false)) {
                pos = end;
                return true;
            }
            return false;
        }

        String identifier() throws NotThisShape {
            skipSpace();
            int start = pos;
            while (pos < s.length() && Character.isJavaIdentifierPart(s.charAt(pos))) {
                pos++;
            }
            if (pos == start || Character.isJavaIdentifierStart(s.charAt(start)) == false) {
                throw new NotThisShape();
            }
            return s.substring(start, pos);
        }

        Node expr() throws NotThisShape {
            Node n = term();
            while (true) {
                if (eat('+')) {
                    n = binary('+', n, term());
                } else if (eat('-')) {
                    n = binary('-', n, term());
                } else {
                    return n;
                }
            }
        }

        Node term() throws NotThisShape {
            Node n = unary();
            while (true) {
                if (eat('*')) {
                    n = binary('*', n, unary());
                } else if (eat('/')) {
                    if (peek('/') || peek('*')) {
                        // A comment.
                        throw new NotThisShape();
                    }
                    n = binary('/', n, unary());
                } else {
                    return n;
                }
            }
        }

        Node unary() throws NotThisShape {
            if (eat('-')) {
                Node c = unary();
                if (c.constant != null) {
                    return switch (c.ty) {
                        case INT -> Node.constant(Ty.INT, -c.constant.intValue());
                        case FLOAT -> Node.constant(Ty.FLOAT, -c.constant.floatValue());
                        case DOUBLE -> Node.constant(Ty.DOUBLE, -c.constant.doubleValue());
                    };
                }
                return new Node(c.ty, null, 'n', c, null);
            }
            if (eat('+')) {
                return unary();
            }
            return primary();
        }

        Node primary() throws NotThisShape {
            skipSpace();
            if (eat('(')) {
                Node n = expr();
                expect(')');
                return n;
            }
            if (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                return number();
            }
            String name = identifier();
            int function = FUNCTIONS.indexOf(name);
            if (function < 0 || call != null) {
                throw new NotThisShape();
            }
            expect('(');
            String query = paramRef();
            expect(',');
            String field = docRef();
            Number norm = null;
            if (eat(',')) {
                if (function != 4) {
                    throw new NotThisShape();
                }
                skipSpace();
                if (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                    norm = number().constant;
                } else if (params.get(paramRef()) instanceof Number n) {
                    norm = n;
                } else {
                    throw new NotThisShape();
                }
                function = 5;
            }
            expect(')');
            call = new Call((byte) function, query, field, norm, params);
            return new Node(Ty.FLOAT, null, 'v', null, null);
        }

        /**
         * {@code params.x}, {@code params['x']} or {@code params["x"]}: the parameter's name. Painless adds the search
         * lookup's {@code doc}, {@code _doc}, {@code _source} and {@code _fields} over the user's params, and resolves
         * {@code params.empty} to {@code Map.isEmpty()}, so those names are left to Painless.
         */
        String paramRef() throws NotThisShape {
            if (word("params") == false) {
                throw new NotThisShape();
            }
            String name;
            if (eat('.')) {
                name = identifier();
                if (name.equals("empty") || name.equals("class")) {
                    throw new NotThisShape();
                }
            } else {
                expect('[');
                name = quoted();
                expect(']');
            }
            if (LOOKUP_NAMES.contains(name)) {
                throw new NotThisShape();
            }
            return name;
        }

        /** {@code doc['f']}, {@code doc["f"]} or {@code doc[params.x]} with x a string: the field. */
        String docRef() throws NotThisShape {
            if (word("doc") == false) {
                throw new NotThisShape();
            }
            expect('[');
            String field;
            if (peek('\'') || peek('"')) {
                field = quoted();
            } else if (params.get(paramRef()) instanceof String f) {
                field = f;
            } else {
                throw new NotThisShape();
            }
            expect(']');
            return field;
        }

        String quoted() throws NotThisShape {
            skipSpace();
            if (pos >= s.length()) {
                throw new NotThisShape();
            }
            char q = s.charAt(pos);
            if (q != '\'' && q != '"') {
                throw new NotThisShape();
            }
            int end = s.indexOf(q, pos + 1);
            if (end < 0) {
                throw new NotThisShape();
            }
            String v = s.substring(pos + 1, end);
            if (v.indexOf('\\') >= 0) {
                throw new NotThisShape();
            }
            pos = end + 1;
            return v;
        }

        /** A decimal literal: {@code 12} an int, {@code 1.5} or {@code 1e3} a double, a {@code f} suffix a float. */
        Node number() throws NotThisShape {
            int start = pos;
            while (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                pos++;
            }
            boolean decimal = false;
            if (pos < s.length() && s.charAt(pos) == '.' && pos + 1 < s.length() && Character.isDigit(s.charAt(pos + 1))) {
                decimal = true;
                pos++;
                while (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                    pos++;
                }
            }
            if (pos < s.length() && (s.charAt(pos) == 'e' || s.charAt(pos) == 'E')) {
                decimal = true;
                pos++;
                if (pos < s.length() && (s.charAt(pos) == '+' || s.charAt(pos) == '-')) {
                    pos++;
                }
                int digits = pos;
                while (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                    pos++;
                }
                if (pos == digits) {
                    throw new NotThisShape();
                }
            }
            String text = s.substring(start, pos);
            char suffix = pos < s.length() ? s.charAt(pos) : ' ';
            if (Character.isJavaIdentifierPart(suffix) || suffix == '.') {
                if (suffix == 'f' || suffix == 'F') {
                    pos++;
                    if (pos < s.length() && Character.isJavaIdentifierPart(s.charAt(pos))) {
                        throw new NotThisShape();
                    }
                    return Node.constant(Ty.FLOAT, Float.parseFloat(text));
                }
                if (suffix == 'd' || suffix == 'D') {
                    pos++;
                    if (pos < s.length() && Character.isJavaIdentifierPart(s.charAt(pos))) {
                        throw new NotThisShape();
                    }
                    return Node.constant(Ty.DOUBLE, Double.parseDouble(text));
                }
                // Long, hex, octal and other literals Painless reads differently.
                throw new NotThisShape();
            }
            if (decimal) {
                return Node.constant(Ty.DOUBLE, Double.parseDouble(text));
            }
            if (text.length() > 1 && text.charAt(0) == '0') {
                // Octal in Painless.
                throw new NotThisShape();
            }
            try {
                return Node.constant(Ty.INT, Integer.parseInt(text));
            } catch (NumberFormatException e) {
                throw new NotThisShape();
            }
        }

        /** {@code left op right}, folded when both are constants, with Java's arithmetic in their promoted type. */
        Node binary(char op, Node left, Node right) throws NotThisShape {
            Ty ty = Ty.promote(left.ty, right.ty);
            if (left.constant == null || right.constant == null) {
                return new Node(ty, null, op, left, right);
            }
            Number a = left.constant;
            Number b = right.constant;
            return switch (ty) {
                case INT -> {
                    int x = a.intValue();
                    int y = b.intValue();
                    if (op == '/' && y == 0) {
                        // ArithmeticException in Painless.
                        throw new NotThisShape();
                    }
                    yield Node.constant(Ty.INT, switch (op) {
                        case '+' -> x + y;
                        case '-' -> x - y;
                        case '*' -> x * y;
                        default -> x / y;
                    });
                }
                case FLOAT -> {
                    float x = a.floatValue();
                    float y = b.floatValue();
                    yield Node.constant(Ty.FLOAT, switch (op) {
                        case '+' -> x + y;
                        case '-' -> x - y;
                        case '*' -> x * y;
                        default -> x / y;
                    });
                }
                case DOUBLE -> {
                    double x = left.ty == Ty.DOUBLE ? a.doubleValue() : left.ty == Ty.FLOAT ? (double) a.floatValue() : (double) a.intValue();
                    double y = right.ty == Ty.DOUBLE ? b.doubleValue() : right.ty == Ty.FLOAT ? (double) b.floatValue() : (double) b.intValue();
                    yield Node.constant(Ty.DOUBLE, switch (op) {
                        case '+' -> x + y;
                        case '-' -> x - y;
                        case '*' -> x * y;
                        default -> x / y;
                    });
                }
            };
        }
    }
}
