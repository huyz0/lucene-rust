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
package org.lucenerust.opensearch.painless;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

import org.opensearch.painless.CompilerSettings;
import org.opensearch.painless.Operation;
import org.opensearch.painless.ScriptClassInfo;
import org.opensearch.painless.antlr.Walker;
import org.opensearch.painless.ir.*;
import org.opensearch.painless.lookup.PainlessCast;
import org.opensearch.painless.lookup.PainlessClassBinding;
import org.opensearch.painless.lookup.PainlessConstructor;
import org.opensearch.painless.lookup.PainlessField;
import org.opensearch.painless.lookup.PainlessLookup;
import org.opensearch.painless.lookup.PainlessMethod;
import org.opensearch.painless.node.SClass;
import org.opensearch.painless.phase.DefaultConstantFoldingOptimizationPhase;
import org.opensearch.painless.phase.DefaultStringConcatenationOptimizationPhase;
import org.opensearch.painless.phase.DocFieldsPhase;
import org.opensearch.painless.phase.IRTreeVisitor;
import org.opensearch.painless.phase.PainlessSemanticAnalysisPhase;
import org.opensearch.painless.phase.PainlessSemanticHeaderPhase;
import org.opensearch.painless.phase.PainlessUserTreeToIRTreePhase;
import org.opensearch.painless.symbol.Decorations.IRNodeDecoration;
import org.opensearch.painless.symbol.FunctionTable.LocalFunction;
import org.opensearch.painless.symbol.ScriptScope;

/**
 * A Painless script compiled by Painless's own front end -- the phases {@code Compiler.compile}
 * runs, up to constant folding -- and written as its IR tree, for the native runtime
 * ({@code lucene-painless}) to execute instead of the class Painless would generate.
 *
 * <p>The IR keeps everything Painless resolved: every expression's type, the casts, the
 * allowlisted method each call reaches, {@code def} sites and the doc fields the script reads.
 * The layout ({@code PIR1}) is the Rust decoder's ({@code lucene_painless::ir}): little-endian,
 * a string is an {@code i32} byte length ({@code -1} for null) and UTF-8, a type is its JVM
 * binary name as a string, and each node is its tag byte, its fields, then its children.
 */
public final class PainlessIr {

    /** A script this layout cannot carry; the message is the fallback reason's detail. */
    public static final class Unsupported extends Exception {
        public Unsupported(String what) {
            super(what, null, false, false);
        }
    }

    /** The compiled script: its IR, and the doc fields it reads by a constant name. */
    public record Compiled(byte[] ir, List<String> docFields) {}

    private PainlessIr() {}

    /**
     * {@code source} compiled for {@code scriptClass} (the context's instance class) with the
     * context's lookup and {@code settings}.
     *
     * @throws Unsupported when the IR holds something the layout has no form for
     * @throws RuntimeException whatever Painless's front end throws on the source
     */
    public static Compiled compile(PainlessLookup lookup, Class<?> scriptClass, String source, CompilerSettings settings)
        throws Unsupported {
        ScriptClassInfo info = new ScriptClassInfo(lookup, scriptClass);
        SClass root = Walker.buildPainlessTree("native", source, settings);
        ScriptScope scope = new ScriptScope(lookup, settings, info, "native", source, root.getIdentifier() + 1);
        new PainlessSemanticHeaderPhase().visitClass(root, scope);
        new PainlessSemanticAnalysisPhase().visitClass(root, scope);
        new DocFieldsPhase().visitClass(root, scope);
        new PainlessUserTreeToIRTreePhase().visitClass(root, scope);
        ClassNode classNode = (ClassNode) scope.getDecoration(root, IRNodeDecoration.class).getIRNode();
        new DefaultStringConcatenationOptimizationPhase().visitClass(classNode, null);
        new DefaultConstantFoldingOptimizationPhase().visitClass(classNode, null);
        Writer w = new Writer(scope.getStaticConstants());
        w.script(classNode);
        if (w.unsupported != null) {
            throw new Unsupported(w.unsupported);
        }
        return new Compiled(w.out.toByteArray(), scope.docFields());
    }

    // Node tags: the Rust decoder's `Tag`.
    static final byte BLOCK = 1, DECLARATION_BLOCK = 2, DECLARATION = 3, STATEMENT_EXPRESSION = 4, RETURN = 5, IF = 6,
        IF_ELSE = 7, WHILE = 8, DO_WHILE = 9, FOR = 10, FOR_EACH_ARRAY = 11, FOR_EACH_ITERABLE = 12, BREAK = 13, CONTINUE = 14,
        THROW = 15, TRY = 16, CATCH = 17;
    static final byte BINARY_IMPL = 32, BINARY_MATH = 33, BOOLEAN = 34, COMPARISON = 35, CONDITIONAL = 36, ELVIS = 37, CAST = 38,
        CONSTANT = 39, NULL = 40, LOAD_VARIABLE = 41, STORE_VARIABLE = 42, UNARY_MATH = 43, INSTANCEOF = 44, DUP = 45,
        FLIP_ARRAY_INDEX = 46, FLIP_COLLECTION_INDEX = 47, FLIP_DEF_INDEX = 48, INVOKE_CALL = 49, INVOKE_CALL_DEF = 50,
        INVOKE_CALL_MEMBER = 51, LIST_INIT = 52, MAP_INIT = 53, NEW_ARRAY = 54, NEW_OBJECT = 55, LOAD_BRACE = 56,
        LOAD_BRACE_DEF = 57, LOAD_DOT_ARRAY_LENGTH = 58, LOAD_DOT_DEF = 59, LOAD_DOT = 60, LOAD_DOT_SHORTCUT = 61,
        LOAD_LIST_SHORTCUT = 62, LOAD_MAP_SHORTCUT = 63, STORE_BRACE = 64, STORE_BRACE_DEF = 65, STORE_DOT_DEF = 66,
        STORE_DOT = 67, STORE_DOT_SHORTCUT = 68, STORE_LIST_SHORTCUT = 69, STORE_MAP_SHORTCUT = 70, STRING_CONCAT = 71,
        NULL_SAFE_SUB = 72, STATIC = 73, LOAD_FIELD_MEMBER = 74, STORE_FIELD_MEMBER = 75, TYPED_INTERFACE_REFERENCE = 76,
        DEF_INTERFACE_REFERENCE = 77, TYPED_CAPTURE_REFERENCE = 78;

    /** Functions every script class has that the runtime answers itself. */
    private static final Set<String> GENERATED = Set.of("getName", "getSource", "getStatements", "$bootstrapDef");

    private static final class Writer implements IRTreeVisitor<Void> {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();
        final Map<String, Object> constants;
        String unsupported;

        Writer(Map<String, Object> constants) {
            this.constants = new java.util.HashMap<>(constants);
        }

        void fail(String what) {
            if (unsupported == null) {
                unsupported = what;
            }
        }

        void u8(int v) {
            out.write(v);
        }

        void bool(boolean v) {
            out.write(v ? 1 : 0);
        }

        void i32(int v) {
            out.write(v);
            out.write(v >>> 8);
            out.write(v >>> 16);
            out.write(v >>> 24);
        }

        void i64(long v) {
            i32((int) v);
            i32((int) (v >>> 32));
        }

        void str(String s) {
            if (s == null) {
                i32(-1);
                return;
            }
            byte[] b = s.getBytes(StandardCharsets.UTF_8);
            i32(b.length);
            out.write(b, 0, b.length);
        }

        void ty(Class<?> c) {
            str(c == null ? null : c.getName());
        }

        void types(List<Class<?>> cs) {
            i32(cs.size());
            for (Class<?> c : cs) {
                ty(c);
            }
        }

        void op(Operation o) {
            str(o == null ? null : o.name());
        }

        /** A method: its Painless class, name, declaring Java class, static flag, parameters, return. */
        void method(PainlessMethod m) {
            if (m == null) {
                bool(false);
                return;
            }
            bool(true);
            ty(m.targetClass);
            str(m.javaMethod.getName());
            ty(m.javaMethod.getDeclaringClass());
            bool(java.lang.reflect.Modifier.isStatic(m.javaMethod.getModifiers()));
            types(m.typeParameters);
            ty(m.returnType);
        }

        void constructor(PainlessConstructor c) {
            if (c == null) {
                bool(false);
                return;
            }
            bool(true);
            ty(c.javaConstructor.getDeclaringClass());
            types(c.typeParameters);
        }

        void cast(PainlessCast c) {
            if (c == null) {
                bool(false);
                return;
            }
            bool(true);
            ty(c.originalType);
            ty(c.targetType);
            bool(c.explicitCast);
            ty(c.unboxOriginalType);
            ty(c.unboxTargetType);
            ty(c.boxOriginalType);
            ty(c.boxTargetType);
        }

        void field(PainlessField f) {
            ty(f.javaField.getDeclaringClass());
            str(f.javaField.getName());
            bool(java.lang.reflect.Modifier.isStatic(f.javaField.getModifiers()));
            ty(f.typeParameter);
        }

        void constant(Object v) {
            if (v == null) {
                u8(0);
            } else if (v instanceof Boolean b) {
                u8(1);
                bool(b);
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
            } else if (v instanceof Pattern p) {
                u8(10);
                str(p.pattern());
                i32(p.flags());
            } else {
                fail("constant " + v.getClass().getName());
                u8(0);
            }
        }

        void expr(ExpressionNode n) {
            ty(n.getExpressionType());
        }

        void node(IRNode n) {
            if (n == null) {
                u8(0);
            } else {
                n.visit(this, null);
            }
        }

        void nodes(List<? extends IRNode> ns) {
            i32(ns.size());
            for (IRNode n : ns) {
                node(n);
            }
        }

        void script(ClassNode c) {
            collectStaticFields(c);
            out.write('P');
            out.write('I');
            out.write('R');
            out.write('1');
            List<FunctionNode> fns = c.getFunctionsNodes().stream().filter(f -> GENERATED.contains(f.getName()) == false).toList();
            i32(fns.size());
            for (FunctionNode f : fns) {
                visitFunction(f, null);
            }
        }

        /**
         * The static fields the class initialiser sets to a constant call (a regex literal's
         * {@code Pattern.compile(source, flags)}): read as that value.
         */
        void collectStaticFields(ClassNode c) {
            for (StatementNode st : c.getClinitBlockNode().getStatementsNodes()) {
                if (st instanceof StatementExpressionNode se
                    && se.getExpressionNode() instanceof StoreFieldMemberNode store
                    && store.isStatic()
                    && store.getChildNode() instanceof BinaryImplNode bin
                    && bin.getRightNode() instanceof InvokeCallNode call
                    && call.getMethod().javaMethod.getDeclaringClass() == Pattern.class
                    && call.getMethod().javaMethod.getName().equals("compile")
                    && call.getArgumentNodes().size() == 2
                    && call.getArgumentNodes().get(0) instanceof ConstantNode pattern
                    && call.getArgumentNodes().get(1) instanceof ConstantNode flags) {
                    constants.put(
                        store.getName(),
                        Pattern.compile((String) pattern.getConstant(), (Integer) flags.getConstant())
                    );
                }
            }
        }

        @Override
        public void visitClass(ClassNode n, Void s) {
            throw new IllegalStateException();
        }

        @Override
        public void visitFunction(FunctionNode n, Void s) {
            str(n.getName());
            ty(n.getReturnType());
            types(n.getTypeParameters());
            i32(n.getParameterNames().size());
            for (String p : n.getParameterNames()) {
                str(p);
            }
            bool(n.isStatic());
            bool(n.isSynthetic());
            i32(n.getMaxLoopCounter());
            BlockNode body = n.getBlockNode();
            // execute's body is wrapped in a try that turns Java errors into ScriptExceptions: the
            // runtime reports its own errors, so only the block inside is kept.
            if (body.getStatementsNodes().size() == 1 && body.getStatementsNodes().get(0) instanceof TryNode t
                && t.getCatchsNodes().stream().allMatch(cn -> cn.getSymbol().startsWith("#"))) {
                body = t.getBlockNode();
            }
            node(body);
        }

        @Override
        public void visitField(FieldNode n, Void s) {
            throw new IllegalStateException();
        }

        @Override
        public void visitStatementExpression(StatementExpressionNode n, Void s) {
            u8(STATEMENT_EXPRESSION);
            node(n.getExpressionNode());
        }

        @Override
        public void visitBlock(BlockNode n, Void s) {
            u8(BLOCK);
            nodes(n.getStatementsNodes());
        }

        @Override
        public void visitIf(IfNode n, Void s) {
            u8(IF);
            node(n.getConditionNode());
            node(n.getBlockNode());
        }

        @Override
        public void visitIfElse(IfElseNode n, Void s) {
            u8(IF_ELSE);
            node(n.getConditionNode());
            node(n.getBlockNode());
            node(n.getElseBlockNode());
        }

        @Override
        public void visitWhileLoop(WhileLoopNode n, Void s) {
            u8(WHILE);
            bool(n.isContinuous());
            node(n.getConditionNode());
            node(n.getBlockNode());
        }

        @Override
        public void visitDoWhileLoop(DoWhileLoopNode n, Void s) {
            u8(DO_WHILE);
            bool(n.isContinuous());
            node(n.getConditionNode());
            node(n.getBlockNode());
        }

        @Override
        public void visitForLoop(ForLoopNode n, Void s) {
            u8(FOR);
            bool(n.isContinuous());
            node(n.getInitializerNode());
            node(n.getConditionNode());
            node(n.getAfterthoughtNode());
            node(n.getBlockNode());
        }

        @Override
        public void visitForEachLoop(ForEachLoopNode n, Void s) {
            node(n.getConditionNode());
        }

        @Override
        public void visitForEachSubArrayLoop(ForEachSubArrayNode n, Void s) {
            u8(FOR_EACH_ARRAY);
            ty(n.getVariableType());
            str(n.getVariableName());
            cast(n.getCast());
            ty(n.getArrayType());
            ty(n.getIndexedType());
            node(n.getConditionNode());
            node(n.getBlockNode());
        }

        @Override
        public void visitForEachSubIterableLoop(ForEachSubIterableNode n, Void s) {
            u8(FOR_EACH_ITERABLE);
            ty(n.getVariableType());
            str(n.getVariableName());
            cast(n.getCast());
            method(n.getMethod());
            node(n.getConditionNode());
            node(n.getBlockNode());
        }

        @Override
        public void visitDeclarationBlock(DeclarationBlockNode n, Void s) {
            u8(DECLARATION_BLOCK);
            nodes(n.getDeclarationsNodes());
        }

        @Override
        public void visitDeclaration(DeclarationNode n, Void s) {
            u8(DECLARATION);
            str(n.getName());
            ty(n.getDeclarationType());
            node(n.getExpressionNode());
        }

        @Override
        public void visitReturn(ReturnNode n, Void s) {
            u8(RETURN);
            node(n.getExpressionNode());
        }

        @Override
        public void visitNullSafeSub(NullSafeSubNode n, Void s) {
            u8(NULL_SAFE_SUB);
            expr(n);
            node(n.getChildNode());
        }

        @Override
        public void visitThrow(ThrowNode n, Void s) {
            u8(THROW);
            node(n.getExpressionNode());
        }

        @Override
        public void visitTry(TryNode n, Void s) {
            u8(TRY);
            node(n.getBlockNode());
            nodes(n.getCatchsNodes());
        }

        @Override
        public void visitCatch(CatchNode n, Void s) {
            u8(CATCH);
            ty(n.getExceptionType());
            str(n.getSymbol());
            node(n.getBlockNode());
        }

        @Override
        public void visitBreak(BreakNode n, Void s) {
            u8(BREAK);
        }

        @Override
        public void visitContinue(ContinueNode n, Void s) {
            u8(CONTINUE);
        }

        @Override
        public void visitBinaryImpl(BinaryImplNode n, Void s) {
            u8(BINARY_IMPL);
            expr(n);
            node(n.getLeftNode());
            node(n.getRightNode());
        }

        @Override
        public void visitUnaryMath(UnaryMathNode n, Void s) {
            u8(UNARY_MATH);
            expr(n);
            op(n.getOperation());
            ty(n.getUnaryType());
            bool(n.getCat());
            bool(n.getOriginallyExplicit());
            node(n.getChildNode());
        }

        @Override
        public void visitBinaryMath(BinaryMathNode n, Void s) {
            u8(BINARY_MATH);
            expr(n);
            op(n.getOperation());
            ty(n.getBinaryType());
            ty(n.getShiftType());
            i32(n.getFlags());
            i32(n.getRegexLimit());
            node(n.getLeftNode());
            node(n.getRightNode());
        }

        @Override
        public void visitStringConcatenation(StringConcatenationNode n, Void s) {
            u8(STRING_CONCAT);
            expr(n);
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitBoolean(BooleanNode n, Void s) {
            u8(BOOLEAN);
            expr(n);
            op(n.getOperation());
            node(n.getLeftNode());
            node(n.getRightNode());
        }

        @Override
        public void visitComparison(ComparisonNode n, Void s) {
            u8(COMPARISON);
            expr(n);
            op(n.getOperation());
            ty(n.getComparisonType());
            node(n.getLeftNode());
            node(n.getRightNode());
        }

        @Override
        public void visitCast(CastNode n, Void s) {
            u8(CAST);
            expr(n);
            cast(n.getCast());
            node(n.getChildNode());
        }

        @Override
        public void visitInstanceof(InstanceofNode n, Void s) {
            u8(INSTANCEOF);
            expr(n);
            ty(n.getInstanceType());
            node(n.getChildNode());
        }

        @Override
        public void visitConditional(ConditionalNode n, Void s) {
            u8(CONDITIONAL);
            expr(n);
            node(n.getConditionNode());
            node(n.getLeftNode());
            node(n.getRightNode());
        }

        @Override
        public void visitElvis(ElvisNode n, Void s) {
            u8(ELVIS);
            expr(n);
            node(n.getLeftNode());
            node(n.getRightNode());
        }

        @Override
        public void visitListInitialization(ListInitializationNode n, Void s) {
            u8(LIST_INIT);
            expr(n);
            constructor(n.getConstructor());
            method(n.getMethod());
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitMapInitialization(MapInitializationNode n, Void s) {
            u8(MAP_INIT);
            expr(n);
            constructor(n.getConstructor());
            method(n.getMethod());
            nodes(n.getKeyNodes());
            nodes(n.getValueNodes());
        }

        @Override
        public void visitNewArray(NewArrayNode n, Void s) {
            u8(NEW_ARRAY);
            expr(n);
            bool(n.getInitialize());
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitNewObject(NewObjectNode n, Void s) {
            u8(NEW_OBJECT);
            expr(n);
            constructor(n.getConstructor());
            bool(n.getRead());
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitConstant(ConstantNode n, Void s) {
            u8(CONSTANT);
            expr(n);
            constant(n.getConstant());
        }

        @Override
        public void visitNull(NullNode n, Void s) {
            u8(NULL);
            expr(n);
        }

        @Override
        public void visitDefInterfaceReference(DefInterfaceReferenceNode n, Void s) {
            u8(DEF_INTERFACE_REFERENCE);
            expr(n);
            str(n.getDefReferenceEncoding());
            i32(n.getCaptures().size());
            for (String c : n.getCaptures()) {
                str(c);
            }
        }

        @Override
        public void visitTypedInterfaceReference(TypedInterfaceReferenceNode n, Void s) {
            u8(TYPED_INTERFACE_REFERENCE);
            expr(n);
            var r = n.getReference();
            str(r.interfaceMethodName);
            str(r.delegateClassName);
            bool(r.isDelegateInterface);
            bool(r.isDelegateAugmented);
            i32(r.delegateInvokeType);
            str(r.delegateMethodName);
            i32(r.delegateMethodType.parameterCount());
            for (Class<?> p : r.delegateMethodType.parameterList()) {
                ty(p);
            }
            ty(r.delegateMethodType.returnType());
            i32(r.interfaceMethodType.parameterCount());
            for (Class<?> p : r.interfaceMethodType.parameterList()) {
                ty(p);
            }
            ty(r.interfaceMethodType.returnType());
            i32(r.factoryMethodType.parameterCount());
            if (r.delegateInjections.length > 0) {
                fail("method reference injections");
            }
            i32(n.getCaptures().size());
            for (String c : n.getCaptures()) {
                str(c);
            }
        }

        @Override
        public void visitTypeCaptureReference(TypedCaptureReferenceNode n, Void s) {
            u8(TYPED_CAPTURE_REFERENCE);
            expr(n);
            str(n.getMethodName());
            i32(n.getCaptures().size());
            for (String c : n.getCaptures()) {
                str(c);
            }
        }

        @Override
        public void visitStatic(StaticNode n, Void s) {
            u8(STATIC);
            expr(n);
        }

        @Override
        public void visitLoadVariable(LoadVariableNode n, Void s) {
            u8(LOAD_VARIABLE);
            expr(n);
            str(n.getName());
        }

        @Override
        public void visitLoadDotArrayLengthNode(LoadDotArrayLengthNode n, Void s) {
            u8(LOAD_DOT_ARRAY_LENGTH);
            expr(n);
        }

        @Override
        public void visitLoadDotDef(LoadDotDefNode n, Void s) {
            u8(LOAD_DOT_DEF);
            expr(n);
            str(n.getValue());
        }

        @Override
        public void visitLoadDot(LoadDotNode n, Void s) {
            u8(LOAD_DOT);
            expr(n);
            field(n.getField());
        }

        @Override
        public void visitLoadDotShortcut(LoadDotShortcutNode n, Void s) {
            u8(LOAD_DOT_SHORTCUT);
            expr(n);
            method(n.getGetter());
        }

        @Override
        public void visitLoadListShortcut(LoadListShortcutNode n, Void s) {
            u8(LOAD_LIST_SHORTCUT);
            expr(n);
            method(n.getGetter());
        }

        @Override
        public void visitLoadMapShortcut(LoadMapShortcutNode n, Void s) {
            u8(LOAD_MAP_SHORTCUT);
            expr(n);
            method(n.getGetter());
        }

        @Override
        public void visitLoadFieldMember(LoadFieldMemberNode n, Void s) {
            if (n.isStatic() && constants.containsKey(n.getName())) {
                // A static constant (a regex): its value, as a constant node.
                u8(CONSTANT);
                expr(n);
                constant(constants.get(n.getName()));
                return;
            }
            u8(LOAD_FIELD_MEMBER);
            expr(n);
            str(n.getName());
            bool(n.isStatic());
        }

        @Override
        public void visitLoadBraceDef(LoadBraceDefNode n, Void s) {
            u8(LOAD_BRACE_DEF);
            expr(n);
            ty(n.getIndexType());
        }

        @Override
        public void visitLoadBrace(LoadBraceNode n, Void s) {
            u8(LOAD_BRACE);
            expr(n);
        }

        @Override
        public void visitStoreVariable(StoreVariableNode n, Void s) {
            u8(STORE_VARIABLE);
            expr(n);
            ty(n.getStoreType());
            str(n.getName());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreDotDef(StoreDotDefNode n, Void s) {
            u8(STORE_DOT_DEF);
            expr(n);
            ty(n.getStoreType());
            str(n.getValue());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreDot(StoreDotNode n, Void s) {
            u8(STORE_DOT);
            expr(n);
            ty(n.getStoreType());
            field(n.getField());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreDotShortcut(StoreDotShortcutNode n, Void s) {
            u8(STORE_DOT_SHORTCUT);
            expr(n);
            ty(n.getStoreType());
            method(n.getSetter());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreListShortcut(StoreListShortcutNode n, Void s) {
            u8(STORE_LIST_SHORTCUT);
            expr(n);
            ty(n.getStoreType());
            method(n.getSetter());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreMapShortcut(StoreMapShortcutNode n, Void s) {
            u8(STORE_MAP_SHORTCUT);
            expr(n);
            ty(n.getStoreType());
            method(n.getSetter());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreFieldMember(StoreFieldMemberNode n, Void s) {
            u8(STORE_FIELD_MEMBER);
            expr(n);
            ty(n.getStoreType());
            str(n.getName());
            bool(n.isStatic());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreBraceDef(StoreBraceDefNode n, Void s) {
            u8(STORE_BRACE_DEF);
            expr(n);
            ty(n.getStoreType());
            ty(n.getIndexType());
            node(n.getChildNode());
        }

        @Override
        public void visitStoreBrace(StoreBraceNode n, Void s) {
            u8(STORE_BRACE);
            expr(n);
            ty(n.getStoreType());
            node(n.getChildNode());
        }

        @Override
        public void visitInvokeCallDef(InvokeCallDefNode n, Void s) {
            u8(INVOKE_CALL_DEF);
            expr(n);
            str(n.getName());
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitInvokeCall(InvokeCallNode n, Void s) {
            u8(INVOKE_CALL);
            expr(n);
            method(n.getMethod());
            ty(n.getBox());
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitInvokeCallMember(InvokeCallMemberNode n, Void s) {
            u8(INVOKE_CALL_MEMBER);
            expr(n);
            LocalFunction lf = n.getLocalFunction();
            PainlessClassBinding cb = n.getClassBinding();
            if (lf != null) {
                u8(1);
                str(lf.getFunctionName());
                bool(lf.isInternal());
                bool(lf.isStatic());
                types(lf.getTypeParameters());
                ty(lf.getReturnType());
            } else if (n.getImportedMethod() != null) {
                u8(2);
                method(n.getImportedMethod());
            } else if (cb != null) {
                u8(3);
                ty(cb.javaConstructor.getDeclaringClass());
                str(cb.javaMethod.getName());
                i32(cb.javaConstructor.getParameterCount());
                i32(n.getClassBindingOffset());
                types(cb.typeParameters);
                ty(cb.returnType);
            } else if (n.getInstanceBinding() != null) {
                u8(4);
                ty(n.getInstanceBinding().targetInstance.getClass());
                str(n.getInstanceBinding().javaMethod.getName());
                types(n.getInstanceBinding().typeParameters);
                ty(n.getInstanceBinding().returnType);
            } else {
                fail("member call without a target");
                u8(0);
            }
            nodes(n.getArgumentNodes());
        }

        @Override
        public void visitFlipArrayIndex(FlipArrayIndexNode n, Void s) {
            u8(FLIP_ARRAY_INDEX);
            expr(n);
            node(n.getChildNode());
        }

        @Override
        public void visitFlipCollectionIndex(FlipCollectionIndexNode n, Void s) {
            u8(FLIP_COLLECTION_INDEX);
            expr(n);
            node(n.getChildNode());
        }

        @Override
        public void visitFlipDefIndex(FlipDefIndexNode n, Void s) {
            u8(FLIP_DEF_INDEX);
            expr(n);
            node(n.getChildNode());
        }

        @Override
        public void visitDup(DupNode n, Void s) {
            u8(DUP);
            expr(n);
            i32(n.getSize());
            i32(n.getDepth());
            node(n.getChildNode());
        }
    }
}
