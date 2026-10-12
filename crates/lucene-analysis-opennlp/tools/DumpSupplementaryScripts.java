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

import java.lang.Character.UnicodeScript;

/**
 * Writes {@code src/tools/supplementary_scripts.rs}: the runs of supplementary
 * code points whose {@code Character.UnicodeScript} is one of the five
 * OpenNLP's {@code StringPattern.recognize} tells apart (COMMON, LATIN, HAN,
 * HIRAGANA, KATAKANA), as the running JDK answers. Run on JDK 25:
 * {@code java crates/lucene-analysis-opennlp/tools/DumpSupplementaryScripts.java
 * > crates/lucene-analysis-opennlp/src/tools/supplementary_scripts.rs}.
 */
public class DumpSupplementaryScripts {
  public static void main(String[] args) {
    UnicodeScript[] wanted = {
      UnicodeScript.COMMON, UnicodeScript.LATIN, UnicodeScript.HAN,
      UnicodeScript.HIRAGANA, UnicodeScript.KATAKANA
    };
    StringBuilder b = new StringBuilder();
    b.append("//! `Character.UnicodeScript.of(cp)` for the supplementary code points of\n");
    b.append("//! the five scripts `StringPattern.recognize` tells apart, as runs\n");
    b.append("//! `(first, last, script)` with script 0 COMMON, 1 LATIN, 2 HAN,\n");
    b.append("//! 3 HIRAGANA, 4 KATAKANA (every other supplementary code point is of\n");
    b.append("//! another script).\n");
    b.append("//!\n");
    b.append("//! Generated -- do not edit: the output of\n");
    b.append("//! `java crates/lucene-analysis-opennlp/tools/DumpSupplementaryScripts.java`\n");
    b.append("//! on JDK ").append(Runtime.version().feature()).append(".\n\n");
    b.append("/// The runs, in code point order.\n");
    b.append("pub static RUNS: &[(u32, u32, u8)] = &[\n");
    int start = -1;
    int script = -1;
    for (int cp = 0x10000; cp <= 0x110000; cp++) {
      int s = -1;
      if (cp <= Character.MAX_CODE_POINT) {
        UnicodeScript us = UnicodeScript.of(cp);
        for (int i = 0; i < wanted.length; i++) {
          if (wanted[i] == us) s = i;
        }
      }
      if (s != script) {
        if (script >= 0) {
          b.append(String.format("    (0x%X, 0x%X, %d),%n", start, cp - 1, script));
        }
        start = cp;
        script = s;
      }
    }
    b.append("];\n");
    System.out.print(b);
  }
}
