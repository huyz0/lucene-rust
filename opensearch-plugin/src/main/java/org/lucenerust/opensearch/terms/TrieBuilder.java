/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lucenerust.opensearch.terms;

import java.io.IOException;
import org.apache.lucene.store.RandomAccessInput;

/**
 * The read side of Lucene 10.5.1's {@code org.apache.lucene.codecs.lucene103.blocktree.TrieBuilder}:
 * its node encodings and child-label strategies, as {@link TrieReader} decodes them ({@code
 * blocktree.rs} reads the same).
 */
final class TrieBuilder {

  static final int SIGN_NO_CHILDREN = 0x00;
  static final int SIGN_SINGLE_CHILD_WITH_OUTPUT = 0x01;
  static final int SIGN_SINGLE_CHILD_WITHOUT_OUTPUT = 0x02;
  static final int SIGN_MULTI_CHILDREN = 0x03;

  static final int LEAF_NODE_HAS_TERMS = 1 << 5;
  static final int LEAF_NODE_HAS_FLOOR = 1 << 6;
  static final long NON_LEAF_NODE_HAS_TERMS = 1L << 1;
  static final long NON_LEAF_NODE_HAS_FLOOR = 1L << 0;

  private TrieBuilder() {}

  enum ChildSaveStrategy {

    /**
     * Store children labels in a bitset, this is likely the most efficient storage as we can
     * compute position with bitCount instruction, so we give it the highest priority.
     */
    BITS(2) {
      @Override
      int lookup(
          int targetLabel, RandomAccessInput in, long offset, int strategyBytes, int minLabel)
          throws IOException {
        int bitIndex = targetLabel - minLabel;
        if (bitIndex >= (strategyBytes << 3)) {
          return -1;
        }
        int wordIndex = bitIndex >>> 6;
        long wordFp = offset + (wordIndex << 3);
        long word = in.readLong(wordFp);
        long mask = 1L << bitIndex;
        if ((word & mask) == 0) {
          return -1;
        }
        int pos = 0;
        for (long fp = offset; fp < wordFp; fp += 8L) {
          pos += Long.bitCount(in.readLong(fp));
        }
        pos += Long.bitCount(word & (mask - 1));
        return pos;
      }
    },

    /**
     * Store labels in an array and lookup with binary search.
     *
     * <p>TODO: Can we use VectorAPI to speed up the lookup? we can check 64 labels once on AVX512!
     */
    ARRAY(1) {
      @Override
      int lookup(
          int targetLabel, RandomAccessInput in, long offset, int strategyBytes, int minLabel)
          throws IOException {
        int low = 0;
        int high = strategyBytes - 1;
        while (low <= high) {
          int mid = (low + high) >>> 1;
          int midLabel = in.readByte(offset + mid) & 0xFF;
          if (midLabel < targetLabel) {
            low = mid + 1;
          } else if (midLabel > targetLabel) {
            high = mid - 1;
          } else {
            return mid + 1;
          }
        }
        return -1;
      }
    },

    /**
     * Store labels that not existing within the range. E.g. store 10(max label) and 3, 5(absent
     * label) for [1, 2, 4, 6, 7, 8, 9, 10].
     *
     * <p>TODO: Can we use VectorAPI to speed up the lookup? we can check 64 labels once on AVX512!
     */
    REVERSE_ARRAY(0) {

      @Override
      int lookup(
          int targetLabel, RandomAccessInput in, long offset, int strategyBytes, int minLabel)
          throws IOException {
        int maxLabel = in.readByte(offset++) & 0xFF;
        if (targetLabel >= maxLabel) {
          return targetLabel == maxLabel ? maxLabel - minLabel - strategyBytes + 1 : -1;
        }
        if (strategyBytes == 1) {
          return targetLabel - minLabel;
        }

        int low = 0;
        int high = strategyBytes - 2;
        while (low <= high) {
          int mid = (low + high) >>> 1;
          int midLabel = in.readByte(offset + mid) & 0xFF;
          if (midLabel < targetLabel) {
            low = mid + 1;
          } else if (midLabel > targetLabel) {
            high = mid - 1;
          } else {
            return -1;
          }
        }
        return targetLabel - minLabel - low;
      }
    };

    private static final ChildSaveStrategy[] STRATEGIES_BY_CODE;

    static {
      STRATEGIES_BY_CODE = new ChildSaveStrategy[ChildSaveStrategy.values().length];
      for (ChildSaveStrategy strategy : ChildSaveStrategy.values()) {
        assert STRATEGIES_BY_CODE[strategy.code] == null;
        STRATEGIES_BY_CODE[strategy.code] = strategy;
      }
    }

    final int code;

    ChildSaveStrategy(int code) {
      this.code = code;
    }

    abstract int lookup(
        int targetLabel, RandomAccessInput in, long offset, int strategyBytes, int minLabel)
        throws IOException;

    static ChildSaveStrategy byCode(int code) {
      return STRATEGIES_BY_CODE[code];
    }

  }
}
