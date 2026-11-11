/*
 * Licensed under the Apache License, Version 2.0 -- see the repository's LICENSE.
 */
package org.lucenerust.opensearch.neural.util.lang;

/** The commons-lang3 {@code StringUtils} methods neural search calls. */
public final class StringUtils {
    public static final String EMPTY = "";

    private StringUtils() {}

    public static boolean isEmpty(CharSequence s) {
        return s == null || s.length() == 0;
    }

    public static boolean isNotEmpty(CharSequence s) {
        return isEmpty(s) == false;
    }

    public static boolean isBlank(CharSequence s) {
        if (s == null) {
            return true;
        }
        for (int i = 0; i < s.length(); i++) {
            if (Character.isWhitespace(s.charAt(i)) == false) {
                return false;
            }
        }
        return true;
    }

    public static boolean isNotBlank(CharSequence s) {
        return isBlank(s) == false;
    }

    public static String defaultString(String s, String defaultStr) {
        return s == null ? defaultStr : s;
    }
}
