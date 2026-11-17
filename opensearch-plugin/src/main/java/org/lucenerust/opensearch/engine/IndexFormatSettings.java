package org.lucenerust.opensearch.engine;

import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;

/**
 * Which Lucene release's index files the engine writes. Lucene 10.6 changed two on-disk formats
 * (doc-values format 3 with its {@code .dvp} file, and {@code segments_N} version 11) that Lucene
 * 10.5 cannot open, so an index written for a 10.6 host cannot be opened by a node still running
 * 10.5 (OpenSearch 3.9). The engine therefore writes the newest format the Lucene on this node
 * reads, unless told otherwise:
 *
 * <ul>
 *   <li>{@code index.codec.index_format} on the index, then
 *   <li>{@code accelerated_search.index_format} in the node's settings, then
 *   <li>{@code auto}: the Lucene this node runs ({@code Version.LATEST}).
 * </ul>
 *
 * Read when the engine opens, so a change takes effect after the index is closed and reopened (segments
 * already written keep their format, and every node reads both). A value naming a newer format than this node's Lucene reads is refused: the node would write
 * files it cannot open itself.
 */
public final class IndexFormatSettings {
    public static final String AUTO = "auto";
    public static final int FORMAT_105 = 105;
    public static final int FORMAT_106 = 106;

    /** {@code auto} (default), {@code 10.5} or {@code 10.6}. */
    public static final Setting<String> INDEX_FORMAT = new Setting<>(
        "index.codec.index_format",
        AUTO,
        IndexFormatSettings::validate,
        Setting.Property.IndexScope
    );

    /** The default for indices that leave {@link #INDEX_FORMAT} at {@code auto}. */
    public static final String NODE_DEFAULT_KEY = "accelerated_search.index_format";

    private static volatile String nodeDefault = AUTO;

    private IndexFormatSettings() {}

    /** Reads {@link #NODE_DEFAULT_KEY} from the node's settings, once at plugin start. */
    public static void readNodeDefault(Settings nodeSettings) {
        nodeDefault = validate(nodeSettings.get(NODE_DEFAULT_KEY, AUTO));
    }

    private static String validate(String v) {
        return switch (v) {
            case AUTO, "10.5", "10.6" -> v;
            default -> throw new IllegalArgumentException("[index.codec.index_format] must be [auto], [10.5] or [10.6], got [" + v + "]");
        };
    }

    /** The format ({@code 105} or {@code 106}) to write for an index with these settings. */
    public static int resolve(Settings indexSettings) {
        return resolve(INDEX_FORMAT.get(indexSettings), nodeDefault, org.apache.lucene.util.Version.LATEST);
    }

    public static int resolve(String indexValue, String nodeValue, org.apache.lucene.util.Version host) {
        int hostFormat = (host.major > 10 || (host.major == 10 && host.minor >= 6)) ? FORMAT_106 : FORMAT_105;
        String chosen = AUTO.equals(indexValue) ? nodeValue : indexValue;
        if (AUTO.equals(chosen)) {
            return hostFormat;
        }
        int format = "10.6".equals(chosen) ? FORMAT_106 : FORMAT_105;
        if (format > hostFormat) {
            throw new IllegalArgumentException(
                "index format [" + chosen + "] is newer than the Lucene this node runs [" + host + "] and could not be opened by it"
            );
        }
        return format;
    }
}
