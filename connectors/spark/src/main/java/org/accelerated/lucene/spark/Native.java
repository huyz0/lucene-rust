package org.accelerated.lucene.spark;

import java.io.IOException;

/**
 * The native library's entry points. The library is loaded from the path in the
 * system property {@code lucene.native.library}, else from {@code LUCENE_NATIVE_LIBRARY},
 * else by its name from {@code java.library.path}.
 */
final class Native {
  private static final Object LOCK = new Object();
  private static boolean loaded;

  private Native() {}

  static void load() {
    synchronized (LOCK) {
      if (loaded) {
        return;
      }
      String path = System.getProperty("lucene.native.library", System.getenv("LUCENE_NATIVE_LIBRARY"));
      if (path != null && !path.isEmpty()) {
        System.load(path);
      } else {
        System.loadLibrary("lucene_ffi");
      }
      loaded = true;
    }
  }

  /** Opens the index at {@code path} as the table {@code spec} (JSON) declares. */
  static native long open(String path, String spec) throws IOException;

  static native void close(long table) throws IOException;

  /** {@code {"segments": n, "docs": n, "max_doc": n}}. */
  static native String info(long table) throws IOException;

  /**
   * Fills the {@code ArrowArrayStream} at address {@code stream} with partition
   * {@code partition} of {@code partitions} of the scan {@code request} (JSON) describes.
   */
  static native void scan(long table, String request, int partition, int partitions, long stream)
      throws IOException;
}
