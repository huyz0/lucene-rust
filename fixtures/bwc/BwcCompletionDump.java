import java.nio.file.Path;

/**
 * Lucene 10.5.0 (with backward-codecs) over an index {@code BwcCompletion} wrote with an older
 * release: {@code GenCompletionQuery}'s every query and answer, as {@code queries.tsv} beside it.
 */
public class BwcCompletionDump {
  public static void main(String[] args) throws Exception {
    GenCompletionQuery.writeQueries(Path.of(args[0]));
  }
}
