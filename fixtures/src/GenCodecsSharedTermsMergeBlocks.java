/** Shared-terms merges with full position blocks, tails and deleted positional fields. */
public class GenCodecsSharedTermsMergeBlocks {
  public static void main(String[] args) throws Exception {
    GenCodecsSharedTermsMerge.run(java.nio.file.Path.of(args[0]), true);
  }
}
