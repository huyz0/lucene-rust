import java.io.*; import java.nio.charset.*; import java.nio.file.*; import java.util.*;
/** The engine study's corpus order: bgl, hdfs, openssh, thunderbird interleaved by Random(7), one line per document, UTF-8. */
public class SubstringCorpus {
  public static void main(String[] a) throws Exception {
    String[] sets = {"bgl","hdfs","openssh","thunderbird"};
    BufferedReader[] r = new BufferedReader[4];
    for (int i=0;i<4;i++) r[i] = new BufferedReader(new InputStreamReader(Files.newInputStream(Paths.get(a[0], sets[i]+".txt")),
        StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPLACE).onUnmappableCharacter(CodingErrorAction.REPLACE)), 1<<16);
    boolean[] done = new boolean[4]; int live=4, n=0; Random rnd = new Random(7);
    try (Writer w = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(a[1]), StandardCharsets.UTF_8), 1<<20)) {
      while (live>0) { int s=rnd.nextInt(4); if(done[s]) continue; String line=r[s].readLine(); if(line==null){done[s]=true;live--;continue;}
        if(line.isEmpty()) continue; w.write(line); w.write('\n'); n++; }
    }
    System.out.println("lines="+n);
  }
}
