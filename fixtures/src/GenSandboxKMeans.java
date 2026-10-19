import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.sandbox.codecs.quantization.KMeans;
import org.apache.lucene.sandbox.codecs.quantization.KMeans.KmeansInitializationMethod;
import org.apache.lucene.sandbox.codecs.quantization.SampleReader;
import org.apache.lucene.util.VectorUtil;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * lucene-sandbox's {@code KMeans} and {@code SampleReader} (package {@code
 * sandbox.codecs.quantization}): three vector sets and the clustering Lucene computes over them
 * under every initialization method, with and without sampling, cosine normalization, the
 * one-cluster and zero-iteration shapes, a cluster count shrunk to the data, and a set of mostly
 * duplicate vectors that leaves centroids empty (so {@code assignCentroids} runs). Also {@code
 * reservoirSample}/{@code reservoirSampleFromArray} outputs and the argument errors.
 *
 * <p>Dimensions stay at or below 16: there both Lucene's scalar and its Panama {@code
 * squareDistance}/{@code dotProduct} are one sequential {@code fma} loop, so every centroid
 * compares bit for bit (the fixtures run the scalar one; this port's kernel is Panama's layout).
 *
 * <p>FORGY takes its centroids in an {@code IntHashSet}'s iteration order, which follows a
 * process-wide seed; that case is written canonically (centroids sorted by their bits, the
 * assignments renumbered to match) so the fixture is deterministic.
 *
 * <p>Output (under {@code sandbox_kmeans/}): {@code vectors.<set>.txt} -- one vector per line,
 * float bits as hex; {@code case.<name>.txt} -- {@code centroids k dim}, k lines of bits, then
 * {@code assign n} and the comma-separated centroid of every vector, or {@code assign null};
 * {@code reservoir.txt} and {@code errors.txt}.
 */
public class GenSandboxKMeans {
  public static void main(String[] args) throws IOException {
    Path out = Path.of(args[0]).resolve("sandbox_kmeans");
    Files.createDirectories(out);

    List<float[]> blobs = blobs(new Random(1), 2000, 8, 12, false);
    List<float[]> sphere = blobs(new Random(2), 2000, 16, 8, true);
    List<float[]> dupes = new ArrayList<>();
    Random r = new Random(3);
    for (int i = 0; i < 1200; i++) {
      if (i < 100 || r.nextInt(10) != 0) {
        dupes.add(new float[] {1, 2, 3, 4});
      } else {
        dupes.add(new float[] {r.nextFloat() * 8, r.nextFloat() * 8, r.nextFloat(), -r.nextFloat()});
      }
    }
    writeVectors(out.resolve("vectors.blobs.txt"), blobs);
    writeVectors(out.resolve("vectors.sphere.txt"), sphere);
    writeVectors(out.resolve("vectors.dupes.txt"), dupes);

    FloatVectorValues a = FloatVectorValues.fromFloats(blobs, 8);
    FloatVectorValues b = FloatVectorValues.fromFloats(sphere, 16);
    FloatVectorValues c = FloatVectorValues.fromFloats(dupes, 4);

    write(out, "default_euclid", KMeans.cluster(a, VectorSimilarityFunction.EUCLIDEAN, 12), false);
    write(out, "default_cosine", KMeans.cluster(b, VectorSimilarityFunction.COSINE, 8), false);
    write(
        out,
        "sampled_pp",
        KMeans.cluster(a, 6, true, 7L, KmeansInitializationMethod.PLUS_PLUS, false, 3, 20, 600),
        false);
    write(
        out,
        "reservoir",
        KMeans.cluster(
            b, 10, true, 11L, KmeansInitializationMethod.RESERVOIR_SAMPLING, true, 2, 10, 100_000),
        false);
    write(
        out,
        "forgy",
        KMeans.cluster(a, 9, true, 13L, KmeansInitializationMethod.FORGY, false, 4, 10, 100_000),
        true);
    write(
        out,
        "no_assign",
        KMeans.cluster(a, 5, false, 17L, KmeansInitializationMethod.PLUS_PLUS, false, 2, 5, 100_000),
        false);
    write(
        out,
        "one_cluster",
        KMeans.cluster(a, 1, true, 5L, KmeansInitializationMethod.PLUS_PLUS, false, 2, 5, 100_000),
        false);
    write(
        out,
        "dupes_reservoir",
        KMeans.cluster(
            c, 8, true, 19L, KmeansInitializationMethod.RESERVOIR_SAMPLING, false, 3, 10, 100_000),
        false);
    write(
        out,
        "shrunk",
        KMeans.cluster(c, 50, true, 29L, KmeansInitializationMethod.PLUS_PLUS, false, 2, 10, 100_000),
        false);
    write(
        out,
        "zero_iters",
        KMeans.cluster(a, 4, true, 23L, KmeansInitializationMethod.PLUS_PLUS, false, 2, 0, 100_000),
        false);
    write(
        out,
        "sampled_forgy_cosine",
        KMeans.cluster(b, 3, true, 31L, KmeansInitializationMethod.FORGY, true, 2, 10, 300),
        true);

    List<String> res = new ArrayList<>();
    res.add("sample 1000 10 5 " + join(SampleReader.reservoirSample(1000, 10, 5)));
    res.add("sample 10 10 5 " + join(SampleReader.reservoirSample(10, 10, 5)));
    res.add("sample 100000 7 42 " + join(SampleReader.reservoirSample(100_000, 7, 42)));
    int[] arr = new int[150];
    for (int i = 0; i < arr.length; i++) {
      arr[i] = i * 3 + 1;
    }
    res.add("array 150 7 9 " + join(SampleReader.reservoirSampleFromArray(arr, 7, 9)));
    res.add("array 150 150 9 " + join(SampleReader.reservoirSampleFromArray(arr, 150, 9)));
    SampleReader sr = SampleReader.createSampleReader(a, 5, 42L);
    StringBuilder sb = new StringBuilder("reader " + sr.size() + " " + sr.dimension());
    for (int i = 0; i < sr.size(); i++) {
      sb.append(' ').append(bits(sr.vectorValue(i)));
    }
    res.add(sb.toString());
    Files.write(out.resolve("reservoir.txt"), res, StandardCharsets.UTF_8);

    List<String> errors = new ArrayList<>();
    for (int k : new int[] {0, 32768}) {
      try {
        KMeans.cluster(a, VectorSimilarityFunction.EUCLIDEAN, k);
        errors.add(k + "\tnone");
      } catch (IllegalArgumentException e) {
        errors.add(k + "\t" + e.getMessage());
      }
    }
    errors.add(
        "empty\t"
            + KMeans.cluster(
                FloatVectorValues.fromFloats(new ArrayList<>(), 4),
                VectorSimilarityFunction.EUCLIDEAN,
                3));
    Files.write(out.resolve("errors.txt"), errors, StandardCharsets.UTF_8);
  }

  /** {@code n} points around {@code k} random centres; on the unit sphere if {@code unit}. */
  static List<float[]> blobs(Random r, int n, int dim, int k, boolean unit) {
    float[][] centres = new float[k][dim];
    for (float[] c : centres) {
      for (int d = 0; d < dim; d++) {
        c[d] = (r.nextFloat() - 0.5f) * 20;
      }
    }
    List<float[]> vs = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      float[] c = centres[r.nextInt(k)];
      float[] v = new float[dim];
      for (int d = 0; d < dim; d++) {
        v[d] = c[d] + (float) r.nextGaussian();
      }
      if (unit) {
        VectorUtil.l2normalize(v);
      }
      vs.add(v);
    }
    return vs;
  }

  static void writeVectors(Path p, List<float[]> vs) throws IOException {
    List<String> lines = new ArrayList<>();
    for (float[] v : vs) {
      lines.add(bits(v));
    }
    Files.write(p, lines, StandardCharsets.UTF_8);
  }

  static String bits(float[] v) {
    StringBuilder sb = new StringBuilder();
    for (int d = 0; d < v.length; d++) {
      if (d > 0) {
        sb.append(',');
      }
      sb.append(Integer.toHexString(Float.floatToRawIntBits(v[d])));
    }
    return sb.toString();
  }

  static String join(int[] xs) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < xs.length; i++) {
      if (i > 0) {
        sb.append(',');
      }
      sb.append(xs[i]);
    }
    return sb.toString();
  }

  static void write(Path out, String name, KMeans.Results res, boolean canonical)
      throws IOException {
    float[][] cs = res.centroids();
    short[] assign = res.vectorCentroids();
    if (canonical) {
      Integer[] order = new Integer[cs.length];
      for (int i = 0; i < order.length; i++) {
        order[i] = i;
      }
      Arrays.sort(order, (x, y) -> bits(cs[x]).compareTo(bits(cs[y])));
      float[][] sorted = new float[cs.length][];
      short[] rank = new short[cs.length];
      for (int i = 0; i < order.length; i++) {
        sorted[i] = cs[order[i]];
        rank[order[i]] = (short) i;
      }
      if (assign != null) {
        short[] mapped = new short[assign.length];
        for (int i = 0; i < assign.length; i++) {
          mapped[i] = rank[assign[i]];
        }
        assign = mapped;
      }
      res = new KMeans.Results(sorted, assign);
    }
    List<String> lines = new ArrayList<>();
    lines.add("centroids " + res.centroids().length + " " + res.centroids()[0].length);
    for (float[] centroid : res.centroids()) {
      lines.add(bits(centroid));
    }
    if (assign == null) {
      lines.add("assign null");
    } else {
      StringBuilder sb = new StringBuilder("assign " + assign.length + " ");
      for (int i = 0; i < assign.length; i++) {
        if (i > 0) {
          sb.append(',');
        }
        sb.append(assign[i]);
      }
      lines.add(sb.toString());
    }
    Files.write(out.resolve("case." + name + ".txt"), lines, StandardCharsets.UTF_8);
  }
}
