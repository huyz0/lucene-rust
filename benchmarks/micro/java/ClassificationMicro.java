import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.core.WhitespaceAnalyzer;
import org.apache.lucene.classification.BM25NBClassifier;
import org.apache.lucene.classification.BooleanPerceptronClassifier;
import org.apache.lucene.classification.CachingNaiveBayesClassifier;
import org.apache.lucene.classification.Classifier;
import org.apache.lucene.classification.KNearestFuzzyClassifier;
import org.apache.lucene.classification.KNearestNeighborClassifier;
import org.apache.lucene.classification.SimpleNaiveBayesClassifier;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.store.FSDirectory;

/**
 * M14 T14.3's pairs (the Rust twin is {@code
 * benchmarks/rust-runner/src/micro_classification.rs}): each classifier's {@code assignClass}
 * over the stored text of the first 40 documents of {@code
 * fixtures/data/classification/random} (240 documents, four classes, {@code WhitespaceAnalyzer}),
 * and {@code cls_perceptron_train}, building the perceptron. Units: classifications (or
 * trainings). Output is {@code name\tns_per_unit\tunits}.
 */
public class ClassificationMicro {
  static final int INPUTS = 40;

  public static void main(String[] args) throws Exception {
    try (DirectoryReader reader =
        DirectoryReader.open(FSDirectory.open(Path.of("fixtures/data/classification/random")))) {
      Analyzer a = new WhitespaceAnalyzer();
      StoredFields stored = reader.storedFields();
      List<String> inputs = new ArrayList<>();
      for (int doc = 0; doc < INPUTS; doc++) {
        inputs.add(stored.document(doc).get("text"));
      }
      Object[][] classifiers = {
        {"cls_snb", new SimpleNaiveBayesClassifier(reader, a, null, "cat", "text")},
        {"cls_cnb", new CachingNaiveBayesClassifier(reader, a, null, "cat", "text")},
        {"cls_bm25nb", new BM25NBClassifier(reader, a, null, "cat", "text")},
        {"cls_knn3", new KNearestNeighborClassifier(reader, null, a, null, 3, 0, 0, "cat", "text")},
        {"cls_fuzzy3", new KNearestFuzzyClassifier(reader, null, a, null, 3, "cat", "text")},
        {"cls_perceptron", new BooleanPerceptronClassifier(reader, a, null, 1, null, "bool", "text")},
      };
      for (Object[] c : classifiers) {
        @SuppressWarnings("unchecked")
        Classifier<Object> classifier = (Classifier<Object>) c[1];
        SweepMicro.measure((String) c[0], () -> {
          for (String input : inputs) {
            SweepMicro.sink += classifier.assignClass(input).assignedClass().hashCode();
          }
          return inputs.size();
        });
      }
      SweepMicro.measure("cls_perceptron_train", () -> {
        SweepMicro.sink +=
            new BooleanPerceptronClassifier(reader, a, null, 1, null, "bool", "text").hashCode();
        return 1;
      });
    }
  }
}
