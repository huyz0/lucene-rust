//! Java twin: `benchmarks/micro/java/StringMockResourceLoaderMicro.java`.
//! Run with `scripts/docker-test.sh bash scripts/bench-string-resource-loader.sh`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use lucene_analysis::factory::ResourceLoader;
use lucene_test_framework::util::string_mock_resource_loader::StringMockResourceLoader;

fn batch(loader: &StringMockResourceLoader) -> u64 {
    let mut sum = 0;
    for _ in 0..256 {
        let bytes = black_box(loader)
            .open_resource(black_box("ignored"))
            .unwrap();
        for byte in black_box(bytes) {
            sum += u64::from(byte);
        }
    }
    black_box(sum)
}

fn run(loader: &StringMockResourceLoader, ms: u64) -> (u64, Duration) {
    let start = Instant::now();
    let mut batches = 0;
    while start.elapsed() < Duration::from_millis(ms) {
        black_box(batch(loader));
        batches += 1;
    }
    (batches, start.elapsed())
}

fn main() {
    for repeat in [8, 256] {
        let text = "caf\u{e9} \u{1f600}\0\n".repeat(repeat);
        let retained_capacity = text.capacity();
        let loader = StringMockResourceLoader::new(text);
        let name = format!("string_resource_loader_{repeat}");
        // Requested storage, excluding allocator metadata (as on the Java side).
        let retained = std::mem::size_of_val(&loader) + retained_capacity;
        let bytes = loader.open_resource("ignored").unwrap();
        let open_bytes = std::mem::size_of_val(&bytes) + bytes.capacity();
        println!("{name}_retained_bytes\t{retained}\t1");
        println!("{name}_open_bytes\t{open_bytes}\t1");
        let digest = batch(&loader);
        println!("#check\t{name}\t{digest}\t256");
        run(&loader, 1000);
        let (batches, elapsed) = run(&loader, 1500);
        let opens = batches * 256;
        println!(
            "{name}\t{:.3}\t{opens}",
            elapsed.as_nanos() as f64 / opens as f64
        );
    }
}
