//! M13 T13.4a's taxonomy pair (the Java twin is
//! `benchmarks/micro/java/TaxonomyMicro.java`) over 20,000 labels
//! `dim<i%10> / l<(i/10)%50> / v<i>` (20,511 categories with their
//! ancestors). `taxo_add_new`: a fresh `DirectoryTaxonomyWriter` (default
//! cache of 4000) adding every label, units: labels; `taxo_add_cached`: the
//! same labels again on a writer whose unbounded cache holds them all, units:
//! labels; `taxo_add_lookup`: the same labels again on a writer reopened over
//! the committed taxonomy with a cache of 1000 that is never filled, so most
//! adds look the category up in the writer's near-real-time reader, units:
//! labels; `taxo_ordinal`: `DirectoryTaxonomyReader.getOrdinal` of every
//! label in order (its 4000-entry cache misses each), units: lookups;
//! `taxo_path`: `getPath` of every ordinal, units: lookups.

use std::hint::black_box;
use std::time::Duration;

use lucene_index::deletion_policy::OpenMode;
use lucene_search::facets::taxonomy::writercache::LruTaxonomyWriterCache;
use lucene_search::facets::taxonomy::{
    DirectoryTaxonomyReader, DirectoryTaxonomyWriter, FacetLabel, TaxonomyReader, TaxonomyWriter,
};
use lucene_store::ByteBuffersDirectory;

use super::measure;

fn labels() -> Vec<FacetLabel> {
    (0..20000)
        .map(|i| {
            FacetLabel::new([
                format!("dim{}", i % 10),
                format!("l{}", (i / 10) % 50),
                format!("v{i}"),
            ])
            .unwrap()
        })
        .collect()
}

pub fn bench_taxonomy(w: Duration, m: Duration) {
    let labels = labels();

    measure("taxo_add_new", w, m, || {
        let d = ByteBuffersDirectory::new();
        let tw = DirectoryTaxonomyWriter::open(&d).unwrap();
        for l in &labels {
            black_box(tw.add_category(l).unwrap());
        }
        tw.close().unwrap();
        labels.len() as u64
    });

    let dir = ByteBuffersDirectory::new();
    {
        let cache = Box::new(LruTaxonomyWriterCache::new(i32::MAX));
        let tw = DirectoryTaxonomyWriter::with_cache(&dir, OpenMode::Create, cache).unwrap();
        for l in &labels {
            tw.add_category(l).unwrap();
        }
        measure("taxo_add_cached", w, m, || {
            for l in &labels {
                black_box(tw.add_category(l).unwrap());
            }
            labels.len() as u64
        });
        tw.close().unwrap();
    }

    {
        let cache = Box::new(LruTaxonomyWriterCache::new(1000));
        let tw = DirectoryTaxonomyWriter::with_cache(&dir, OpenMode::Append, cache).unwrap();
        tw.set_cache_misses_until_fill(i32::MAX).unwrap();
        measure("taxo_add_lookup", w, m, || {
            for l in &labels {
                black_box(tw.add_category(l).unwrap());
            }
            labels.len() as u64
        });
        tw.close().unwrap();
    }

    let r = DirectoryTaxonomyReader::open(&dir).unwrap();
    measure("taxo_ordinal", w, m, || {
        for l in &labels {
            black_box(r.get_ordinal(l).unwrap());
        }
        labels.len() as u64
    });
    let size = r.get_size().unwrap();
    measure("taxo_path", w, m, || {
        for o in 0..size {
            black_box(r.get_path(o).unwrap().len());
        }
        size as u64
    });
}
