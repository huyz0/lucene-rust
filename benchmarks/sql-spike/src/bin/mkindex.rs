fn main() {
    let a: Vec<String> = std::env::args().collect();
    sql_spike::data::build(
        &a[1],
        a[2].parse().unwrap(),
        a.get(3).map_or(4, |s| s.parse().unwrap()),
    );
}
