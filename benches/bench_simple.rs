use criterion::{criterion_group, criterion_main, Criterion};

fn simple_benchmark(c: &mut Criterion) {
    c.bench_function("Simple Loop", |b| {
        b.iter(|| {
            let sum: u64 = (1..1000).sum();
            criterion::black_box(sum);
        });
    });
}

criterion_group!(benches, simple_benchmark);
criterion_main!(benches);
