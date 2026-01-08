use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use curve25519_dalek::{
    constants::RISTRETTO_BASEPOINT_POINT as G,
    ecdlp::{self, ECDLPArguments, ECDLPTables},
    Scalar,
};
use rand::Rng;
use std::{path::Path, time::Duration, thread};

pub fn ecdlp_bench(c: &mut Criterion) {
    let avail = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    if !Path::new("ecdlp_table.bin").exists() {
        let tables = ECDLPTables::generate_par(26, avail).unwrap();
        tables.write_to_file("ecdlp_table.bin").unwrap();
    }

    let tables = ECDLPTables::load_from_file(26, "ecdlp_table.bin").unwrap();
    let view = tables.view();

    const TEST_VAL: u64 = 1u64 << 46;

    for &t in &[1usize, 2, 4, 8, 16] {
        if t > avail {
            continue;
        }

        c.bench_function(&format!("par fast ecdlp T={t}"), |b| {
            let num: u64 = TEST_VAL;
            let point = Scalar::from(num) * G;
            b.iter(|| {
                let res = ecdlp::par_decode(
                    &view,
                    black_box(point),
                    ECDLPArguments::new_with_range(0, 1 << 63).n_threads(t),
                );
                assert_eq!(res, Some(num as i64));
            });
        });

        c.bench_function(&format!("par fast ecdlp T={t} scalar"), |b| {
            let num: u64 = TEST_VAL;
            let point = Scalar::from(num) * G;
            b.iter(|| {
                let res = ecdlp::par_decode_scalar(
                    &view,
                    black_box(point),
                    ECDLPArguments::new_with_range(0, 1 << 63).n_threads(t),
                );
                assert_eq!(res, Some(num as i64));
            });
        });
    }

    c.bench_function("fast ecdlp find 0", |b| {
        let num: u64 = 0;
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 47)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 47));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(
        &format!("fast ecdlp for number < {} T=2", (1i64 << 47)),
        |b| {
            let num = rand::thread_rng().gen_range(0u64..(1 << 47));
            let point = Scalar::from(num) * G;
            b.iter(|| {
                let res = ecdlp::par_decode(
                    &view,
                    black_box(point),
                    ECDLPArguments::new_with_range(0, 1 << 48).n_threads(1),
                );
                assert_eq!(res, Some(num as i64));
            });
        },
    );

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 44)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 44));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 43)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 43));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 26)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 26));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 27)), |b| {
        let num = rand::thread_rng().gen_range(0u64..(1 << 27));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });
}

fn bench_table_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Table Generation");

    // Configure the benchmark for faster execution
    // Reduce to minimum samples
    group.sample_size(10);
    // Less measurement time
    group.measurement_time(Duration::from_secs(5));

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    for l1 in [13, 18, 21].iter() {
        group.bench_with_input(BenchmarkId::new("Sequential", l1), l1, |b, &l1| {
            b.iter(|| ECDLPTables::generate(l1).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("Parallel", l1), l1, |b, &l1| {
            b.iter(|| ECDLPTables::generate_par(l1, n_threads).unwrap());
        });
    }

    group.finish();
}

criterion_group!(ecdlp, ecdlp_bench);
criterion_group!(tables, bench_table_generation);
criterion_main!(ecdlp, tables);