use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use curve25519_dalek::field::FieldElement;

const A: FieldElement = FieldElement::from_bytes(&[
    0x1a, 0x0e, 0x97, 0x8a, 0x90, 0xf6, 0x62, 0x2d, 
    0x37, 0x47, 0x02, 0x3f, 0x8a, 0xd8, 0x26, 0x4d,
    0xa7, 0x58, 0xaa, 0x1b, 0x88, 0xe0, 0x40, 0xd1,
    0x58, 0x9e, 0x7b, 0x7f, 0x23, 0x76, 0xef, 0x09,
]);

const B: FieldElement = FieldElement::from_bytes(&[
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f,
]);

macro_rules! bench_op {
    ($fn_name:ident, $group_name:expr, |$arr_ident:ident| $body:block) => {
        fn $fn_name(c: &mut Criterion) {
            let mut group = c.benchmark_group($group_name);

            let array = [A, A, A, A];
            group.bench_function(BenchmarkId::from_parameter(array.len()), |bench| {
                bench.iter(|| {
                    let mut $arr_ident = array;
                    $body
                });
            });

            group.finish();
        }
    };
}

// Auto-generate benches for add/mul/square using the macro
bench_op!(bench_batch_add, "batch_add", |arr| {
    FieldElement::batch_add(black_box(&mut arr), black_box(&B));
});

bench_op!(bench_batch_mul, "batch_mul", |arr| {
    let b = [B; 4];
    FieldElement::batch_mul(black_box(&mut arr), black_box(&b));
});

bench_op!(bench_batch_square, "batch_square", |arr| {
    FieldElement::batch_square(black_box(&mut arr));
});

bench_op!(bench_batch_subtract, "batch_subtract", |arr| {
    FieldElement::batch_subtract(black_box(&mut arr), black_box(&B));
});

// Serial

bench_op!(bench_subtract, "subtract", |arr| {
    for fe in arr.iter_mut() {
        *fe = black_box(black_box(&*fe) - black_box(&B));
    }
});

bench_op!(bench_add, "add", |arr| {
    for fe in arr.iter_mut() {
        *fe = black_box(black_box(&*fe) + black_box(&B));
    }
});

bench_op!(bench_mul, "mul", |arr| {
    for fe in arr.iter_mut() {
        *fe = black_box(black_box(&*fe) * black_box(&B));
    }
});

bench_op!(bench_square, "square", |arr| {
    for fe in arr.iter_mut() {
        *fe = black_box(&*fe).square();
    }
});

criterion_group!(
    field_benches,
    bench_batch_subtract,
    bench_subtract,
    bench_batch_add,
    bench_add,
    bench_batch_mul,
    bench_mul,
    bench_batch_square,
    bench_square,
);
criterion_main!(field_benches);
