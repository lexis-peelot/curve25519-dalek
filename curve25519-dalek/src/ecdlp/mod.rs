//! # Elliptic Curve Discrete Logarithm Problem (ECDLP)
//!
//! This file enables the decoding of integers from [`RistrettoPoint`]s. As this requires
//! a bruteforce operation, this can take quite a long time (multiple seconds) for large input spaces.
//!
//! The algorithm depends on a constant `L1` parameter, which enables changing the space/time tradeoff.
//! To use a given `L1` constant, you need to generate a precomputed tables file, or download a pre-generated one.
//! The resulting file may be quite big, which is why it is recommended to load it at runtime using [`ecdlp::ECDLPTablesFile`].
//!
//! The algorithm works for any range, but keep in mind that the time the algorithm takes grows exponentially: with the same
//! precomputed tables, decoding an `n+1`-bit integer will take 2x as long as an `n`-bit integer. By default, unless the "pseudo"
//! constant time mode is enabled, integers which are closer to the start of the range will be found exponentially faster: for
//! example, integers in the `n-1`-bit first half of the `n`-bit decoding range will take 1/2 of the time compared to the second half,
//! and numbers near the very beginning will be found almost immediately.
//!
//! # Space / Time tradeoff and benchmarks
//!
//! To choose an `L1` constant, you may to see benchmark performance and tables size [here](ecdlp_perf.md).
//!
//! # Table generation
//!
//! For now, table generation can be done using the `gen_t1_t2` test, or using the unstable [`ecdlp::generation`] module.
//!
//! # Constant time
//!
//! This algorithm cannot be constant-time because of hashmap lookups. However a "pseudo"
//! constant time mode is implemented which lets the algorithm continue to run even when it
//! has found the answer.
//!
//! # Example
//!
//! Decoding a 48bit number using a L1=26 precomputed tables file.
//! ```no_run
//! use curve25519_dalek::{
//!     constants::{RISTRETTO_BASEPOINT_POINT as G},
//!     ecdlp::{ECDLPTables, decode, ECDLPArguments},
//!     Scalar,
//!     RistrettoPoint,
//! };
//!
//! let precomputed_tables = ECDLPTables::load_from_file(26, "ecdlp_table_26.bin")
//!     .unwrap();
//!
//! let num = 258383831730230u64;
//! let to_decode = Scalar::from(num) * G;
//!
//! assert_eq!(
//!     decode(&precomputed_tables.view(), to_decode, ECDLPArguments::new_with_range(0, 1 << 48)),
//!     Some(num as i64)
//! );
//! ```

// Notes of the ECDLP implementation.
mod ecdlp_notes {
    //! The algorithm implemented here is BSGS (Baby Step Giant Step), and the implementation
    //! details are based on [Solving Small Exponential ECDLP in EC-based Additively Homomorphic Encryption and Applications][fast-ecdlp-paper].
    //!
    //! The gist of BSGS goes as follows:
    //! - Our target point, which we want to decode, represents an integer in the range \[0, 2^(L1 + L2)\].
    //! - We have a T1 hash table, where the key is the curve point and value is the decoded
    //!   point. T1 = <i * G => i | i in \[1, 2^l1\]>
    //! - We have a T2 linear table (an array), where T2 = \[j * 2^l1 * G | j in \[1, 2^l2\]\]
    //! - For each j in 0..2^l2
    //!   Compute the difference between T2\[j\] and the target point
    //!   if let Some(i) = T1.get(the difference) => the decoded integer is j * 2^L1 + i.
    //!
    //! On top of this regular BSGS algorithm, we add the following optimizations:
    //! - Batching. The paper uses a tree-based Montgomery trick - instead, we use the batched
    //!   inversion which is implemented in FieldElement.
    //! - T1 only contains the truncated x coordinates. The table uses Cuckoo hashing, and
    //!   the hash of a point is directly just a subset of the bytes of the point.
    //! - We need a canonical encoding of a point before any hashmap lookup: this means that
    //!   we must work with affine coordinates. Addition of affine Montgomery points requires
    //!   less inversions than Edwards points, so we use that instead.  
    //! - Using the fact -(x, y) = (x, -y) on the Montgomery curve, we can shift the inputs so
    //!   that we only need half of T1 and T2 and half of the modular inversions.
    //! - The L2 constant has been fixed here, because we can just shift the input after every
    //!   batch. This means that L2 has a constant size of about 16Ko, which is preferable
    //!   to >100Mo when L2 = 22, for example. This results in slightly more modular inversions,
    //!   however this has no visible impact on performance. Shifting the inputs like this
    //!   also means that we support arbitrary decoding ranges for a given constant tables file.
    //!
    //! Note: We are dealing with a curve which has cofactors; as such, we need to multiply
    //! by the cofactor before running ECDLP to clear it and guarantee a canonical encoding of our points.
    //! The tables also need to be based on `num * cofactor` to match.
    //!
    //! [fast-ecdlp-paper]: https://eprint.iacr.org/2022/1573
}

mod affine_montgomery;
mod scheduler;
mod table;

use crate::{
    RistrettoPoint, Scalar,
    constants::{MONTGOMERY_A_NEG, RISTRETTO_BASEPOINT_POINT as G},
    field::FieldElement,
};
use core::{
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
};

pub use affine_montgomery::AffineMontgomeryPoint;
pub use scheduler::*;
pub use table::*;

use table::{BATCH_SIZE, L2};

/// A trait to represent progress report functions.
/// It is auto-implemented on any `F: Fn(f64) -> ControlFlow<()>`.
pub trait ProgressReportFunction {
    /// Run the progress report function.
    fn report(&self, progress: f64) -> ControlFlow<()>;
}
impl<F: Fn(f64) -> ControlFlow<()>> ProgressReportFunction for F {
    #[inline(always)]
    fn report(&self, progress: f64) -> ControlFlow<()> {
        self(progress)
    }
}
/// The Noop (no operation) report function. It does nothing and will never break.
pub struct NoopReportFn;
impl ProgressReportFunction for NoopReportFn {
    #[inline(always)]
    fn report(&self, _progress: f64) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// Builder for the ECDLP algorithm parameters.
pub struct ECDLPArguments<R: ProgressReportFunction = NoopReportFn> {
    range_start: i64,
    range_end: i64,
    pseudo_constant_time: bool,
    n_threads: usize,
    progress_report_function: R,
}

impl ECDLPArguments<NoopReportFn> {
    /// Creates a new `ECDLPArguments` with default arguments, to run on a specific range.
    pub fn new_with_range(range_start: i64, range_end: i64) -> Self {
        Self {
            range_start,
            range_end,
            pseudo_constant_time: false,
            progress_report_function: NoopReportFn,
            n_threads: 1,
        }
    }
}

impl<F: ProgressReportFunction> ECDLPArguments<F> {
    /// Enable the "pseudo constant-time" mode. This means that the algorithm will not stop
    /// once it has found the answer. Keep in mind that **this is not actually constant-time**,
    /// in fact, the algorithm cannot be constant-time because it relies on hashmap lookups.
    /// This setting is also useful for benchmarking, as any input will result in roughly the same
    /// execution time.
    pub fn pseudo_constant_time(self, pseudo_constant_time: bool) -> Self {
        Self {
            pseudo_constant_time,
            ..self
        }
    }

    /// Sets the progress report function.
    ///
    /// This function will be periodically called when the algorithm is running.
    /// The `progress` argument represents the current progress, from `0.0` to `1.0`.
    /// Returning `ControlFlow::Break(())` will stop the algorithm.
    ///
    /// Please keep in mind that this report function should not take too long or nuke
    /// the cache, as it would impact the performance of the algorithm.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use curve25519_dalek::ecdlp::ECDLPArguments;
    /// use std::ops::ControlFlow;
    ///
    /// let ecdlp_args = ECDLPArguments::new_with_range(0, 1 << 48)
    ///     .progress_report_function(|_progress| {
    ///         // do something with `progress`
    ///         ControlFlow::Continue(())
    ///     });
    /// ```
    pub fn progress_report_function<R: ProgressReportFunction>(
        self,
        progress_report_function: R,
    ) -> ECDLPArguments<R> {
        ECDLPArguments {
            progress_report_function,
            range_start: self.range_start,
            range_end: self.range_end,
            pseudo_constant_time: self.pseudo_constant_time,
            n_threads: self.n_threads,
        }
    }

    /// Configures the number of threads used.
    /// This only affects the execution of the [`par_decode`] function.
    ///
    /// # Example
    ///
    /// ```
    /// use curve25519_dalek::ecdlp::ECDLPArguments;
    ///
    /// let n_threads = std::thread::available_parallelism()
    ///     .expect("cannot get available parallelism")
    ///     .get();
    /// let ecdlp_args = ECDLPArguments::new_with_range(0, 1 << 48)
    ///     .n_threads(n_threads);
    /// ```
    pub fn n_threads(self, n_threads: usize) -> Self {
        Self { n_threads, ..self }
    }
}

/// Offset calculations common to [`par_decode`] and [`decode`].
fn decode_prep<R: ProgressReportFunction>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: &ECDLPArguments<R>,
    n_threads: usize,
    thread_i: usize, // Add thread_i parameter
) -> (i64, RistrettoPoint, usize) {
    let amplitude = (args.range_end - args.range_start).max(0);

    let offset = args.range_start
        + ((1 << (L2 - 1)) << precomputed_tables.get_l1())
        + (1 << (precomputed_tables.get_l1() - 1));

    // Calculate thread-specific offset adjustment
    let thread_scalar_offset = if n_threads > 1 {
        // Divide the range into n_threads parts and adjust offset for this thread
        (amplitude / n_threads as i64) * thread_i as i64
    } else {
        0
    };

    // Adjust the normalized point for this specific thread
    let normalized =
        point - RistrettoPoint::mul_base(&i64_to_scalar(offset + thread_scalar_offset));

    let j_end = (amplitude >> precomputed_tables.get_l1()) as usize;
    let divceil = |a: usize, b: usize| a.div_ceil(b);

    // Calculate appropriate num_batches for this thread
    let thread_j_end = if n_threads > 1 {
        j_end / n_threads
    } else {
        j_end
    };

    let num_batches = divceil(thread_j_end, 1 << L2);

    (offset + thread_scalar_offset, normalized, num_batches)
}

/// Returns an iterator of batches for a given thread. Common to [`par_decode`] and [`decode`].
/// Iterator item is (index, j_start, target_montgomery, progress).
fn make_point_iterator(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {

    // clear the cofactor, we want the repr to be canonical
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
    let els_per_batch = 1u64 << (L2 + precomputed_tables.get_l1());

    let batch_step = -(els_per_batch as i64);
    let [target_montgomery, batch_step_montgomery] = AffineMontgomeryPoint::from_points(
        [
            &normalized.0,
            &(i64_to_scalar(batch_step) * G).0.mul_by_cofactor()
        ]
    );

    struct BatchedIterator {
        current_batch: [AffineMontgomeryPoint; 4],
        step_x4: AffineMontgomeryPoint,
        batch_idx: usize,
        j: usize,
        num_batches: usize,
    }

    impl Iterator for BatchedIterator {
        type Item = (usize, usize, AffineMontgomeryPoint, f64);

        fn next(&mut self) -> Option<Self::Item> {
            if self.j >= self.num_batches {
                return None;
            }

            // Refill batch when we've consumed all 4
            if self.batch_idx >= 4 && self.j < self.num_batches {
                // Use 4-way SIMD operation to advance all points by 4
                self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct(
                    &self.current_batch,
                    &self.step_x4
                );
                self.batch_idx = 0;
            }

            let result = (
                0, // index (not used in ECDLP)
                self.j * (1 << L2), // j_start
                self.current_batch[self.batch_idx],
                self.j as f64 / self.num_batches as f64
            );

            self.batch_idx += 1;
            self.j += 1;

            Some(result)
        }
    }

    // Initialize first 4 points
    let p0 = target_montgomery;
    let p1 = p0.addition_not_ct(&batch_step_montgomery);
    let p2 = p1.addition_not_ct(&batch_step_montgomery);
    let p3 = p2.addition_not_ct(&batch_step_montgomery);

    // Pre-compute 4*step
    let step_x4 = batch_step_montgomery.addition_not_ct(&batch_step_montgomery)
        .addition_not_ct(&batch_step_montgomery)
        .addition_not_ct(&batch_step_montgomery);

    BatchedIterator {
        current_batch: [p0, p1, p2, p3],
        step_x4,
        batch_idx: 0,
        j: 0,
        num_batches,
    }
}

/// Decode a [`RistrettoPoint`] to the represented integer.
/// This may take a long time, so if you are running on an event-loop such as `tokio`, you
/// should wrap this in a `tokio::block_on` task.
pub fn decode<R: ProgressReportFunction>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    let (offset, normalized, num_batches) = decode_prep(precomputed_tables, point, &args, 1, 0);
    let point_iter = make_point_iterator(precomputed_tables, normalized, num_batches);

    let (t2_cache, t2_cache_alpha) = prepare_t2_cache(precomputed_tables);

    fast_ecdlp(
        precomputed_tables,
        normalized,
        point_iter,
        args.pseudo_constant_time,
        args.progress_report_function,
        &t2_cache,
        &t2_cache_alpha,
    )
    .map(|v| v as i64 + offset)
}

/// Prepares the T2 cache for fast ECDLP.
#[inline]
fn prepare_t2_cache(precomputed_tables: &ECDLPTablesFileView<'_>) -> ([AffineMontgomeryPoint; BATCH_SIZE], [FieldElement; BATCH_SIZE]) {
    // Pre compute the T2 cache
    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [MONTGOMERY_A_NEG; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        let mut points_u = [FieldElement::ZERO; BATCH_SIZE];

        for (i, (cache, u)) in t2_cache
            .iter_mut()
            .zip(points_u.iter_mut())
            .enumerate()
        {
            let point = t2_table.index(i);
            *u = point.u;
            *cache = point;
        }

        // Compute alphas = A - T2[j]_x
        FieldElement::batch_subtract_n(&mut t2_cache_alpha, &points_u);
    }

    (t2_cache, t2_cache_alpha)
}

/// Decode a [`RistrettoPoint`] to the represented integer, in parallel.
/// This uses [`std::thread`] as a threading primitive, and as such, it is only available when the `std` feature is enabled.
/// This may take a long time, so if you are running on an event-loop such as `tokio`, you
/// should wrap this in a `tokio::block_on` task.
#[cfg(feature = "std")]
pub fn par_decode<S, R>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64>
where
    S: Scheduler,
    R: ProgressReportFunction + Sync,
{
    let end_flag = AtomicBool::new(false);

    // Pre compute the T2 cache
    let (t2_cache, t2_cache_alpha) = prepare_t2_cache(precomputed_tables);

    S::scope(|s| {
        let handles = (0..args.n_threads)
            .map(|thread_i| {
                let (offset, normalized, num_batches) =
                    decode_prep(precomputed_tables, point, &args, args.n_threads, thread_i);

                let end_flag = &end_flag;

                let progress_report = &args.progress_report_function;
                let progress_report = |progress| {
                    if !args.pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
                        ControlFlow::Break(())
                    } else {
                        let ret = progress_report.report(progress);
                        if ret.is_break() {
                            // we need to tell the other threads that the user requested to stop
                            end_flag.store(true, Ordering::SeqCst);
                        }
                        ret
                    }
                };

                s.spawn(move || {
                    let point_iter =
                        make_point_iterator(precomputed_tables, normalized, num_batches);
                    let res = fast_ecdlp(
                        precomputed_tables,
                        normalized,
                        point_iter,
                        args.pseudo_constant_time,
                        progress_report,
                        &t2_cache,
                        &t2_cache_alpha,
                    );

                    if !args.pseudo_constant_time && res.is_some() {
                        end_flag.store(true, Ordering::SeqCst);
                    }

                    res.map(|v| offset + v as i64)
                })
            })
            .collect::<Vec<_>>();

        let mut res = None;
        for el in handles {
            let v = el.join().expect("child thread panicked");
            res = res.or(v);
        }

        res
    })
}

fn is_point_equal(v: i64, target: &RistrettoPoint) -> bool {
   i64_to_scalar(v) * G == *target
}

fn fast_ecdlp(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    target_point: RistrettoPoint,
    point_iterator: impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)>,
    pseudo_constant_time: bool,
    progress_report: impl ProgressReportFunction,
    t2_cache: &[AffineMontgomeryPoint; BATCH_SIZE],
    t2_cache_alpha: &[FieldElement; BATCH_SIZE],
) -> Option<u64> {
    let t1_table = precomputed_tables.get_t1();

    let mut found = None;
    let mut consider_candidate = |m| {
        let equal = is_point_equal(m, &target_point);
        if equal {
            found = found.or(Some(m as u64));
        }

        equal
    };

    // Precompute origins for batching
    let mut alphas_origin = [FieldElement::ZERO; BATCH_SIZE];
    let mut batch_origin = [FieldElement::ZERO; BATCH_SIZE];
    let mut lambdas_origin = [FieldElement::ZERO; BATCH_SIZE];

    for i in 0..BATCH_SIZE {
        alphas_origin[i] = t2_cache_alpha[i];

        let t2_point = &t2_cache[i];
        batch_origin[i] = t2_point.u;
        lambdas_origin[i] = t2_point.v;
    }

    // Also prepare negated lambdas
    let mut lambdas_neg_origin = lambdas_origin;
    FieldElement::batch_negate(&mut lambdas_neg_origin);

    'outer: for (index, j_start, target_montgomery, progress) in point_iterator {
        // amortize the potential cost of the report function
        if index % BATCH_SIZE == 0 {
            if let ControlFlow::Break(_) = progress_report.report(progress) {
                break 'outer;
            }
        }

        // Case 0: target is 0. Has to be handled separately.
        let j_start_shifted = (j_start as i64) << precomputed_tables.get_l1();
        if target_montgomery.is_identity_not_ct() {
            consider_candidate(j_start_shifted);
            if !pseudo_constant_time {
                break 'outer;
            }
        }

        // Case 2: j=0. Has to be handled separately.
        if t1_table
            .lookup(&target_montgomery.u.to_bytes(), |i| {
                consider_candidate(j_start_shifted + i as i64)
                    || consider_candidate(j_start_shifted - i as i64)
            })
            .is_some()
            && !pseudo_constant_time
        {
            break 'outer;
        }

        let mut batch = batch_origin;
        FieldElement::batch_subtract(&mut batch, &target_montgomery.u);

        // Z = T2[j]_x - Pm_x
        let mut has_zero = false;
        for (i, batch) in batch.iter().enumerate() {
            if batch.is_zero_not_ct() {
                let j = i + 1;
                // Case 1: (Montgomery addition) exceptional case when T2[j] = Pm.
                // m1 = j * 2^L1, m2 = -j * 2^L1
                let found =
                    consider_candidate((j_start as i64 + j as i64) << precomputed_tables.get_l1())
                        || consider_candidate(
                            (j_start as i64 - j as i64) << precomputed_tables.get_l1(),
                        );

                // should always be found here
                if !pseudo_constant_time && found {
                    break 'outer;
                }

                has_zero = true;
            }
        }

        // nu = Z^-1
        if has_zero {
            FieldElement::invert_batch(&mut batch);
        } else {
            FieldElement::invert_batch_checked(&mut batch);
        }

        let mut alphas = alphas_origin;
        FieldElement::batch_subtract(&mut alphas, &target_montgomery.u);

        // lambda = (T2[j]_y - Pm_y) * nu
        // Q_x = lambda^2 - A - T2[j]_x - Pm_x
        let mut lambdas = lambdas_origin;
        FieldElement::batch_sub_mul_square_add(&mut lambdas, &target_montgomery.v, &batch, &alphas);

        for (batch_i, qx) in lambdas.iter().enumerate() {
            let j = batch_i + 1;
            // Case 3: general case, negative j.
            let j_start_shifted = (j_start as i64 - j as i64) << precomputed_tables.get_l1();
            if t1_table
                .lookup(&qx.to_bytes(), |i| {
                    consider_candidate(j_start_shifted + i as i64)
                        || consider_candidate(j_start_shifted - i as i64)
                })
                .is_some()
            {
                // m1 = -j * 2^L1 + i, m2 = -j * 2^L1 - i
                if !pseudo_constant_time {
                    break 'outer;
                }
            }
        }

        // Recompute nu for the positive j case
        // lambda = (T2[j]_y - Pm_y) * nu
        // Q_x = lambda^2 - A - T2[j]_x - Pm_x
        let mut lambdas = lambdas_neg_origin;
        FieldElement::batch_sub_mul_square_add(&mut lambdas, &target_montgomery.v, &batch, &alphas);

        for (batch_i, qx) in lambdas.iter().enumerate() {
            let j = batch_i + 1;
            // Case 4: general case, positive j.
            let j_start_shifted = (j_start as i64 + j as i64) << precomputed_tables.get_l1();
            if t1_table
                .lookup(&qx.to_bytes(), |i| {
                    consider_candidate(j_start_shifted + i as i64)
                        || consider_candidate(j_start_shifted - i as i64)
                })
                .is_some()
            {
                // m1 = j * 2^L1 + i, m2 = j * 2^L1 - i
                if !pseudo_constant_time {
                    break 'outer;
                }
            }
        }
    }

    found
}

// FIXME(upstrean): should be an impl From<i64> for Scalar
#[inline]
fn i64_to_scalar(n: i64) -> Scalar {
    if n >= 0 {
        Scalar::from(n as u64)
    } else {
        -&Scalar::from((-n) as u64)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{Arc, Mutex},
    };

    use super::*;
    use rand::{Rng, rng};

    const L1: usize = 26;

    // Necessary for one ECDLP tables allocation only
    static TABLES: Mutex<Option<Arc<ECDLPTables>>> = Mutex::new(None);

    fn read_or_gen_tables() -> Arc<ECDLPTables> {
        let mut tables = TABLES.lock().expect("acquire tables lock");
        if let Some(v) = tables.as_ref().cloned() {
            return v;
        }

        let inner = if !Path::new("ecdlp_table.bin").exists() {
            let tables = ECDLPTables::generate(L1).unwrap();
            tables.write_to_file("ecdlp_table.bin").unwrap();
            tables
        } else {
            ECDLPTables::load_from_file(L1, "ecdlp_table.bin").unwrap()
        };

        let t = Arc::new(inner);
        *tables = Some(t.clone());

        t
    }

    #[test]
    fn test_ecdlp_cofactors() {
        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in (0..(1u64 << 48)).step_by(1 << L1).take(1 << 12) {
            let delta = rng().random_range(0..(1 << L1));

            let num = i + delta;
            let point = RistrettoPoint::mul_base(&Scalar::from(num));

            // take a random point from the coset4
            let coset_i = rng().random_range(0..4);
            let point = point.coset4()[coset_i];

            let res = decode(
                &view,
                RistrettoPoint(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        }
    }

    #[test]
    fn test_ecdlp_decode() {
        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in (0..(1u64 << 48)).step_by(1 << L1).take(1 << 12) {
            let num = i;
            let mut point = RistrettoPoint::mul_base(&Scalar::from(num));

            if rng().random() {
                // do a round of compression/decompression to mess up the Z and Ts
                // & ecdlp will need to clear the cofactor
                point = point.compress().decompress().unwrap();
            }

            let res = decode(&view, point, ECDLPArguments::new_with_range(0, 1 << 48));
            assert_eq!(res, Some(num as i64));
        }
    }

    #[test]
    fn test_ecdlp_par_decode() {
        let base: u64 = (1 << 48) / 16;

        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in 0..17 {
            let value = base * i;

            let point = RistrettoPoint::mul_base(&Scalar::from(value));
            let res = par_decode::<DefaultScheduler, _>(
                &view,
                point,
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .n_threads(4)
                    .pseudo_constant_time(true),
            );
            assert_eq!(res, Some(value as i64));
        }
    }

    #[test]
    fn test_table_par() {
        // Measure parallel generation time
        let tables_par = ECDLPTables::generate_par::<DefaultScheduler>(
            18,
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
        )
        .unwrap();

        // Measure sequential generation time
        let tables_seq = ECDLPTables::generate(18).unwrap();

        // Verify both tables are identical
        assert_eq!(
            tables_seq.as_slice(),
            tables_par.as_slice(),
            "Sequential and parallel generated tables should be identical"
        );
    }
}
