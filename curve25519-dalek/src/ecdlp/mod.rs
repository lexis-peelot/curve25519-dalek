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
//! For now, table generation can be done using the `gen_t1_t2` test, or using the unstable [`ecdlp::table_generation`] module.
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

pub(crate) mod simd_types;
mod affine_montgomery;
mod table;
mod field_simd;

use crate::{
    RistrettoPoint, Scalar, constants::MONTGOMERY_A_NEG, constants::RISTRETTO_BASEPOINT_POINT as G,
    field::FieldElement,
};
use affine_montgomery::AffineMontgomeryPoint;
use core::{
    ops::ControlFlow,
    sync::atomic::{AtomicBool, Ordering},
};
use cfg_if::cfg_if;

pub use table::{
    ECDLPTablesFileView, NoOpProgressTableGenerationReportFunction,
    ProgressTableGenerationReportFunction, ReportStep, table_generation,
};

use table::{BATCH_SIZE, L2};
use multiversion::multiversion;

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

/// A struct to ensure that the bytes are aligned on 32 bytes.
/// This is required for the table generation.
#[derive(Default, bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
#[repr(C, align(32))]
struct ForcedAlign32([u8; 32]);

/// The tables file is a big array of ForcedAlign32, which is a 32-byte aligned array of bytes.
/// Some bytes may be used as padding only.
/// This prevent using memory-mapped files, as the alignment is not guaranteed.
pub struct ECDLPTables {
    bytes: Vec<ForcedAlign32>,
    l1: usize,
    size: usize,
}

impl ECDLPTables {
    /// Get the expected final bytes size and number of vec elements in the tables.
    pub fn get_required_sizes(l1: usize) -> (usize, usize) {
        let size = table_generation::table_file_len(l1);
        let mut n = size / 32;
        if size % 32 != 0 {
            n += 1;
        }
        (size, n)
    }

    /// Create a new empty precomputed tables.
    pub fn empty(l1: usize) -> Self {
        let (size, n) = Self::get_required_sizes(l1);
        Self {
            l1,
            bytes: vec![Default::default(); n],
            size,
        }
    }

    /// Generate a new precomputed tables
    pub fn generate(l1: usize) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file(l1, zelf.as_mut_slice())?;

        Ok(zelf)
    }

    /// Generate a new precomputed tables, with multithreading
    pub fn generate_par(l1: usize, n_threads: usize) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file_par(l1, n_threads, zelf.as_mut_slice())?;

        Ok(zelf)
    }

    /// Generate a new precomputed tables with a progress report function.
    pub fn generate_with_progress_report<P: ProgressTableGenerationReportFunction>(
        l1: usize,
        p: P,
    ) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file_with_progress_report(l1, zelf.as_mut_slice(), p)?;

        Ok(zelf)
    }

    /// Generate a new precomputed tables with a progress report function, with multithreading.
    pub fn generate_with_progress_report_par<P: ProgressTableGenerationReportFunction + Sync>(
        l1: usize,
        n_threads: usize,
        p: P,
    ) -> std::io::Result<Self> {
        let mut zelf = Self::empty(l1);
        table_generation::create_table_file_with_progress_report_par(
            l1,
            n_threads,
            zelf.as_mut_slice(),
            p,
        )?;

        Ok(zelf)
    }

    /// Load the tables from a bytes slice.
    pub fn from_bytes(l1: usize, bytes: &[u8]) -> Self {
        let mut zelf = Self::empty(l1);
        zelf.as_mut_slice().copy_from_slice(bytes);

        zelf
    }

    /// Load the tables from a file.
    #[cfg(feature = "std")]
    pub fn load_from_file(l1: usize, path: &str) -> std::io::Result<Self> {
        use std::io::Read;
        let mut zelf = Self::empty(l1);

        let mut file = std::fs::File::open(path)?;
        file.read_exact(zelf.as_mut_slice())?;

        Ok(zelf)
    }

    /// Write the tables to a file.
    #[cfg(feature = "std")]
    pub fn write_to_file(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;
        file.write_all(self.as_slice())?;
        Ok(())
    }

    /// Get the tables as a slice of bytes.
    pub fn as_slice(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.bytes)[..self.size]
    }

    /// Get the tables a mutable slice of bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut bytemuck::cast_slice_mut(&mut self.bytes)[..self.size]
    }

    /// Get a view of the tables.
    pub fn view(&self) -> ECDLPTablesFileView<'_> {
        ECDLPTablesFileView::from_bytes(self.as_slice(), self.l1)
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


fn decode_prep<R: ProgressReportFunction>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: &ECDLPArguments<R>,
    n_threads: usize,
    thread_i: usize, // Add thread_i parameter
) -> (i64, RistrettoPoint, usize) {
    let amplitude = (args.range_end - args.range_start).max(0) / n_threads as i64;

    let offset = args.range_start + amplitude*thread_i as i64
        + ((1 << (L2 - 1)) << precomputed_tables.get_l1())
        + (1 << (precomputed_tables.get_l1() - 1));

    let normalized =
        point - RistrettoPoint::mul_base(&i64_to_scalar(offset));

    let j_end = ((amplitude as i64) >> precomputed_tables.get_l1()) as usize;
    let divceil = |a, b| (a + b - 1) / b;

    let num_batches = divceil(j_end, 1 << L2);

    (offset, normalized, num_batches)
}


/// Returns an iterator of batches for a given thread. Common to [`par_decode`] and [`decode`].
/// Iterator item is (index, j_start, target_montgomery, progress).
fn make_point_iterator(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {
    let thread_iter = (0..num_batches).map(move |j| {
        let progress = j as f64 / num_batches as f64;
        (j, j * (1 << L2), progress)
    });

    // clear the cofactor, we want the repr to be canonical
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());

    let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());

    // starting point for this thread
    let mut target_montgomery = AffineMontgomeryPoint::from(&normalized.0);
    let batch_step = -(els_per_batch as i64);
    let batch_step_montgomery =
        AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());

    thread_iter.map(move |(j, j_start, progress)| {
        let current = target_montgomery;

        target_montgomery =
            AffineMontgomeryPoint::addition_not_ct(&target_montgomery, &batch_step_montgomery);

        (j, j_start, current, progress)
    })
}

fn make_point_iterator_simd(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)> {
    // Apply same transformations as original make_point_iterator
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
    let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());
    
    let initial = AffineMontgomeryPoint::from(&normalized.0);
    let batch_step = -(els_per_batch as i64);
    let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
    
    struct OptimizedIterator {
        current_batch: [AffineMontgomeryPoint; 4],
        #[allow(dead_code)]
        step: AffineMontgomeryPoint,
        step_x4: AffineMontgomeryPoint,
        batch_idx: usize,
        j: usize,
        num_batches: usize,
    }
    
    impl Iterator for OptimizedIterator {
        type Item = (usize, usize, AffineMontgomeryPoint, f64);
        
        fn next(&mut self) -> Option<Self::Item> {
            if self.j >= self.num_batches {
                return None;
            }
            
            // Refill batch when we've consumed all 4
            if self.batch_idx >= 4 && self.j < self.num_batches {
                // Use 4-way SIMD operation to advance all points by 4
                self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
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
    let p0 = initial;
    let p1 = p0.addition_not_ct(&step);
    let p2 = p1.addition_not_ct(&step);
    let p3 = p2.addition_not_ct(&step);
    
    // Pre-compute 4*step
    let step_x4 = step.addition_not_ct(&step)
        .addition_not_ct(&step)
        .addition_not_ct(&step);
    
    OptimizedIterator {
        current_batch: [p0, p1, p2, p3],
        step,
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

    // Pre compute the T2 cache
    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache
            .iter_mut()
            .zip(t2_cache_alpha.iter_mut())
            .enumerate()
        {
            let point = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &point.u;
            *cache = point;
        }
    }

    fast_ecdlp(
        precomputed_tables,
        normalized,
        point_iter,
        args.pseudo_constant_time,
        &AtomicBool::new(false),
        args.progress_report_function,
        &t2_cache,
        &t2_cache_alpha,
    )
    .map(|v| v as i64 + offset)
}

/// Decode a [`RistrettoPoint`] to the represented integer, in parallel.
///
/// This implementation uses **manual parallelism via [`std::thread::scope`]** with
/// a fixed number of worker threads and *static work partitioning*.
/// Each worker processes disjoint chunks of the search space and cooperatively
/// terminates early using a shared atomic flag once a solution is found.
///
/// ### Platform support
/// This function depends on the Rust standard library (`std`) and is therefore
/// **not compatible with `#![no_std]` targets**, including WebAssembly
/// (`wasm32-unknown-unknown`) and bare-metal environments.
///
/// ### Cancellation semantics
/// Early termination is **cooperative**:
/// once a result is found, other threads observe a shared atomic flag and
/// exit at the next cancellation check. Threads are not forcibly interrupted,
/// so up to one in-flight chunk per worker may still complete.
///
/// ### Performance notes
/// Work is assigned statically to minimize synchronization and preserve cache
/// locality and SIMD efficiency.
///
/// ### Async runtimes
/// This function is **CPU-bound and blocking**. When called from an async runtime
/// (such as `tokio`), it should be executed inside a dedicated blocking task
/// (e.g. `tokio::task::spawn_blocking`) to avoid stalling the executor.
///
/// ### Alternatives
/// * For single-threaded execution or `no_std` compatibility, use [`decode`].
pub fn par_decode<R: ProgressReportFunction + Sync>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    // For 32-bit, use scalar implementation until 32-bit SIMD is properly implemented
    #[cfg(curve25519_dalek_bits = "32")]
    {
        return par_decode_scalar(precomputed_tables, point, args);
    }

    #[cfg(curve25519_dalek_bits = "64")]
    {
        use std::ops::ControlFlow;
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };

    let n_threads = args.n_threads.max(1);

    let chunk_size: u64 = 64 * 320_000_000_000_0
        / ((30 - precomputed_tables.get_l1()).max(0) * 2).max(1) as u64;

    let total_range = args.range_end as u64 - args.range_start as u64;
    let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;

    let end_flag = Arc::new(AtomicBool::new(false));

    // Precompute shared caches once (read-only afterwards)
    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache
            .iter_mut()
            .zip(t2_cache_alpha.iter_mut())
            .enumerate()
        {
            let p = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &p.u;
            *cache = p;
        }
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_threads);

        for thread_index in 0..n_threads {
            let end_flag = Arc::clone(&end_flag);

            // Borrow shared inputs within the scope
            let precomputed_tables = precomputed_tables;
            let args = &args;
            let t2_cache = &t2_cache;
            let t2_cache_alpha = &t2_cache_alpha;

            let point = point;

            handles.push(scope.spawn(move || -> Option<i64> {
                // Per-thread derived constants
                let t2_u_values: [FieldElement; BATCH_SIZE] = {
                    let mut u_values = [FieldElement::ZERO; BATCH_SIZE];
                    for i in 0..BATCH_SIZE {
                        u_values[i] = t2_cache[i].u;
                    }
                    u_values
                };
                let t2_vs: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| p.v.clone());
                let t2_vs_neg: [FieldElement; BATCH_SIZE] = t2_cache.map(|p| -&p.v);

                // Per-thread scratch (allocated once)
                let mut batch = [FieldElement::ZERO; BATCH_SIZE];
                let mut alphas = [FieldElement::ZERO; BATCH_SIZE];
                let mut qxs = [FieldElement::ZERO; BATCH_SIZE];
                let mut neg_qxs = [FieldElement::ZERO; BATCH_SIZE];

                // Static chunk scheduling: i, i+n_threads, ...
                for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                    if end_flag.load(Ordering::Relaxed) {
                        return None;
                    }

                    let start = args.range_start + (chunk_index as u64 * chunk_size) as i64;
                    let end = ((chunk_index as u64 + 1) * chunk_size)
                        .min(args.range_end as u64) as i64;

                    let chunk_args = ECDLPArguments {
                        range_start: start,
                        range_end: end,
                        n_threads: 1,
                        pseudo_constant_time: args.pseudo_constant_time,
                        progress_report_function: NoopReportFn, // TODO: global-scale progress
                    };

                    let (offset, normalized, num_batches) =
                        decode_prep(precomputed_tables, point, &chunk_args, 1, 0);

                    let progress_wrapper = |progress: f64| {
                        if !chunk_args.pseudo_constant_time && end_flag.load(Ordering::Relaxed) {
                            ControlFlow::Break(())
                        } else {
                            let result = args.progress_report_function.report(progress);
                            if result.is_break() {
                                end_flag.store(true, Ordering::SeqCst);
                            }
                            result
                        }
                    };

                    let point_iter =
                        make_point_iterator_simd(precomputed_tables, normalized, num_batches);

                    if let Some(res) = fast_ecdlp_simd(
                        precomputed_tables,
                        normalized,
                        point_iter,
                        chunk_args.pseudo_constant_time,
                        &end_flag,
                        progress_wrapper,
                        t2_cache_alpha,
                        &t2_u_values,
                        &t2_vs,
                        &t2_vs_neg,
                        &mut batch,
                        &mut alphas,
                        &mut qxs,
                        &mut neg_qxs,
                    ) {
                        end_flag.store(true, Ordering::SeqCst);
                        return Some(offset + res as i64);
                    }
                }

                None
            }));
        }

        // Join all threads; return first success
        let mut found: Option<i64> = None;
        for h in handles {
            match h.join() {
                Ok(res) if found.is_none() && res.is_some() => found = res,
                Ok(_) => {}
                Err(_) => {
                    end_flag.store(true, Ordering::SeqCst);
                }
            }
        }
        found
    })
    } // end cfg(curve25519_dalek_bits = "64")
}

#[allow(dead_code)]
fn make_point_iterator_simd_batched(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    normalized: RistrettoPoint,
    num_batches: usize,
) -> impl Iterator<Item = [(usize, usize, AffineMontgomeryPoint, f64); 4]> {
    let normalized = RistrettoPoint(normalized.0.mul_by_cofactor());
    let els_per_batch: u64 = 1u64 << (L2 + precomputed_tables.get_l1());
    
    let initial = AffineMontgomeryPoint::from(&normalized.0);
    let batch_step = -(els_per_batch as i64);
    let step = AffineMontgomeryPoint::from(&(i64_to_scalar(batch_step) * G).0.mul_by_cofactor());
    
    struct BatchedIterator {
        current_batch: [AffineMontgomeryPoint; 4],
        step: AffineMontgomeryPoint,
        step_x4: AffineMontgomeryPoint,
        j: usize,
        num_batches: usize,
    }
    
    impl Iterator for BatchedIterator {
        type Item = [(usize, usize, AffineMontgomeryPoint, f64); 4];
        
        fn next(&mut self) -> Option<Self::Item> {
            if self.j >= self.num_batches {
                return None;
            }
            
            // Build result array with current batch
            let mut result = [(0, 0, AffineMontgomeryPoint::identity(), 0.0); 4];
            let mut count = 0;
            
            for i in 0..4 {
                if self.j < self.num_batches {
                    result[i] = (
                        0,
                        self.j * (1 << L2),
                        self.current_batch[i],
                        self.j as f64 / self.num_batches as f64
                    );
                    self.j += 1;
                    count += 1;
                }
            }
            
            if count == 0 {
                return None;
            }
            
            // Advance all 4 points for next iteration
            self.current_batch = AffineMontgomeryPoint::batch_addition_not_ct_4way(
                &self.current_batch,
                &self.step_x4
            );
            
            Some(result)
        }
    }
    
    // Initialize first 4 points
    let p0 = initial;
    let p1 = p0.addition_not_ct(&step);
    let p2 = p1.addition_not_ct(&step);
    let p3 = p2.addition_not_ct(&step);
    
    // Pre-compute 4*step
    let step_x4 = step.addition_not_ct(&step)
        .addition_not_ct(&step)
        .addition_not_ct(&step);
    
    BatchedIterator {
        current_batch: [p0, p1, p2, p3],
        step,
        step_x4,
        j: 0,
        num_batches,
    }
}

pub fn par_decode_scalar<R: ProgressReportFunction + Sync>(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    point: RistrettoPoint,
    args: ECDLPArguments<R>,
) -> Option<i64> {
    use std::ops::ControlFlow;
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    let n_threads = args.n_threads.max(1);

    let chunk_size: u64 = 320_000_000_000_0
        / ((30 - precomputed_tables.get_l1()).max(0) * 2).max(1) as u64;

    let total_range = args.range_end as u64 - args.range_start as u64;
    let num_chunks = ((total_range + chunk_size - 1) / chunk_size) as usize;

    let end_flag = Arc::new(AtomicBool::new(false));

    // Precompute shared caches once (read-only afterwards)
    let mut t2_cache = [AffineMontgomeryPoint::identity(); BATCH_SIZE];
    let mut t2_cache_alpha = [FieldElement::ZERO; BATCH_SIZE];
    {
        let t2_table = precomputed_tables.get_t2();
        for (i, (cache, alpha)) in t2_cache
            .iter_mut()
            .zip(t2_cache_alpha.iter_mut())
            .enumerate()
        {
            let p = t2_table.index(i);
            *alpha = &MONTGOMERY_A_NEG - &p.u;
            *cache = p;
        }
    }

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(n_threads);

        for thread_index in 0..n_threads {
            let end_flag = Arc::clone(&end_flag);

            // Borrow shared inputs within the scope
            let precomputed_tables = precomputed_tables;
            let args = &args;
            let t2_cache = &t2_cache;
            let t2_cache_alpha = &t2_cache_alpha;

            let point = point;

            handles.push(scope.spawn(move || -> Option<i64> {
                for chunk_index in (thread_index..num_chunks).step_by(n_threads) {
                    if end_flag.load(Ordering::Relaxed) {
                        return None;
                    }

                    let start = args.range_start + (chunk_index as u64 * chunk_size) as i64;
                    let end = ((chunk_index as u64 + 1) * chunk_size)
                        .min(args.range_end as u64) as i64;

                    let chunk_args = ECDLPArguments {
                        range_start: start,
                        range_end: end,
                        n_threads: 1,
                        pseudo_constant_time: args.pseudo_constant_time,
                        progress_report_function: NoopReportFn, // TODO: global-scale progress
                    };

                    let (offset, normalized, num_batches) =
                        decode_prep(precomputed_tables, point, &chunk_args, 1, 0);

                    let progress_wrapper = |progress: f64| {
                        if !chunk_args.pseudo_constant_time && end_flag.load(Ordering::Relaxed) {
                            ControlFlow::Break(())
                        } else {
                            let result = args.progress_report_function.report(progress);
                            if result.is_break() {
                                end_flag.store(true, Ordering::SeqCst);
                            }
                            result
                        }
                    };

                    let point_iter = make_point_iterator(precomputed_tables, normalized, num_batches);

                    if let Some(res) = fast_ecdlp(
                        precomputed_tables,
                        normalized,
                        point_iter,
                        chunk_args.pseudo_constant_time,
                        &end_flag,
                        progress_wrapper,
                        t2_cache,
                        t2_cache_alpha,
                    ) {
                        end_flag.store(true, Ordering::SeqCst);
                        return Some(offset + res as i64);
                    }
                }

                None
            }));
        }

        // Join all threads; return first success
        let mut found: Option<i64> = None;
        for h in handles {
            match h.join() {
                Ok(res) if found.is_none() && res.is_some() => found = res,
                Ok(_) => {}
                Err(_) => {
                    end_flag.store(true, Ordering::SeqCst);
                }
            }
        }
        found
    })
}


fn fast_ecdlp(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    target_point: RistrettoPoint,
    point_iterator: impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)>,
    pseudo_constant_time: bool,
    end_flag: &AtomicBool,
    progress_report: impl ProgressReportFunction,
    t2_cache: &[AffineMontgomeryPoint; BATCH_SIZE],
    t2_cache_alpha: &[FieldElement; BATCH_SIZE],
) -> Option<u64> {
    let t1_table = precomputed_tables.get_t1();

    let mut found = None;
    let mut consider_candidate = |m| {
        if i64_to_scalar(m) * G == target_point {
            found = found.or(Some(m as u64));
            
            // Signal other threads to stop when we find a result
            if !pseudo_constant_time {
                end_flag.store(true, Ordering::SeqCst);
            }
            true
        } else {
            false
        }
    };

    let mut batch = [FieldElement::ZERO; BATCH_SIZE];
    'outer: for (index, j_start, target_montgomery, progress) in point_iterator {
        // Check end_flag more frequently - at the start of each major iteration
        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

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

        // Z = T2[j]_x - Pm_x
        for (i, batch) in batch.iter_mut().enumerate() {
            let j = i + 1;
            let t2_point = &t2_cache[i];
            let diff = &t2_point.u - &target_montgomery.u;

            if diff.is_zero_not_ct() {
                // Case 1: (Montgomery addition) exceptional case when T2[j] = Pm.
                // m1 = j * 2^L1, m2 = -j * 2^L1
                let found =
                    consider_candidate((j_start as i64 + j as i64) << precomputed_tables.get_l1())
                        || consider_candidate(
                            (j_start as i64 - j as i64) << precomputed_tables.get_l1(),
                        );
                if !pseudo_constant_time && found {
                    break 'outer;
                }
            }
            *batch = diff;
        }

        // nu = Z^-1
        FieldElement::batch_invert(&mut batch);

        for (batch_i, nu) in batch.iter().enumerate() {
            let j = batch_i + 1;
            // Montgomery addition: general case
            let t2_point = &t2_cache[batch_i];
            let alpha = &t2_cache_alpha[batch_i] - &target_montgomery.u;

            // lambda = (T2[j]_y - Pm_y) * nu
            // Q_x = lambda^2 - A - T2[j]_x - Pm_x
            let lambda = &(&t2_point.v - &target_montgomery.v) * nu;
            let qx = &lambda.square() + &alpha;

            if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
                break 'outer;
            }

            // Case 3: general case, negative j.
            let j_start_shifted = (j_start as i64 - j as i64) << precomputed_tables.get_l1();
            if t1_table
                .lookup(&qx.to_bytes(), |i| {
                    consider_candidate(j_start_shifted + i as i64)
                    || consider_candidate(
                        j_start_shifted - i as i64,
                    )
                })
                .is_some()
            {
                // m1 = -j * 2^L1 + i, m2 = -j * 2^L1 - i
                if !pseudo_constant_time {
                    break 'outer;
                }
            }

            // lambda = (p - T2[j]_y - Pm_y) * nu
            // Q_x = lambda^2 - A - T2[j]_x - Pm_x
            let lambda = &(&-&t2_point.v - &target_montgomery.v) * nu;
            let qx = &lambda.square() + &alpha;

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

#[inline(always)]
fn batch_field_subtract<const N: usize>(
    batch: &mut [FieldElement; N],
    u_values: &[FieldElement; N],
    target_u: &FieldElement,
) {
    for i in 0..N {
        batch[i] = &u_values[i] - target_u;
    }
}

#[inline(always)]
pub fn batch_field_add<const N: usize>(
    out: &mut [FieldElement; N],
    a_values: &[FieldElement; N],
    b_values: &[FieldElement; N],
) {
    for i in 0..N {
        out[i] = &a_values[i] + &b_values[i];
    }
}

#[inline(always)]
#[allow(dead_code)]
fn batch_field_mul_and_square<const N: usize>(
    output: &mut [FieldElement; N],
    a: &[FieldElement; N],
    b: &[FieldElement; N],
) {
    let mut pos = 0;

    #[cfg(curve25519_dalek_bits = "64")]
    {
        const CHUNK_SIZE: usize = 4;
        while pos + CHUNK_SIZE <= N {
            let a_chunk: &[FieldElement; 4] = (&a[pos..pos + 4]).try_into().unwrap();
            let b_chunk: &[FieldElement; 4] = (&b[pos..pos + 4]).try_into().unwrap();
            
            let products = FieldElement::batch_mul_4way(a_chunk, b_chunk);
            let squared = FieldElement::batch_square_4way(&products);
            
            output[pos..pos + 4].copy_from_slice(&squared);
            pos += CHUNK_SIZE;
        }
    }

    #[cfg(curve25519_dalek_bits = "32")]
    {
        const CHUNK_SIZE: usize = 8;
        while pos + CHUNK_SIZE <= N {
            let a_chunk: &[FieldElement; 8] = (&a[pos..pos + 8]).try_into().unwrap();
            let b_chunk: &[FieldElement; 8] = (&b[pos..pos + 8]).try_into().unwrap();
            
            let products = FieldElement::batch_mul_8way(a_chunk, b_chunk);
            let squared = FieldElement::batch_square_8way(&products);
            
            output[pos..pos + 8].copy_from_slice(&squared);
            pos += CHUNK_SIZE;
        }
    }

    // Scalar fallback for remaining elements or when SIMD not available
    while pos < N {
        output[pos] = (&a[pos] * &b[pos]).square();
        pos += 1;
    }
}

/// Batch converts multiple i64 values to Scalar types using SIMD where possible
pub fn batch_i64_to_scalar(inputs: &[i64], outputs: &mut [Scalar]) {
    {
        use crate::ecdlp::simd_types::{i64x4};
        
        assert_eq!(inputs.len(), outputs.len());
        let len = inputs.len();
        let mut pos = 0;
        
        // Process in chunks of 4 using SIMD
        while pos + 4 <= len {
            // Create SIMD vector from array
            let values = i64x4::from([
                inputs[pos],
                inputs[pos+1],
                inputs[pos+2],
                inputs[pos+3],
            ]);
            
            // Manual comparison with bitwise ops
            // Use cmp_gt with -1 instead of cmp_ge with 0
            let is_positive = values.cmp_gt(i64x4::splat(-1));
            
            // Negate values
            let neg_values = i64x4::splat(0) - values;
            
            // Convert to arrays for manual selection since select isn't available
            let values_array = values.to_array();
            let neg_values_array = neg_values.to_array();
            let is_positive_array = is_positive.to_array();
            
            // Process individual conversions
            for i in 0..4 {
                // Manual selection based on mask
                let abs_value = if is_positive_array[i] > 0 {
                    values_array[i]
                } else {
                    neg_values_array[i]
                };
                
                let scalar_pos = Scalar::from(abs_value as u64);
                
                // Use the sign mask to determine if we need negation
                outputs[pos + i] = if is_positive_array[i] > 0 {
                    scalar_pos
                } else {
                    -&scalar_pos
                };
            }
            
            pos += 4;
        }
        
        // Handle remaining elements with scalar code
        for i in pos..len {
            outputs[i] = i64_to_scalar(inputs[i]);
        }
    }
}

pub fn batch_compute_shifts<const N: usize>(
    neg_shifts: &mut [i64; N],
    pos_shifts: &mut [i64; N],
    j_start: usize,
    l1: usize
) {
    {
        use crate::ecdlp::simd_types::i64x4;
        
        let mut pos = 0;
        let j_start_i64 = j_start as i64;
        let shift_amount = l1 as i64;
        let shift_factor = 1i64 << shift_amount; // Compute shift factor once
        
        // Process in chunks of 4 using SIMD
        while pos + 4 <= N {
            // Create indices vector [1,2,3,4] + pos
            let indices = [
                (pos + 1) as i64,
                (pos + 2) as i64,
                (pos + 3) as i64,
                (pos + 4) as i64
            ];
            let j_indices = i64x4::from(indices);
            
            // Create j_start vector [j_start, j_start, j_start, j_start]
            let j_start_vec = i64x4::splat(j_start_i64);
            
            // Compute j_start - j and j_start + j
            let neg_j = j_start_vec - j_indices;
            let pos_j = j_start_vec + j_indices;
            
            // Manually shift the values by multiplying
            let neg_shifted = neg_j * i64x4::splat(shift_factor);
            let pos_shifted = pos_j * i64x4::splat(shift_factor);
            
            // Convert to arrays for access
            let neg_shifted_array = neg_shifted.to_array();
            let pos_shifted_array = pos_shifted.to_array();
            
            // Store the results
            for i in 0..4 {
                if pos + i < N {
                    neg_shifts[pos + i] = neg_shifted_array[i];
                    pos_shifts[pos + i] = pos_shifted_array[i];
                }
            }
            
            pos += 4;
        }
        
        // Handle remaining elements with scalar code
        for i in pos..N {
            let j = i + 1;
            neg_shifts[i] = (j_start as i64 - j as i64) << l1;
            pos_shifts[i] = (j_start as i64 + j as i64) << l1;
        }
    }
}

#[multiversion(targets(
    "x86_64+avx2",
    "x86_64+sse2",
    "aarch64+neon",
))]
fn fast_ecdlp_simd(
    precomputed_tables: &ECDLPTablesFileView<'_>,
    target_point: RistrettoPoint,
    point_iterator: impl Iterator<Item = (usize, usize, AffineMontgomeryPoint, f64)>,
    pseudo_constant_time: bool,
    end_flag: &AtomicBool,
    progress_report: impl ProgressReportFunction,
    t2_cache_alpha: &[FieldElement; BATCH_SIZE],
    t2_u_values: &[FieldElement; BATCH_SIZE],
    t2_vs: &[FieldElement; BATCH_SIZE],
    t2_vs_neg: &[FieldElement; BATCH_SIZE],
    batch: &mut [FieldElement; BATCH_SIZE],
    alphas: &mut [FieldElement; BATCH_SIZE],
    qxs: &mut [FieldElement; BATCH_SIZE],
    neg_qxs: &mut [FieldElement; BATCH_SIZE],
) -> Option<u64> {
    // These SIMD helper functions are defined within the multiversion scope,
    // so they get recompiled with the correct target features for each variant.

    #[inline(always)]
    fn batch_field_mul_and_square_inline<const N: usize>(
        output: &mut [FieldElement; N],
        a: &[FieldElement; N],
        b: &[FieldElement; N],
    ) {
        let mut pos = 0;

        cfg_if! {
            if #[cfg(all(curve25519_dalek_bits = "64", target_feature = "avx2"))] {
                const CHUNK_SIZE: usize = 4;
                while pos + CHUNK_SIZE <= N {
                    let a_chunk: &[FieldElement; 4] = (&a[pos..pos + 4]).try_into().unwrap();
                    let b_chunk: &[FieldElement; 4] = (&b[pos..pos + 4]).try_into().unwrap();

                    let products = FieldElement::batch_mul_4way(a_chunk, b_chunk);
                    let squared = FieldElement::batch_square_4way(&products);

                    output[pos..pos + 4].copy_from_slice(&squared);
                    pos += CHUNK_SIZE;
                }
            }
        }

        // Scalar fallback for remaining elements or when SIMD not available
        while pos < N {
            output[pos] = (&a[pos] * &b[pos]).square();
            pos += 1;
        }
    }

    let t1_table = precomputed_tables.get_t1();

    let mut found = None;
    let l1 = precomputed_tables.get_l1();

    let (qx_out, qx_tmp) = (qxs, neg_qxs);
    
    'outer: for (index, j_start, target_montgomery, progress) in point_iterator {
        let mut consider_candidate = |m| {
            if i64_to_scalar(m) * G == target_point {
                found = found.or(Some(m as u64));
                
                true
            } else {
                false
            }
        };

        // Check end_flag more frequently - at the start of each major iteration
        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

        // amortize the potential cost of the report function
        if index % BATCH_SIZE == 0 {
            if let ControlFlow::Break(_) = progress_report.report(progress) {
                break 'outer;
            }
        }

        // // Case 0: target is 0. Has to be handled separately.
        let j_start_shifted = (j_start as i64) << l1;
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

        batch_field_subtract(batch, &t2_u_values, &target_montgomery.u);

        // TODO: make a helper version of this function that has AVX512/8-way operations
        // Would use runtime dispatch by caching the path to be taken based on available SIMD width

        // 4-lane batch invert inlined
        {
            const NUM_CHUNKS: usize = BATCH_SIZE / 4;
            
            // Pre-allocated scratch space (could be passed in as parameter to avoid allocation)
            let mut scratch = [FieldElement::ONE; NUM_CHUNKS * 4];
            
            // 4 parallel accumulators
            let mut acc_lanes = [FieldElement::ONE; 4];
            
            // Forward pass
            for chunk_idx in 0..NUM_CHUNKS {
                let base = chunk_idx * 4;
                let scratch_base = chunk_idx * 4;
                
                // Store current accumulators
                scratch[scratch_base] = acc_lanes[0];
                scratch[scratch_base + 1] = acc_lanes[1];
                scratch[scratch_base + 2] = acc_lanes[2];
                scratch[scratch_base + 3] = acc_lanes[3];
                
                // Load input chunk directly from batch
                let input_chunk = [
                    batch[base],
                    batch[base + 1],
                    batch[base + 2],
                    batch[base + 3],
                ];
                
                // Update accumulators using SIMD
                acc_lanes = FieldElement::batch_mul_4way(&acc_lanes, &input_chunk);
            }
            
            // Combined inversion approach
            let p01 = &acc_lanes[0] * &acc_lanes[1];
            let p23 = &acc_lanes[2] * &acc_lanes[3];
            let p0123 = &p01 * &p23;
            let inv_p0123 = p0123.invert();
            
            // Extract individual inverses
            let factors = [
                &acc_lanes[1] * &p23,
                &acc_lanes[0] * &p23,
                &p01 * &acc_lanes[3],
                &p01 * &acc_lanes[2],
            ];
            
            let inv_broadcast = [inv_p0123; 4];
            acc_lanes = FieldElement::batch_mul_4way(&inv_broadcast, &factors);
            
            // Reverse pass
            for chunk_idx in (0..NUM_CHUNKS).rev() {
                let base = chunk_idx * 4;
                let scratch_base = chunk_idx * 4;
                
                // Load input chunk
                let input_chunk = [
                    batch[base],
                    batch[base + 1],
                    batch[base + 2],
                    batch[base + 3],
                ];
                
                // Load scratch chunk
                let scratch_chunk = [
                    scratch[scratch_base],
                    scratch[scratch_base + 1],
                    scratch[scratch_base + 2],
                    scratch[scratch_base + 3],
                ];
                
                // Compute results using SIMD
                let results = FieldElement::batch_mul_4way(&acc_lanes, &scratch_chunk);
                
                // Store results back to batch
                batch[base] = results[0];
                batch[base + 1] = results[1];
                batch[base + 2] = results[2];
                batch[base + 3] = results[3];
                
                // Update accumulators using SIMD
                acc_lanes = FieldElement::batch_mul_4way(&acc_lanes, &input_chunk);
            }
        }
    
        batch_field_subtract(
                  alphas,
                  &t2_cache_alpha,
                  &target_montgomery.u
        );

        // Batch compute qxs for the regular case
        // lambda = (T2[j]_y - Pm_y) * nu
        batch_field_subtract(qx_tmp, &t2_vs, &target_montgomery.v);
        batch_field_mul_and_square_inline(qx_out, &qx_tmp, &batch);
        batch_field_add(qx_tmp, &qx_out, alphas);

        // Process in groups of 8
        for chunk_idx in 0..(BATCH_SIZE / 8) {
            let base = chunk_idx * 8;
            let queries = [
                qx_tmp[base].to_bytes(),
                qx_tmp[base + 1].to_bytes(),
                qx_tmp[base + 2].to_bytes(),
                qx_tmp[base + 3].to_bytes(),
                qx_tmp[base + 4].to_bytes(),
                qx_tmp[base + 5].to_bytes(),
                qx_tmp[base + 6].to_bytes(),
                qx_tmp[base + 7].to_bytes(),
            ];
            
            let results = t1_table.lookup_batch_8(&queries);
            
            for (lane, (found_match, value)) in results.iter().enumerate() {
                if *found_match {
                    let j = base + lane + 1;
                    let j_start_shifted = (j_start as i64 - j as i64) << l1;
                    if consider_candidate(j_start_shifted + *value as i64) ||
                       consider_candidate(j_start_shifted - *value as i64) {
                        if !pseudo_constant_time {
                            break 'outer;
                        }
                    }
                }
            }
        }

        if !pseudo_constant_time && end_flag.load(Ordering::SeqCst) {
            break 'outer;
        }

        batch_field_subtract(qx_tmp, &t2_vs_neg, &target_montgomery.v);
        batch_field_mul_and_square_inline(qx_out, &qx_tmp, &batch);
        batch_field_add(qx_tmp, &qx_out, alphas);

        // Process in groups of 8
        for chunk_idx in 0..(BATCH_SIZE / 8) {
            let base = chunk_idx * 8;
            let queries = [
                qx_tmp[base].to_bytes(),
                qx_tmp[base + 1].to_bytes(),
                qx_tmp[base + 2].to_bytes(),
                qx_tmp[base + 3].to_bytes(),
                qx_tmp[base + 4].to_bytes(),
                qx_tmp[base + 5].to_bytes(),
                qx_tmp[base + 6].to_bytes(),
                qx_tmp[base + 7].to_bytes(),
            ];
            
            let results = t1_table.lookup_batch_8(&queries);
            
            for (lane, (found_match, value)) in results.iter().enumerate() {
                if *found_match {
                    let j = base + lane + 1;
                    let j_start_shifted = (j_start as i64 + j as i64) << l1;
                    if consider_candidate(j_start_shifted + *value as i64) ||
                       consider_candidate(j_start_shifted - *value as i64) {
                        if !pseudo_constant_time {
                            break 'outer;
                        }
                    }
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

// Public API: choose SIMD or scalar based on platform
// 64-bit uses SIMD, 32-bit uses scalar (for now)
#[cfg(curve25519_dalek_bits = "64")]
pub use par_decode as par_decode_default;

#[cfg(curve25519_dalek_bits = "32")]
pub use par_decode_scalar as par_decode_default;

// Test-only flag to force 32-bit SIMD testing on 64-bit platforms
// Usage: RUSTFLAGS='--cfg curve25519_dalek_bits="32" --cfg force_ecdlp_32bit_simd_test'
#[cfg(all(test, curve25519_dalek_bits = "32", force_ecdlp_32bit_simd_test))]
pub use par_decode as par_decode_32bit_simd_test;

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{Arc, Mutex},
    };

    use super::*;
    use rand::Rng;

    const L1: usize = 26;

    // Necessary for one ECDLP tables allocation only
    static TABLES: Mutex<Option<Arc<ECDLPTables>>> = Mutex::new(None);

    fn read_or_gen_tables() -> Arc<ECDLPTables> {
        let mut tables = TABLES.lock().expect("acquire tables lock");
        if let Some(v) = tables.as_ref().cloned() {
            return v;
        }

        let inner = if !Path::new("ecdlp_table.bin").exists() {
            let tables = ECDLPTables::generate_par(L1, 16).unwrap();
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
    fn test_ecdlp_par_decode() {
        let tables = read_or_gen_tables();
        let view = tables.view();

        for i in (0..(1u64 << 48)).step_by(1 << L1).take(1 << 12) {
            let value = i;

            let mut point = RistrettoPoint::mul_base(&Scalar::from(value));

            if rand::rng().random_bool(0.5) {
                // do a round of compression/decompression to mess up the Z and Ts
                // & ecdlp will need to clear the cofactor
                point = point.compress().decompress().unwrap();
            }

            let res = par_decode(
                &view,
                point,
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .n_threads(4)
                    .pseudo_constant_time(false),
            );
            assert_eq!(res, Some(value as i64));
        }
    }

    #[test]
    fn test_ecdlp_single_large_value_timing() {
        use std::time::Instant;
        use std::hint::black_box;
        use rand::Rng;

        const N: usize = 100;
        const N_THREADS: usize = 8;

        let tables = read_or_gen_tables();
        let view = tables.view();

        // Generate N random values and corresponding points
        let mut rng = rand::thread_rng();
        let mut test_values: Vec<u64> = Vec::with_capacity(N);
        let mut test_points: Vec<RistrettoPoint> = Vec::with_capacity(N);
        
        println!("Generating {} random test points...", N);
        for _ in 0..N {
            // Generate random value in range [0, 2^63 / 50000)
            let value: u64 = rng.gen_range(0..(1u64 << 63) / 50000);
            test_values.push(value);
            
            let mut point = RistrettoPoint::mul_base(&Scalar::from(value));
            
            if rng.gen_bool(0.5) {
                // Optionally alter the point via compression/decompression
                point = point.compress().decompress().unwrap();
            }
            
            test_points.push(point);
        }
        
        println!("Running benchmarks on {} points...\n", N);

        // Benchmark par_decode
        let now = Instant::now();
        for i in 0..N {
            let args = ECDLPArguments::new_with_range(0, 1 << 63)
                .n_threads(N_THREADS)
                .pseudo_constant_time(false);
            let res = par_decode(&view, black_box(test_points[i]), args);
            assert_eq!(res, Some(test_values[i] as i64), 
                      "par_decode failed for value {} at index {}", test_values[i], i);
        }
        let total_decode_time = now.elapsed().as_secs_f64();
        let avg_decode = total_decode_time / N as f64;
        println!("SIMD decode:");
        println!("  Total time: {:.6} seconds", total_decode_time);
        println!("  Average per value: {:.6} seconds", avg_decode);

        // Benchmark par_decode_scalar
        let now = Instant::now();
        for i in 0..N {
            let args = ECDLPArguments::new_with_range(0, 1 << 63)
                .n_threads(N_THREADS)
                .pseudo_constant_time(false);
            let res = par_decode_scalar(&view, black_box(test_points[i]), args);
            assert_eq!(res, Some(test_values[i] as i64),
                      "par_decode_scalar failed for value {} at index {}", test_values[i], i);
        }
        let total_scalar_time = now.elapsed().as_secs_f64();
        let avg_scalar = total_scalar_time / N as f64;
        println!("\nScalar decode:");
        println!("  Total time: {:.6} seconds", total_scalar_time);
        println!("  Average per value: {:.6} seconds", avg_scalar);

        // Print speedup comparisons
        println!("\nSpeedup comparisons:");
        println!("  SIMD vs Scalar: {:.2}x", total_scalar_time / total_decode_time);
    }

    #[test]
    fn test_table_par() {
        let tables_par = ECDLPTables::generate_par(
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

    fn fill_array_with<F, const N: usize>(mut f: F) -> [FieldElement; N]
    where
        F: FnMut() -> FieldElement,
    {
        let mut arr = [FieldElement::ZERO; N];
        for elem in &mut arr {
            *elem = f();
        }
        arr
    }

    // Assuming CUCKOO_K = 3 for this example
    const CUCKOO_K: usize = 3;
    
    // SIMD-friendly lookup that processes multiple queries at once
    pub struct SimdCuckooT1HashMapView<'a> {
        pub keys: &'a [u32],
        pub values: &'a [u32],
        pub cuckoo_len: usize,
    }
    
    impl<'a> SimdCuckooT1HashMapView<'a> {
        /// Batch lookup for 4 queries at once using SIMD
        /// Returns a [Option<u64>; 4] array
        pub fn lookup_batch_4(
            &self,
            queries: &[[u8; 32]; 4],
        ) -> [(bool, u64); 4] {
            #[cfg(target_arch = "x86_64")]
            #[target_feature(enable = "sse4.1")]
            #[inline]
            unsafe fn lookup_batch_4_sse(
                keys_slice: &[u32],
                values_slice: &[u32],
                cuckoo_len: usize,
                queries: &[[u8; 32]; 4],
            ) -> [(bool, u64); 4] {
                use std::arch::x86_64::*;
                
                let mut results = [(false, 0u64); 4];
                let cuckoo_len = cuckoo_len as u32;
                
                // Process each cuckoo position
                for i in 0..CUCKOO_K {
                    let start = i * 8;
                    let key_offset = start + 4;
                    
                    // Extract keys and hashes for all 4 queries
                    let keys = _mm_setr_epi32(
                        u32::from_be_bytes(queries[0][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[1][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[2][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[3][key_offset..key_offset + 4].try_into().unwrap()) as i32,
                    );
                    
                    let hashes = _mm_setr_epi32(
                        u32::from_be_bytes(queries[0][start..start + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[1][start..start + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[2][start..start + 4].try_into().unwrap()) as i32,
                        u32::from_be_bytes(queries[3][start..start + 4].try_into().unwrap()) as i32,
                    );
                    
                    // Compute indices (modulo operation)
                    // For power-of-2 cuckoo_len, use AND mask instead
                    let indices = if cuckoo_len.is_power_of_two() {
                        let mask = _mm_set1_epi32((cuckoo_len - 1) as i32);
                        _mm_and_si128(hashes, mask)
                    } else {
                        // Fallback to scalar modulo for non-power-of-2
                        let h0 = _mm_extract_epi32(hashes, 0) as u32 % cuckoo_len;
                        let h1 = _mm_extract_epi32(hashes, 1) as u32 % cuckoo_len;
                        let h2 = _mm_extract_epi32(hashes, 2) as u32 % cuckoo_len;
                        let h3 = _mm_extract_epi32(hashes, 3) as u32 % cuckoo_len;
                        _mm_setr_epi32(h0 as i32, h1 as i32, h2 as i32, h3 as i32)
                    };
                    
                    // Gather table keys
                    let idx0 = _mm_extract_epi32(indices, 0) as usize;
                    let idx1 = _mm_extract_epi32(indices, 1) as usize;
                    let idx2 = _mm_extract_epi32(indices, 2) as usize;
                    let idx3 = _mm_extract_epi32(indices, 3) as usize;
                    
                    let table_keys = _mm_setr_epi32(
                        keys_slice[idx0] as i32,
                        keys_slice[idx1] as i32,
                        keys_slice[idx2] as i32,
                        keys_slice[idx3] as i32,
                    );
                    
                    // Compare keys
                    let matches = _mm_cmpeq_epi32(keys, table_keys);
                    let match_mask = _mm_movemask_ps(_mm_castsi128_ps(matches));
                    
                    // Extract values for matches
                    if match_mask & 0x1 != 0 && !results[0].0 {
                        results[0] = (true, values_slice[idx0] as u64);
                    }
                    if match_mask & 0x2 != 0 && !results[1].0 {
                        results[1] = (true, values_slice[idx1] as u64);
                    }
                    if match_mask & 0x4 != 0 && !results[2].0 {
                        results[2] = (true, values_slice[idx2] as u64);
                    }
                    if match_mask & 0x8 != 0 && !results[3].0 {
                        results[3] = (true, values_slice[idx3] as u64);
                    }
                }
                
                results
            }
            
            #[cfg(target_arch = "aarch64")]
            #[target_feature(enable = "neon")]
            #[inline]
            unsafe fn lookup_batch_4_neon(
                keys_slice: &[u32],
                values_slice: &[u32],
                cuckoo_len: usize,
                queries: &[[u8; 32]; 4],
            ) -> [(bool, u64); 4] {
                use std::arch::aarch64::*;
                
                let mut results = [(false, 0u64); 4];
                let cuckoo_len = cuckoo_len as u32;
                
                // Process each cuckoo position
                for i in 0..CUCKOO_K {
                    let start = i * 8;
                    let key_offset = start + 4;
                    
                    // Extract keys and hashes for all 4 queries (scalar for now)
                    let mut keys_array = [0u32; 4];
                    let mut hashes_array = [0u32; 4];
                    
                    for j in 0..4 {
                        keys_array[j] = u32::from_be_bytes(queries[j][key_offset..key_offset + 4].try_into().unwrap());
                        hashes_array[j] = u32::from_be_bytes(queries[j][start..start + 4].try_into().unwrap());
                    }
                    
                    let keys = vld1q_u32(keys_array.as_ptr());
                    let hashes = vld1q_u32(hashes_array.as_ptr());
                    
                    // Compute indices (modulo operation)
                    let mut indices_array = [0u32; 4];
                    if cuckoo_len.is_power_of_two() {
                        let mask = vdupq_n_u32(cuckoo_len - 1);
                        let indices = vandq_u32(hashes, mask);
                        vst1q_u32(indices_array.as_mut_ptr(), indices);
                    } else {
                        // Fallback to scalar modulo for non-power-of-2
                        vst1q_u32(hashes_array.as_mut_ptr(), hashes);
                        for j in 0..4 {
                            indices_array[j] = hashes_array[j] % cuckoo_len;
                        }
                    }
                    
                    // Gather table keys (scalar - NEON doesn't have gather)
                    let mut table_keys_array = [0u32; 4];
                    for j in 0..4 {
                        table_keys_array[j] = keys_slice[indices_array[j] as usize];
                    }
                    let table_keys = vld1q_u32(table_keys_array.as_ptr());
                    
                    // Compare keys
                    let matches = vceqq_u32(keys, table_keys);
                    
                    // Extract match mask (scalar extraction)
                    let mut matches_array = [0u32; 4];
                    vst1q_u32(matches_array.as_mut_ptr(), matches);
                    
                    // Extract values for matches
                    for j in 0..4 {
                        if matches_array[j] != 0 && !results[j].0 {
                            results[j] = (true, values_slice[indices_array[j] as usize] as u64);
                        }
                    }
                }
                
                results
            }
            
            cfg_if::cfg_if! {
                if #[cfg(target_arch = "x86_64")] {
                    unsafe { lookup_batch_4_sse(self.keys, self.values, self.cuckoo_len, queries) }
                } else if #[cfg(target_arch = "aarch64")] {
                    unsafe { lookup_batch_4_neon(self.keys, self.values, self.cuckoo_len, queries) }
                } else {
                    // Scalar fallback
                    let mut results = [(false, 0u64); 4];
                    let cuckoo_len = self.cuckoo_len as u32;
                    
                    for i in 0..CUCKOO_K {
                        let start = i * 8;
                        let key_offset = start + 4;
                        
                        for (query_idx, query) in queries.iter().enumerate() {
                            if results[query_idx].0 {
                                continue; // Already found
                            }
                            
                            let key = u32::from_be_bytes(query[key_offset..key_offset + 4].try_into().unwrap());
                            let hash = u32::from_be_bytes(query[start..start + 4].try_into().unwrap());
                            let idx = (hash % cuckoo_len) as usize;
                            
                            if self.keys[idx] == key {
                                results[query_idx] = (true, self.values[idx] as u64);
                            }
                        }
                    }
                    results
                }
            }
        }
    }

    use crate::ecdlp::table::CuckooT1HashMapView;

    fn reference_lookup_batch(
        t1_table: &CuckooT1HashMapView<'_>,
        queries: &[[u8; 32]],
    ) -> Vec<Option<u64>> {
        queries.iter().map(|query| {
            let mut found = None;
            t1_table.lookup(query, |value| {
                found = Some(value);
                true // Stop on first match
            });
            found
        }).collect()
    }
    

    #[test]
    fn test_simd_lookup_correctness() {
        // Create a test table with known values
        let mut keys = vec![0u32; 1000];
        let mut values = vec![0u32; 1000];
        
        // Insert some test data
        let test_entries = vec![
            (0x12345678u32, 100u32),
            (0x87654321u32, 200u32),
            (0xABCDEF00u32, 300u32),
            (0xDEADBEEFu32, 400u32),
        ];
        
        // Simple hash function for testing
        for (key, value) in &test_entries {
            let hash = (*key as usize) % 1000;
            keys[hash] = *key;
            values[hash] = *value;
        }
        
        let t1_table = CuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: 1000,
        };
        
        let simd_table = SimdCuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: 1000,
        };
        
        // Test with queries that should match
        let mut queries = [[0u8; 32]; 4];
        
        // Place keys at the expected positions (assuming CUCKOO_K=3)
        // First cuckoo position (bytes 4-8)
        queries[0][4..8].copy_from_slice(&0x12345678u32.to_be_bytes());
        queries[0][0..4].copy_from_slice(&(0x12345678u32 % 1000).to_be_bytes());
        
        queries[1][4..8].copy_from_slice(&0x87654321u32.to_be_bytes());
        queries[1][0..4].copy_from_slice(&(0x87654321u32 % 1000).to_be_bytes());
        
        // Test non-matching queries
        queries[2][4..8].copy_from_slice(&0xFFFFFFFFu32.to_be_bytes());
        queries[3][4..8].copy_from_slice(&0x00000000u32.to_be_bytes());
        
        // Get results from both implementations
        let reference_results = reference_lookup_batch(&t1_table, &queries);
        let simd_results = simd_table.lookup_batch_4(&queries);
        
        // Verify results match
        for i in 0..4 {
            let ref_result = reference_results[i];
            let simd_result = if simd_results[i].0 { Some(simd_results[i].1) } else { None };
            
            assert_eq!(ref_result, simd_result, 
                "Mismatch at index {}: reference={:?}, simd={:?}", 
                i, ref_result, simd_result);
        }
        
        println!("✓ SIMD lookup correctness test passed");
    }
    
    #[test]
    fn test_simd_lookup_stress() {
        // Stress test with random data
        let mut rng = rand::rng();
        let table_size = 10000;
        
        let mut keys = vec![0u32; table_size];
        let mut values = vec![0u32; table_size];
        
        // Fill with random data
        rng.fill(&mut keys[..table_size / 2]);
        rng.fill(&mut values[..table_size / 2]);
        
        let t1_table = CuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: table_size,
        };
        
        let simd_table = SimdCuckooT1HashMapView {
            keys: &keys,
            values: &values,
            cuckoo_len: table_size,
        };
        
        // Test 1000 random query batches
        for _ in 0..1000 {
            let mut queries = [[0u8; 32]; 4];
            for q in &mut queries {
                rng.fill(q);
            }
            
            let reference_results = reference_lookup_batch(&t1_table, &queries);
            let simd_results = simd_table.lookup_batch_4(&queries);
            
            for i in 0..4 {
                let ref_result = reference_results[i];
                let simd_result = if simd_results[i].0 { Some(simd_results[i].1) } else { None };
                assert_eq!(ref_result, simd_result);
            }
        }
        
        println!("✓ SIMD lookup stress test passed (1000 random batches)");
    }

    #[test]
    fn test_simple_ecdlp_decode() {
        println!("\n=== Testing simple ECDLP decode ===");

        let tables = read_or_gen_tables();
        let view = tables.view();

        // Test very small values first - compare SIMD vs scalar
        for value in [0u64, 1, 2, 100] {
            println!("\n=== Testing value: {} ===", value);

            let point = RistrettoPoint::mul_base(&Scalar::from(value));

            // Try scalar version first
            println!("Testing with scalar decode (par_decode_scalar)...");
            let args_scalar = ECDLPArguments::new_with_range(0, 1 << 48)
                .n_threads(1)
                .pseudo_constant_time(false);

            let result_scalar = par_decode_scalar(&view, point, args_scalar);
            println!("Scalar result: {:?}", result_scalar);

            // Try SIMD version
            println!("Testing with SIMD decode (par_decode)...");
            let args_simd = ECDLPArguments::new_with_range(0, 1 << 48)
                .n_threads(1)
                .pseudo_constant_time(false);

            let result_simd = par_decode(&view, point, args_simd);
            println!("SIMD result: {:?}", result_simd);

            assert_eq!(
                result_scalar,
                Some(value as i64),
                "Scalar version failed to decode value {}", value
            );

            assert_eq!(
                result_simd,
                Some(value as i64),
                "SIMD version failed to decode value {}", value
            );

            println!("✓ Both scalar and SIMD decoded {} correctly", value);
        }

        println!("\n✓ Simple ECDLP decode passed");
    }

    #[test]
    fn stress_test_ecdlp_simd_correctness() {
        let tables = read_or_gen_tables();
        let view = tables.view();
        let mut rng = rand::rng();
        
        const MAX_VALUE: u64 = 1u64 << 48;
        const NUM_TESTS: usize = 1000;
        
        for i in 0..NUM_TESTS {
            let test_value = rng.random_range(0..MAX_VALUE);
            let point = RistrettoPoint::mul_base(&Scalar::from(test_value));
            
            let args = ECDLPArguments::new_with_range(0, MAX_VALUE as i64)
                .n_threads(16)
                .pseudo_constant_time(false);
            
            let result = par_decode(&view, point, args);
            assert_eq!(result, Some(test_value as i64), 
                      "Failed on iteration {} with value {}", i, test_value);
            
            if i % 100 == 0 && i > 0 {
                println!("✓ Completed {} iterations", i);
            }
        }
        
        println!("✅ All {} random tests passed!", NUM_TESTS);
    }
}
