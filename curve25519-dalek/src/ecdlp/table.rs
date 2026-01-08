//! Table generation module. This module is public but should be treated as
//! unstable. The API may change without notice.

use bytemuck::{Pod, Zeroable};
use std::{
    mem::size_of,
    ops::ControlFlow,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::constants::RISTRETTO_BASEPOINT_POINT;
use crate::field::FieldElement;
use crate::traits::Identity;
use crate::{EdwardsPoint, RistrettoPoint, Scalar};

use super::affine_montgomery::AffineMontgomeryPoint;

pub(crate) const L2: usize = 9; // corresponds to a batch size of 256 and a T2 table of a few Ko.
pub(crate) const BATCH_SIZE: usize = 1 << (L2 - 1);

pub(crate) const I_BITS: usize = L2 - 1;
pub(crate) const I_MAX: usize = (1 << I_BITS) + 1; // there needs to be one more element in T2
pub(crate) const CUCKOO_K: usize = 3; // number of cuckoo lookups before giving up

// Note: file layout is just T2 followed by T1 keys and then T1 values.
// We just do casts using `bytemuck` since everything are PODs.

/// A view into an ECDLP precomputed table. This is a wrapper around a read-only byte array, which you could back by an memory mapped file, for example.
pub struct ECDLPTablesFileView<'a> {
    bytes: &'a [u8],
    l1: usize,
}

impl<'a> ECDLPTablesFileView<'a> {
    /// Calculate the cuckoo length for a given `l1` const.
    pub fn cuckoo_len(l1: usize) -> usize {
        let j_max: u64 = 1 << (l1 - 1);
        // x1.3 is the load factor of the cuckoo hashmap.
        ((j_max * 30 / 100) + j_max) as usize
    }

    /// ECDLP algorithm may panic if the alignment or size of `bytes` is wrong.
    pub fn from_bytes(bytes: &'a [u8], l1: usize) -> Self {
        // TODO(merge): check align/size of `bytes` here
        Self { bytes, l1 }
    }

    /// Get the T1 table.
    pub(crate) fn get_t1(&self) -> CuckooT1HashMapView<'_> {
        let t1_keys_values: &[u32] =
            bytemuck::cast_slice(&self.bytes[(size_of::<T2MontgomeryCoordinates>() * I_MAX)..]);

        let cuckoo_len = Self::cuckoo_len(self.l1);
        CuckooT1HashMapView {
            keys: &t1_keys_values[0..cuckoo_len],
            values: &t1_keys_values[cuckoo_len..],
            cuckoo_len,
        }
    }

    /// Get the `l1` constant used to generate the tables.
    #[inline(always)]
    pub fn get_l1(&self) -> usize {
        self.l1
    }

    /// Get the T2 table.
    pub(crate) fn get_t2(&self) -> T2LinearTableView<'_> {
        let t2: &[T2MontgomeryCoordinates] =
            bytemuck::cast_slice(&self.bytes[0..(size_of::<T2MontgomeryCoordinates>() * I_MAX)]);
        T2LinearTableView(t2)
    }
}

/// Canonical FieldElement type.
type CompressedFieldElement = [u8; 32];

#[repr(C, align(32))]
#[derive(Clone, Copy, Default, Pod, Zeroable, Debug)]
/// An entry in the T2 table. Represents a (u,v) coordinate on the Montgomery curve.
pub(crate) struct T2MontgomeryCoordinates {
    /// The `u` coordinate.
    pub u: CompressedFieldElement,
    /// The `v` coordinate.
    pub v: CompressedFieldElement,
}

impl From<T2MontgomeryCoordinates> for AffineMontgomeryPoint {
    fn from(e: T2MontgomeryCoordinates) -> Self {
        Self {
            u: FieldElement::from_bytes(&e.u),
            v: FieldElement::from_bytes(&e.v),
        }
    }
}

impl From<AffineMontgomeryPoint> for T2MontgomeryCoordinates {
    fn from(e: AffineMontgomeryPoint) -> Self {
        Self {
            u: e.u.to_bytes(),
            v: e.v.to_bytes(),
        }
    }
}

/// A view into the T2 table.
pub(crate) struct T2LinearTableView<'a>(pub &'a [T2MontgomeryCoordinates]);

impl T2LinearTableView<'_> {
    #[inline]
    pub fn index(&self, index: usize) -> AffineMontgomeryPoint {
        let T2MontgomeryCoordinates { u, v } = self.0[index];
        AffineMontgomeryPoint::from_bytes(&u, &v)
    }
}

/// A view into the T1 table.
pub(crate) struct CuckooT1HashMapView<'a> {
    /// Cuckoo keys
    pub keys: &'a [u32],
    /// Cuckoo values
    pub values: &'a [u32],
    /// Cuckoo length for provided L1
    pub cuckoo_len: usize,
}

#[allow(dead_code)]
impl CuckooT1HashMapView<'_> {
    pub(crate) fn lookup(
        &self,
        x: &[u8],
        mut is_problem_answer: impl FnMut(u64) -> bool,
    ) -> Option<u64> {
        for i in 0..CUCKOO_K {
            let start = i * 8;
            let end = start + 4;
            let key = u32::from_be_bytes(x[end..end + 4].try_into().expect("key u32"));
            let h = u32::from_be_bytes(x[start..start + 4].try_into().expect("h u32")) as usize
                % self.cuckoo_len;
            if self.keys[h] == key {
                let value = self.values[h] as u64;
                if is_problem_answer(value) {
                    return Some(value);
                }
            }
        }
        None
    }

    pub(crate) fn lookup_batch_4(&self, queries: &[[u8; 32]; 4]) -> [(bool, u64); 4] {
        let mut results = [(false, 0u64); 4];
        let cuckoo_len = self.cuckoo_len as u32;

        use crate::ecdlp::simd_types::u32x4;
        let p2_mask = u32x4::splat(cuckoo_len - 1);
        
        // Process each cuckoo position
        for i in 0..CUCKOO_K {
            let start = i * 8;
            let key_offset = start + 4;
            
            // Extract keys for all 4 queries
            let keys = u32x4::new([
                u32::from_be_bytes(queries[0][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[1][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[2][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[3][key_offset..key_offset + 4].try_into().unwrap()),
            ]);
            
            // Extract hashes
            let hashes = u32x4::new([
                u32::from_be_bytes(queries[0][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[1][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[2][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[3][start..start + 4].try_into().unwrap()),
            ]);
            
            // Compute indices
            let indices = if cuckoo_len.is_power_of_two() {
                // Use bitwise AND for power-of-2 modulo
                hashes & p2_mask
            } else {
                // Fallback to scalar modulo - need to_array() first
                let hash_array = hashes.to_array();
                u32x4::new([
                    hash_array[0] % cuckoo_len,
                    hash_array[1] % cuckoo_len,
                    hash_array[2] % cuckoo_len,
                    hash_array[3] % cuckoo_len,
                ])
            };
            
            // Extract indices as array
            let idx_array = indices.to_array();
            
            // Gather table keys (still need scalar gather)
            let table_keys = u32x4::new([
                self.keys[idx_array[0] as usize],
                self.keys[idx_array[1] as usize],
                self.keys[idx_array[2] as usize],
                self.keys[idx_array[3] as usize],
            ]);
            
            // Compare keys
            let matches = keys.cmp_eq(table_keys);
            let match_array = matches.to_array();
            
            // Process matches
            if match_array[0] != 0 && !results[0].0 {
                results[0] = (true, self.values[idx_array[0] as usize] as u64);
            }
            if match_array[1] != 0 && !results[1].0 {
                results[1] = (true, self.values[idx_array[1] as usize] as u64);
            }
            if match_array[2] != 0 && !results[2].0 {
                results[2] = (true, self.values[idx_array[2] as usize] as u64);
            }
            if match_array[3] != 0 && !results[3].0 {
                results[3] = (true, self.values[idx_array[3] as usize] as u64);
            }
        }
        
        results
    }

    pub(crate) fn lookup_batch_8(&self, queries: &[[u8; 32]; 8]) -> [(bool, u64); 8] {
        let mut results = [(false, 0u64); 8];
        let cuckoo_len = self.cuckoo_len as u32;

        use crate::ecdlp::simd_types::u32x8;
        let p2_mask = u32x8::splat(cuckoo_len - 1);
        
        // Process each cuckoo position
        for i in 0..CUCKOO_K {
            let start = i * 8;
            let key_offset = start + 4;
            
            // Extract keys for all 8 queries
            let keys = u32x8::new([
                u32::from_be_bytes(queries[0][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[1][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[2][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[3][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[4][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[5][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[6][key_offset..key_offset + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[7][key_offset..key_offset + 4].try_into().unwrap()),
            ]);
            
            // Extract hashes
            let hashes = u32x8::new([
                u32::from_be_bytes(queries[0][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[1][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[2][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[3][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[4][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[5][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[6][start..start + 4].try_into().unwrap()),
                u32::from_be_bytes(queries[7][start..start + 4].try_into().unwrap()),
            ]);
            
            // Compute indices
            let indices = if cuckoo_len.is_power_of_two() {
                // Use bitwise AND for power-of-2 modulo
                hashes & p2_mask
            } else {
                // Fallback to scalar modulo
                let hash_array = hashes.to_array();
                u32x8::new([
                    hash_array[0] % cuckoo_len,
                    hash_array[1] % cuckoo_len,
                    hash_array[2] % cuckoo_len,
                    hash_array[3] % cuckoo_len,
                    hash_array[4] % cuckoo_len,
                    hash_array[5] % cuckoo_len,
                    hash_array[6] % cuckoo_len,
                    hash_array[7] % cuckoo_len,
                ])
            };
            
            // Extract indices as array
            let idx_array = indices.to_array();
            
            // Gather table keys (still need scalar gather)
            let table_keys = u32x8::new([
                self.keys[idx_array[0] as usize],
                self.keys[idx_array[1] as usize],
                self.keys[idx_array[2] as usize],
                self.keys[idx_array[3] as usize],
                self.keys[idx_array[4] as usize],
                self.keys[idx_array[5] as usize],
                self.keys[idx_array[6] as usize],
                self.keys[idx_array[7] as usize],
            ]);
            
            // Compare keys
            let matches = keys.cmp_eq(table_keys);
            let match_array = matches.to_array();
            
            // Process matches
            for j in 0..8 {
                if match_array[j] != 0 && !results[j].0 {
                    results[j] = (true, self.values[idx_array[j] as usize] as u64);
                }
            }
        }
        
        results
    }
}

/// A progress report step.
/// This is used to report progress to the user and give additional context.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportStep {
    /// Generating T1 points.
    T1PointsGeneration,
    /// Setting up the T1 cuckoo hashmap.
    T1CuckooSetup,
    /// Generating the T2 table.
    T2Table,
}

/// A trait for reporting progress during table generation.
pub trait ProgressTableGenerationReportFunction {
    /// Run the progress report function.
    fn report(&self, progress: f64, step: ReportStep) -> ControlFlow<()>;
}

/// A no-op progress report function.
pub struct NoOpProgressTableGenerationReportFunction;

impl ProgressTableGenerationReportFunction for NoOpProgressTableGenerationReportFunction {
    fn report(&self, _progress: f64, _step: ReportStep) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// A trait for automatically converting a closure into a progress report function.
impl<F: Fn(f64, ReportStep) -> ControlFlow<()>> ProgressTableGenerationReportFunction for F {
    #[inline(always)]
    fn report(&self, progress: f64, step: ReportStep) -> ControlFlow<()> {
        self(progress, step)
    }
}

pub mod table_generation {
    //! Generate the precomputed tables.

    use super::*;
    use std::{io, sync::atomic::AtomicBool, thread};

    fn t1_cuckoo_setup<P: ProgressTableGenerationReportFunction>(
        cuckoo_len: usize,
        j_max: usize,
        all_entries: &[CompressedFieldElement],
        t1_values: &mut [u32],
        t1_keys: &mut [u32],
        progress_report: &P,
    ) -> io::Result<()> {
        use core::mem::swap;

        /// Dumb cuckoo rehashing threshold.
        const CUCKOO_MAX_INSERT_SWAPS: usize = 500;

        let mut hash_index = vec![0u8; cuckoo_len];

        for i in 0..=j_max {
            let mut v = i as _;
            let mut old_hash_id = 1u8;

            if i % (j_max / 1000 + 1) == 0 {
                let progress = i as f64 / j_max as f64;
                if let ControlFlow::Break(_) =
                    progress_report.report(progress, ReportStep::T1CuckooSetup)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "Interrupted by progress report",
                    ));
                }
            }

            for j in 0..CUCKOO_MAX_INSERT_SWAPS {
                let x = &all_entries[v as usize];
                let start = (old_hash_id as usize - 1) * 8;
                let end = start + 4;
                let mut key = u32::from_be_bytes(x[end..end + 4].try_into().expect("key u32"));
                let h1 =
                    u32::from_be_bytes(x[start..start + 4].try_into().expect("h1 u32")) as usize;
                let h = h1 % cuckoo_len;

                if hash_index[h] == 0 {
                    hash_index[h] = old_hash_id;
                    t1_values[h] = v;
                    t1_keys[h] = key;
                    break;
                } else {
                    swap(&mut old_hash_id, &mut hash_index[h]);
                    swap(&mut v, &mut t1_values[h]);
                    swap(&mut key, &mut t1_keys[h]);
                    old_hash_id = old_hash_id % 3 + 1;

                    if j == CUCKOO_MAX_INSERT_SWAPS - 1 {
                        // We actually don't have to implement the case where we need to rehash the
                        // whole map.
                        panic!("Cuckoo hashmap insert needs rehashing.")
                    }
                }
            }
        }

        Ok(())
    }

    fn create_t1_table<P: ProgressTableGenerationReportFunction>(
        l1: usize,
        dest: &mut [u8],
        progress_report: &P,
    ) -> io::Result<()> {
        let j_max = 1 << (l1 - 1);
        let cuckoo_len = (j_max as u64 * 30 / 100) as usize + j_max;

        let mut all_entries = Vec::with_capacity(j_max + 1);

        let acc = RistrettoPoint::identity().0;
        let step = RISTRETTO_BASEPOINT_POINT.0.mul_by_cofactor(); // table is based on number*cofactor

        let mut acc = AffineMontgomeryPoint::from(&acc);
        let step = AffineMontgomeryPoint::from(&step);
        for i in 0..=j_max {
            // acc is i * G
            let point = acc; // i * G

            if i % (j_max / 1000 + 1) == 0 {
                if let ControlFlow::Break(_) =
                    progress_report.report(i as f64 / j_max as f64, ReportStep::T1PointsGeneration)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "Interrupted by progress report",
                    ));
                }
            }

            all_entries.push(point.u.to_bytes());
            acc = acc.addition_not_ct(&step);
        }

        let (t1_keys_dest, t1_values_dest) = dest.split_at_mut(cuckoo_len * size_of::<u32>());
        let t1_keys_dest: &mut [u32] = bytemuck::cast_slice_mut(t1_keys_dest);
        let t1_values_dest: &mut [u32] = bytemuck::cast_slice_mut(t1_values_dest);

        t1_cuckoo_setup(
            cuckoo_len,
            j_max,
            &all_entries,
            t1_values_dest,
            t1_keys_dest,
            progress_report,
        )?;

        Ok(())
    }

    fn create_t2_table<P: ProgressTableGenerationReportFunction>(
        l1: usize,
        dest: &mut [u8],
        progress_report: &P,
    ) -> io::Result<()> {
        let two_to_l1 = EdwardsPoint::mul_base(&Scalar::from(1u32 << l1)); // 2^l1
        let two_to_l1 = two_to_l1.mul_by_cofactor(); // clear cofactor

        let arr: &mut [T2MontgomeryCoordinates] = bytemuck::cast_slice_mut(dest);

        let two_to_l1 = AffineMontgomeryPoint::from(&two_to_l1);
        let mut acc = two_to_l1;
        for j in 1..I_MAX {
            if let ControlFlow::Break(_) =
                progress_report.report(j as f64 / I_MAX as f64, ReportStep::T2Table)
            {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "Interrupted by progress report",
                ));
            }

            arr[j - 1] = acc.into();
            acc = acc.addition_not_ct(&two_to_l1);
        }

        Ok(())
    }

    fn create_t1_table_par<P: ProgressTableGenerationReportFunction + Sync>(
        l1: usize,
        n_threads: usize,
        dest: &mut [u8],
        progress_report: &P,
    ) -> io::Result<()> {
        let j_max = 1 << (l1 - 1);
        let cuckoo_len = (j_max as u64 * 30 / 100) as usize + j_max;

        // Use atomic counter for progress tracking
        let interrupted = AtomicBool::new(false);
        let progress_counter = AtomicUsize::new(0);
        let report_every = j_max / 1000 + 1;

        // Pre-allocate the full array
        let mut all_entries = Vec::new();

        // Calculate chunks
        let chunk_size = (j_max + 1).div_ceil(n_threads); // ceiling division

        thread::scope(|s| {
            let handles = (0..n_threads)
                .filter_map(|thread_i| {
                    let start_idx = thread_i * chunk_size;
                    let end_idx = ((thread_i + 1) * chunk_size).min(j_max + 1);

                    if start_idx >= end_idx {
                        return None;
                    }

                    let interrupted = &interrupted;
                    let progress_counter = &progress_counter;

                    Some(s.spawn(move || {
                        let step = AffineMontgomeryPoint::from(
                            &RISTRETTO_BASEPOINT_POINT.0.mul_by_cofactor(),
                        );

                        // Compute starting point for this chunk
                        let start_point = if start_idx == 0 {
                            AffineMontgomeryPoint::from(&RistrettoPoint::identity().0)
                        } else {
                            let scalar = Scalar::from(start_idx as u64);
                            let point = EdwardsPoint::mul_base(&scalar).mul_by_cofactor();
                            AffineMontgomeryPoint::from(&point)
                        };

                        let mut acc = start_point;
                        let mut chunk_entries = Vec::with_capacity(end_idx - start_idx);

                        for i in start_idx..end_idx {
                            chunk_entries.push(acc.u.to_bytes());

                            if i % report_every == 0 {
                                let old_count = progress_counter.fetch_add(1, Ordering::Relaxed);
                                if old_count % 10 == 0 {
                                    let progress = (old_count * report_every) as f64 / j_max as f64;
                                    if let ControlFlow::Break(_) = progress_report
                                        .report(progress, ReportStep::T1PointsGeneration)
                                    {
                                        // Can't easily interrupt from inside thread, would need to add a flag
                                        interrupted.store(true, Ordering::Relaxed);
                                        return chunk_entries;
                                    }
                                }
                            }

                            acc = acc.addition_not_ct(&step);
                        }

                        // Write results to the shared array
                        chunk_entries
                    }))
                })
                .collect::<Vec<_>>();

            // Wait for all threads to complete
            for handle in handles {
                let entries = handle.join().expect("Thread panicked");
                all_entries.extend(entries);
            }
        });

        if interrupted.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Interrupted by progress report",
            ));
        }

        // The cuckoo setup remains sequential as it's harder to parallelize
        let (t1_keys_dest, t1_values_dest) = dest.split_at_mut(cuckoo_len * size_of::<u32>());
        let t1_keys_dest: &mut [u32] = bytemuck::cast_slice_mut(t1_keys_dest);
        let t1_values_dest: &mut [u32] = bytemuck::cast_slice_mut(t1_values_dest);

        t1_cuckoo_setup(
            cuckoo_len,
            j_max,
            &all_entries,
            t1_values_dest,
            t1_keys_dest,
            progress_report,
        )?;

        Ok(())
    }

    fn create_t2_table_par<P: ProgressTableGenerationReportFunction + Sync>(
        l1: usize,
        n_threads: usize,
        dest: &mut [u8],
        progress_report: &P,
    ) -> io::Result<()> {
        let two_to_l1 = EdwardsPoint::mul_base(&Scalar::from(1u32 << l1)).mul_by_cofactor();
        let two_to_l1_affine = AffineMontgomeryPoint::from(&two_to_l1);

        let coordinates: &mut [T2MontgomeryCoordinates] = bytemuck::cast_slice_mut(dest);

        let progress_counter = AtomicUsize::new(0);
        let interrupted = AtomicBool::new(false);
        let total_points = I_MAX - 1;
        let report_every = total_points / 1000 + 1;

        // Calculate chunks
        let chunk_size = total_points.div_ceil(n_threads);

        thread::scope(|s| {
            // Split the array into chunks for each thread
            let arr_chunks = coordinates.chunks_mut(chunk_size);

            let handles = arr_chunks
                .enumerate()
                .filter_map(|(thread_i, chunk)| {
                    let start_offset = thread_i * chunk_size + 1; // 1-indexed
                    let end_offset = ((thread_i + 1) * chunk_size + 1).min(I_MAX);

                    if start_offset >= end_offset {
                        return None;
                    }

                    let interrupted = &interrupted;
                    let progress_counter = &progress_counter;

                    Some(s.spawn(move || {
                        // Calculate starting point: start_offset * two_to_l1
                        let scalar = Scalar::from(start_offset as u64);
                        let start_point =
                            EdwardsPoint::mul_base(&(scalar * Scalar::from(1u32 << l1)))
                                .mul_by_cofactor();
                        let mut acc = AffineMontgomeryPoint::from(&start_point);

                        for j in start_offset..end_offset {
                            // Access the chunk relative to its beginning
                            let chunk_idx = j - start_offset;
                            chunk[chunk_idx] = acc.into();

                            if (j - 1) % report_every == 0 {
                                let old_count = progress_counter.fetch_add(1, Ordering::Relaxed);
                                if old_count % 10 == 0 {
                                    let progress =
                                        (old_count * report_every) as f64 / total_points as f64;
                                    if let ControlFlow::Break(_) =
                                        progress_report.report(progress, ReportStep::T2Table)
                                    {
                                        // Can't easily interrupt from inside thread
                                        interrupted.store(true, Ordering::Relaxed);
                                        return;
                                    }
                                }
                            }

                            acc = acc.addition_not_ct(&two_to_l1_affine);
                        }
                    }))
                })
                .collect::<Vec<_>>();

            // Wait for all threads
            for handle in handles {
                handle.join().expect("Thread panicked");
            }
        });

        if interrupted.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "Interrupted by progress report",
            ));
        }

        Ok(())
    }

    /// Length of the table file for a given `l1` const.
    pub const fn table_file_len(l1: usize) -> usize {
        let j_max: u64 = 1 << (l1 - 1);
        // x1.3 is the load factor of the cuckoo hashmap.
        let cuckoo_len = ((j_max * 30 / 100) + j_max) as usize;

        I_MAX * size_of::<T2MontgomeryCoordinates>() + (cuckoo_len * 2) * size_of::<u32>()
    }

    /// Generate the ECDLP precomputed tables file.
    /// To prepare `dest`, you should use an memory mapped file or a 32-byte aligned byte array.
    /// The byte array length should be the return value of [`table_file_len`].
    /// No progress report will be done.
    pub fn create_table_file(l1: usize, dest: &mut [u8]) -> io::Result<()> {
        create_table_file_with_progress_report(l1, dest, NoOpProgressTableGenerationReportFunction)
    }

    /// Generate the ECDLP precomputed tables file, with multithreading.
    /// To prepare `dest`, you should use an memory mapped file or a 32-byte aligned byte array.
    /// The byte array length should be the return value of [`table_file_len`].
    /// No progress report will be done.
    pub fn create_table_file_par(l1: usize, n_threads: usize, dest: &mut [u8]) -> io::Result<()> {
        create_table_file_with_progress_report_par(
            l1,
            n_threads,
            dest,
            NoOpProgressTableGenerationReportFunction,
        )
    }

    /// Generate the ECDLP precomputed tables file.
    /// To prepare `dest`, you should use an memory mapped file or a 32-byte aligned byte array.
    /// The byte array length should be the return value of [`table_file_len`].
    /// This function will report progress using the provided function.
    pub fn create_table_file_with_progress_report<P: ProgressTableGenerationReportFunction>(
        l1: usize,
        dest: &mut [u8],
        progress_report: P,
    ) -> io::Result<()> {
        let (t2_bytes, t1_bytes) = dest.split_at_mut(I_MAX * size_of::<T2MontgomeryCoordinates>());
        create_t2_table(l1, t2_bytes, &progress_report)?;
        create_t1_table(l1, t1_bytes, &progress_report)
    }

    /// Generate the ECDLP precomputed tables file, with multithreading.
    /// To prepare `dest`, you should use an memory mapped file or a 32-byte aligned byte array.
    /// The byte array length should be the return value of [`table_file_len`].
    /// This function will report progress using the provided function.
    pub fn create_table_file_with_progress_report_par<
        P: ProgressTableGenerationReportFunction + Sync,
    >(
        l1: usize,
        n_threads: usize,
        dest: &mut [u8],
        progress_report: P,
    ) -> io::Result<()> {
        let (t2_bytes, t1_bytes) = dest.split_at_mut(I_MAX * size_of::<T2MontgomeryCoordinates>());
        create_t2_table_par(l1, n_threads, t2_bytes, &progress_report)?;
        create_t1_table_par(l1, n_threads, t1_bytes, &progress_report)
    }
}
