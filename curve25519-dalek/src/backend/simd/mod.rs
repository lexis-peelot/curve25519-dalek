//! SIMD backend trait for field element batch operations
//!
//! This module provides a trait abstraction over different field element
//! representations (32-bit vs 64-bit limbs) to eliminate cfg checks in field_simd.rs

use crate::field::FieldElement;

/// Trait for SIMD-accelerated field element backends
#[allow(dead_code)]
pub trait SimdBackend: Sized {
    /// SIMD vector type (U64x4 for 64-bit batching 4 elements, U32x8 for 32-bit batching 8 elements)
    type Vector: Copy + core::ops::Add<Output = Self::Vector>
                       + core::ops::Sub<Output = Self::Vector>
                       + core::ops::Mul<Output = Self::Vector>;

    /// Number of field elements batched (4 for 64-bit, 8 for 32-bit)
    const BATCH_SIZE: usize;

    /// Number of limbs in field element representation (5 for 64-bit, 10 for 32-bit)
    const NUM_LIMBS: usize;

    /// Offset for limb 0 in batch subtraction
    const OFFSET_FIRST: Self::Vector;

    /// Get offset for limb at given index
    fn offset_for_limb(limb_idx: usize) -> Self::Vector;

    /// Create vector from array of field element limbs at given index
    fn gather_limb_4(fields: &[FieldElement; 4], limb_idx: usize) -> Self::Vector;

    /// Create vector from array of field element limbs (for 32-bit backend with 8 elements)
    fn gather_limb_8(fields: &[FieldElement; 8], limb_idx: usize) -> Self::Vector;

    /// Create vector with all lanes set to the same limb value
    fn splat_limb(field: &FieldElement, limb_idx: usize) -> Self::Vector;

    /// Extract vector back to array of limb values (4 elements for 64-bit)
    fn to_limb_array_4(vec: Self::Vector) -> [u64; 4];

    /// Extract vector back to array of limb values (8 elements for 32-bit)
    fn to_limb_array_8(vec: Self::Vector) -> [u64; 8];

    /// Perform comparison: vec < other
    fn cmp_lt(vec: Self::Vector, other: Self::Vector) -> Self::Vector;

    /// Blend: mask ? true_val : false_val
    fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector;
}

/// 64-bit backend using 5 limbs of 51 bits each, batching 4 field elements
pub struct Backend64;

#[cfg(curve25519_dalek_bits = "64")]
impl SimdBackend for Backend64 {
    type Vector = crate::ecdlp::simd_types::U64x4;
    const BATCH_SIZE: usize = 4;
    const NUM_LIMBS: usize = 5;

    const OFFSET_FIRST: Self::Vector = Self::Vector::new([36028797018963664u64; 4]);

    #[inline(always)]
    fn offset_for_limb(limb_idx: usize) -> Self::Vector {
        if limb_idx == 0 {
            Self::OFFSET_FIRST
        } else {
            Self::Vector::new([36028797018963952u64; 4])
        }
    }

    #[inline(always)]
    fn gather_limb_4(fields: &[FieldElement; 4], limb_idx: usize) -> Self::Vector {
        Self::Vector::from_array([
            fields[0].0[limb_idx],
            fields[1].0[limb_idx],
            fields[2].0[limb_idx],
            fields[3].0[limb_idx],
        ])
    }

    #[inline(always)]
    fn gather_limb_8(_fields: &[FieldElement; 8], _limb_idx: usize) -> Self::Vector {
        unimplemented!("Backend64 only supports 4-way batching")
    }

    #[inline(always)]
    fn splat_limb(field: &FieldElement, limb_idx: usize) -> Self::Vector {
        Self::Vector::splat(field.0[limb_idx])
    }

    #[inline(always)]
    fn to_limb_array_4(vec: Self::Vector) -> [u64; 4] {
        vec.to_array()
    }

    #[inline(always)]
    fn to_limb_array_8(_vec: Self::Vector) -> [u64; 8] {
        unimplemented!("Backend64 only supports 4-way batching")
    }

    #[inline(always)]
    fn cmp_lt(vec: Self::Vector, other: Self::Vector) -> Self::Vector {
        vec.cmp_lt(other)
    }

    #[inline(always)]
    fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        mask.blend(true_val, false_val)
    }
}

/// 32-bit backend using 10 limbs, batching 4 field elements with U32x4
#[allow(dead_code)]
pub struct Backend32x4;

#[cfg(curve25519_dalek_bits = "32")]
impl SimdBackend for Backend32x4 {
    type Vector = crate::ecdlp::simd_types::U32x4;
    const BATCH_SIZE: usize = 4;
    const NUM_LIMBS: usize = 10;

    const OFFSET_FIRST: Self::Vector = Self::Vector::new([(0x3ffffed << 4); 4]);

    #[inline(always)]
    fn offset_for_limb(limb_idx: usize) -> Self::Vector {
        if limb_idx == 0 {
            Self::OFFSET_FIRST
        } else if limb_idx % 2 == 0 {
            Self::Vector::new([(0x3ffffff << 4); 4])
        } else {
            Self::Vector::new([(0x1ffffff << 4); 4])
        }
    }

    #[inline(always)]
    fn gather_limb_4(fields: &[FieldElement; 4], limb_idx: usize) -> Self::Vector {
        Self::Vector::from_array([
            fields[0].0[limb_idx] as u32,
            fields[1].0[limb_idx] as u32,
            fields[2].0[limb_idx] as u32,
            fields[3].0[limb_idx] as u32,
        ])
    }

    #[inline(always)]
    fn gather_limb_8(_fields: &[FieldElement; 8], _limb_idx: usize) -> Self::Vector {
        unimplemented!("Backend32x4 only supports 4-way batching")
    }

    #[inline(always)]
    fn splat_limb(field: &FieldElement, limb_idx: usize) -> Self::Vector {
        Self::Vector::splat(field.0[limb_idx] as u32)
    }

    #[inline(always)]
    fn to_limb_array_4(vec: Self::Vector) -> [u64; 4] {
        let arr = vec.to_array();
        [arr[0] as u64, arr[1] as u64, arr[2] as u64, arr[3] as u64]
    }

    #[inline(always)]
    fn to_limb_array_8(_vec: Self::Vector) -> [u64; 8] {
        unimplemented!("Backend32x4 only supports 4-way batching")
    }

    #[inline(always)]
    fn cmp_lt(vec: Self::Vector, other: Self::Vector) -> Self::Vector {
        vec.cmp_lt(other)
    }

    #[inline(always)]
    fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        mask.blend(true_val, false_val)
    }
}

/// 32-bit backend using 10 limbs, batching 8 field elements with U32x8
#[allow(dead_code)]
pub struct Backend32;

#[cfg(curve25519_dalek_bits = "32")]
impl SimdBackend for Backend32 {
    type Vector = crate::ecdlp::simd_types::U32x8;
    const BATCH_SIZE: usize = 8;
    const NUM_LIMBS: usize = 10;

    const OFFSET_FIRST: Self::Vector = Self::Vector::new([(0x3ffffed << 4); 8]);

    #[inline(always)]
    fn offset_for_limb(limb_idx: usize) -> Self::Vector {
        if limb_idx == 0 {
            Self::OFFSET_FIRST
        } else if limb_idx % 2 == 0 {
            Self::Vector::new([(0x3ffffff << 4); 8])
        } else {
            Self::Vector::new([(0x1ffffff << 4); 8])
        }
    }

    #[inline(always)]
    fn gather_limb_4(_fields: &[FieldElement; 4], _limb_idx: usize) -> Self::Vector {
        unimplemented!("Backend32 only supports 8-way batching")
    }

    #[inline(always)]
    fn gather_limb_8(fields: &[FieldElement; 8], limb_idx: usize) -> Self::Vector {
        Self::Vector::from_array([
            fields[0].0[limb_idx] as u32,
            fields[1].0[limb_idx] as u32,
            fields[2].0[limb_idx] as u32,
            fields[3].0[limb_idx] as u32,
            fields[4].0[limb_idx] as u32,
            fields[5].0[limb_idx] as u32,
            fields[6].0[limb_idx] as u32,
            fields[7].0[limb_idx] as u32,
        ])
    }

    #[inline(always)]
    fn splat_limb(field: &FieldElement, limb_idx: usize) -> Self::Vector {
        Self::Vector::splat(field.0[limb_idx] as u32)
    }

    #[inline(always)]
    fn to_limb_array_4(_vec: Self::Vector) -> [u64; 4] {
        unimplemented!("Backend32 only supports 8-way batching")
    }

    #[inline(always)]
    fn to_limb_array_8(vec: Self::Vector) -> [u64; 8] {
        let arr = vec.to_array();
        [
            arr[0] as u64, arr[1] as u64, arr[2] as u64, arr[3] as u64,
            arr[4] as u64, arr[5] as u64, arr[6] as u64, arr[7] as u64,
        ]
    }

    #[inline(always)]
    fn cmp_lt(vec: Self::Vector, other: Self::Vector) -> Self::Vector {
        vec.cmp_lt(other)
    }

    #[inline(always)]
    fn blend(mask: Self::Vector, true_val: Self::Vector, false_val: Self::Vector) -> Self::Vector {
        mask.blend(true_val, false_val)
    }
}

/// Select the appropriate backend based on configuration
/// Default is 4-way batching for both 32-bit and 64-bit
#[cfg(curve25519_dalek_bits = "64")]
pub type DefaultBackend = Backend64;

#[cfg(curve25519_dalek_bits = "32")]
pub type DefaultBackend = Backend32x4;

/// 8-way backend (only available for 32-bit)
#[cfg(curve25519_dalek_bits = "32")]
pub type Backend8Way = Backend32;
