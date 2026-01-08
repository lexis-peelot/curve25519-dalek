use crate::field::FieldElement;
use crate::backend::simd::SimdBackend;
use crate::ecdlp::simd_types::{U64x4 as u64x4};

// Generic 4-way batch subtraction
#[inline(always)]
fn batch_subtract_4way_impl<B: SimdBackend>(
    batch: &[FieldElement; 4],
    target: &FieldElement,
) -> [FieldElement; 4] {
    let mut results = [FieldElement::ZERO; 4];

    for limb_idx in 0..B::NUM_LIMBS {
        let batch_limbs = B::gather_limb_4(batch, limb_idx);
        let target_limb = B::splat_limb(target, limb_idx);
        let offset = B::offset_for_limb(limb_idx);

        let diff = batch_limbs + offset - target_limb;
        let diff_array = B::to_limb_array_4(diff);

        for i in 0..4 {
            results[i].0[limb_idx] = diff_array[i] as _;
        }
    }

    [
        &results[0] - &FieldElement::ZERO,
        &results[1] - &FieldElement::ZERO,
        &results[2] - &FieldElement::ZERO,
        &results[3] - &FieldElement::ZERO,
    ]
}

// Generic 4-way batch subtraction with vectorized targets
#[inline(always)]
#[allow(dead_code)]
fn batch_subtract_4way_vec_impl<B: SimdBackend>(
    batch: &[FieldElement; 4],
    targets: &[FieldElement; 4],
) -> [FieldElement; 4] {
    let mut results = [FieldElement::ZERO; 4];

    for limb_idx in 0..B::NUM_LIMBS {
        let batch_limbs = B::gather_limb_4(batch, limb_idx);
        let target_limbs = B::gather_limb_4(targets, limb_idx);
        let offset = B::offset_for_limb(limb_idx);

        let diff = batch_limbs + offset - target_limbs;
        let diff_array = B::to_limb_array_4(diff);

        for i in 0..4 {
            results[i].0[limb_idx] = diff_array[i] as _;
        }
    }

    [
        &results[0] - &FieldElement::ZERO,
        &results[1] - &FieldElement::ZERO,
        &results[2] - &FieldElement::ZERO,
        &results[3] - &FieldElement::ZERO,
    ]
}

// Generic 8-way batch subtraction
#[inline(always)]
#[allow(dead_code)]
fn batch_subtract_8way_impl<B: SimdBackend>(
    batch: &[FieldElement; 8],
    target: &FieldElement,
) -> [FieldElement; 8] {
    let mut results = [FieldElement::ZERO; 8];

    for limb_idx in 0..B::NUM_LIMBS {
        let batch_limbs = B::gather_limb_8(batch, limb_idx);
        let target_limb = B::splat_limb(target, limb_idx);
        let offset = B::offset_for_limb(limb_idx);

        let diff = batch_limbs + offset - target_limb;
        let diff_array = B::to_limb_array_8(diff);

        for i in 0..8 {
            results[i].0[limb_idx] = diff_array[i] as _;
        }
    }

    results.map(|r| &r - &FieldElement::ZERO)
}

// Generic 4-way batch addition
#[inline(always)]
fn batch_add_4way_impl<B: SimdBackend>(
    batch: &[FieldElement; 4],
    target: &FieldElement,
) -> [FieldElement; 4] {
    let mut results = [FieldElement::ZERO; 4];

    for limb_idx in 0..B::NUM_LIMBS {
        let batch_limbs = B::gather_limb_4(batch, limb_idx);
        let target_limb = B::splat_limb(target, limb_idx);

        let sum = batch_limbs + target_limb;
        let sum_array = B::to_limb_array_4(sum);

        for i in 0..4 {
            results[i].0[limb_idx] = sum_array[i] as _;
        }
    }

    [
        &results[0] + &FieldElement::ZERO,
        &results[1] + &FieldElement::ZERO,
        &results[2] + &FieldElement::ZERO,
        &results[3] + &FieldElement::ZERO,
    ]
}

// Generic 8-way batch addition
#[inline(always)]
#[allow(dead_code)]
fn batch_add_8way_impl<B: SimdBackend>(
    batch: &[FieldElement; 8],
    target: &FieldElement,
) -> [FieldElement; 8] {
    let mut results = [FieldElement::ZERO; 8];

    for limb_idx in 0..B::NUM_LIMBS {
        let batch_limbs = B::gather_limb_8(batch, limb_idx);
        let target_limb = B::splat_limb(target, limb_idx);

        let sum = batch_limbs + target_limb;
        let sum_array = B::to_limb_array_8(sum);

        for i in 0..8 {
            results[i].0[limb_idx] = sum_array[i] as _;
        }
    }

    results.map(|r| &r + &FieldElement::ZERO)
}

// Generic 4-way vector addition
#[inline(always)]
#[allow(dead_code)]
fn batch_vecadd_4way_impl<B: SimdBackend>(
    a: &[FieldElement; 4],
    b: &[FieldElement; 4],
) -> [FieldElement; 4] {
    let mut results = [FieldElement::ZERO; 4];

    for limb_idx in 0..B::NUM_LIMBS {
        let a_limbs = B::gather_limb_4(a, limb_idx);
        let b_limbs = B::gather_limb_4(b, limb_idx);

        let sum = a_limbs + b_limbs;
        let sum_array = B::to_limb_array_4(sum);

        for i in 0..4 {
            results[i].0[limb_idx] = sum_array[i] as _;
        }
    }

    results.map(|r| &r + &FieldElement::ZERO)
}

// Generic 8-way vector addition
#[inline(always)]
#[allow(dead_code)]
fn batch_vecadd_8way_impl<B: SimdBackend>(
    a: &[FieldElement; 8],
    b: &[FieldElement; 8],
) -> [FieldElement; 8] {
    let mut results = [FieldElement::ZERO; 8];

    for limb_idx in 0..B::NUM_LIMBS {
        let a_limbs = B::gather_limb_8(a, limb_idx);
        let b_limbs = B::gather_limb_8(b, limb_idx);

        let sum = a_limbs + b_limbs;
        let sum_array = B::to_limb_array_8(sum);

        for i in 0..8 {
            results[i].0[limb_idx] = sum_array[i] as _;
        }
    }

    results.map(|r| &r + &FieldElement::ZERO)
}

#[allow(dead_code)]
impl FieldElement {
    #[inline(always)]
    pub(crate) fn batch_subtract_4way(batch: &[Self; 4], target: &Self) -> [Self; 4] {
        batch_subtract_4way_impl::<crate::backend::simd::DefaultBackend>(batch, target)
    }

    #[inline]
    pub(crate) fn batch_subtract_4way_vec(batch: &[Self; 4], targets: &[Self; 4]) -> [Self; 4] {
        batch_subtract_4way_vec_impl::<crate::backend::simd::DefaultBackend>(batch, targets)
    }

    #[inline]
    pub(crate) fn batch_add_4way(batch: &[Self; 4], target: &Self) -> [Self; 4] {
        batch_add_4way_impl::<crate::backend::simd::DefaultBackend>(batch, target)
    }

    #[inline]
    pub(crate) fn batch_vecadd_4way(a: &[Self; 4], b: &[Self; 4]) -> [Self; 4] {
        batch_vecadd_4way_impl::<crate::backend::simd::DefaultBackend>(a, b)
    }

    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_subtract_8way(batch: &[Self; 8], target: &Self) -> [Self; 8] {
        batch_subtract_8way_impl::<crate::backend::simd::Backend8Way>(batch, target)
    }
    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_add_8way(batch: &[Self; 8], target: &Self) -> [Self; 8] {
        batch_add_8way_impl::<crate::backend::simd::Backend8Way>(batch, target)
    }
    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_vecadd_8way(a: &[Self; 8], b: &[Self; 8]) -> [Self; 8] {
        batch_vecadd_8way_impl::<crate::backend::simd::Backend8Way>(a, b)
    }

    #[inline]
    #[cfg(curve25519_dalek_bits = "64")]
    pub(crate) fn batch_mul_4way(a: &[Self; 4], b: &[Self; 4]) -> [Self; 4] {
        Self::batch_mul_4way_64bit(a, b)
    }

    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_mul_4way(_a: &[Self; 4], _b: &[Self; 4]) -> [Self; 4] {
        unimplemented!("32-bit SIMD multiplication not supported")
    }

    #[inline]
    #[cfg(curve25519_dalek_bits = "64")]
    pub(crate) fn batch_mul_4way_ct(a: &[Self; 4], b: &[Self; 4]) -> [Self; 4] {
        Self::batch_mul_4way_64bit_ct(a, b)
    }

    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_mul_4way_ct(_a: &[Self; 4], _b: &[Self; 4]) -> [Self; 4] {
        unimplemented!("32-bit SIMD multiplication not supported")
    }

    #[cfg(curve25519_dalek_bits = "64")]
    #[inline(always)]
    fn batch_mul_4way_64bit(a_batch: &[Self; 4], b_batch: &[Self; 4]) -> [Self; 4] {
        const LOW_51: u64 = (1 << 51) - 1;
        const FACTOR_19: u64x4 = u64x4::new([19, 19, 19, 19]);
        const MASK: u64x4 = u64x4::new([LOW_51, LOW_51, LOW_51, LOW_51]);

        #[inline(always)]
        fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
            const MASK_32: u64x4 = u64x4::new([0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
            
            let a_lo = a & MASK_32;
            let a_hi = a >> 32;
            let b_lo = b & MASK_32;
            let b_hi = b >> 32;
            
            let lo_lo = a_lo * b_lo;
            let lo_hi = a_lo * b_hi;
            let hi_lo = a_hi * b_lo;
            let hi_hi = a_hi * b_hi;
            
            let mid = lo_hi + hi_lo;
            let mid_lo = mid << 32;
            let mid_hi = mid >> 32;
            
            let res_lo: u64x4 = lo_lo + mid_lo;
            let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let res_hi: u64x4 = hi_hi + mid_hi + carry;
            
            (res_lo, res_hi)
        }

        #[inline(always)]
        fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
            let sum_lo = a_lo + b_lo;
            let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let sum_hi = a_hi + b_hi + carry;
            (sum_lo, sum_hi)
        }

        let mut a = [u64x4::splat(0); 5];
        let mut b = [u64x4::splat(0); 5];
        
        for i in 0..5 {
            a[i] = u64x4::new([
                a_batch[0].0[i], a_batch[1].0[i], 
                a_batch[2].0[i], a_batch[3].0[i]
            ]);
            b[i] = u64x4::new([
                b_batch[0].0[i], b_batch[1].0[i],
                b_batch[2].0[i], b_batch[3].0[i]
            ]);
        }

        let b1_19 = b[1] * FACTOR_19;
        let b2_19 = b[2] * FACTOR_19;
        let b3_19 = b[3] * FACTOR_19;
        let b4_19 = b[4] * FACTOR_19;

        let a0_b0 = mul64_to_128_simd(a[0], b[0]);
        let a0_b1 = mul64_to_128_simd(a[0], b[1]);
        let a0_b2 = mul64_to_128_simd(a[0], b[2]);
        let a0_b3 = mul64_to_128_simd(a[0], b[3]);
        let a0_b4 = mul64_to_128_simd(a[0], b[4]);

        let a1_b0 = mul64_to_128_simd(a[1], b[0]);
        let a1_b1 = mul64_to_128_simd(a[1], b[1]);
        let a1_b2 = mul64_to_128_simd(a[1], b[2]);
        let a1_b3 = mul64_to_128_simd(a[1], b[3]);
        let a1_b4_19 = mul64_to_128_simd(a[1], b4_19);

        let a2_b0 = mul64_to_128_simd(a[2], b[0]);
        let a2_b1 = mul64_to_128_simd(a[2], b[1]);
        let a2_b2 = mul64_to_128_simd(a[2], b[2]);
        let a2_b3_19 = mul64_to_128_simd(a[2], b3_19);
        let a2_b4_19 = mul64_to_128_simd(a[2], b4_19);

        let a3_b0 = mul64_to_128_simd(a[3], b[0]);
        let a3_b1 = mul64_to_128_simd(a[3], b[1]);
        let a3_b2_19 = mul64_to_128_simd(a[3], b2_19);
        let a3_b3_19 = mul64_to_128_simd(a[3], b3_19);
        let a3_b4_19 = mul64_to_128_simd(a[3], b4_19);

        let a4_b0 = mul64_to_128_simd(a[4], b[0]);
        let a4_b1_19 = mul64_to_128_simd(a[4], b1_19);
        let a4_b2_19 = mul64_to_128_simd(a[4], b2_19);
        let a4_b3_19 = mul64_to_128_simd(a[4], b3_19);
        let a4_b4_19 = mul64_to_128_simd(a[4], b4_19);

        let (c0_lo, c0_hi) = {
            let (lo, hi) = a0_b0;
            let (lo, hi) = add_128_simd(lo, hi, a4_b1_19.0, a4_b1_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a3_b2_19.0, a3_b2_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a2_b3_19.0, a2_b3_19.1);
            add_128_simd(lo, hi, a1_b4_19.0, a1_b4_19.1)
        };

        let (c1_lo, c1_hi) = {
            let (lo, hi) = a1_b0;
            let (lo, hi) = add_128_simd(lo, hi, a0_b1.0, a0_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a4_b2_19.0, a4_b2_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a3_b3_19.0, a3_b3_19.1);
            add_128_simd(lo, hi, a2_b4_19.0, a2_b4_19.1)
        };

        let (c2_lo, c2_hi) = {
            let (lo, hi) = a2_b0;
            let (lo, hi) = add_128_simd(lo, hi, a1_b1.0, a1_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a0_b2.0, a0_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a4_b3_19.0, a4_b3_19.1);
            add_128_simd(lo, hi, a3_b4_19.0, a3_b4_19.1)
        };

        let (c3_lo, c3_hi) = {
            let (lo, hi) = a3_b0;
            let (lo, hi) = add_128_simd(lo, hi, a2_b1.0, a2_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a1_b2.0, a1_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a0_b3.0, a0_b3.1);
            add_128_simd(lo, hi, a4_b4_19.0, a4_b4_19.1)
        };

        let (c4_lo, c4_hi) = {
            let (lo, hi) = a4_b0;
            let (lo, hi) = add_128_simd(lo, hi, a3_b1.0, a3_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a2_b2.0, a2_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a1_b3.0, a1_b3.1);
            add_128_simd(lo, hi, a0_b4.0, a0_b4.1)
        };

        let mut limb0 = c0_lo & MASK;
        let mut carry = (c0_hi << 13) | (c0_lo >> 51);

        macro_rules! propagate_carry {
            ($c_lo:expr, $c_hi:expr) => {{
                let acc: u64x4 = $c_lo + carry;
                let limb = acc & MASK;
                let mut new_carry = ($c_hi << 13) | (acc >> 51);
                
                // Check for overflow (rare: ~0.23% of cases)
                let overflow_mask = acc.cmp_lt($c_lo);
                if overflow_mask.to_array() != [0, 0, 0, 0] {
                    new_carry = new_carry + overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                }
                carry = new_carry;
                limb
            }};
        }

        let limb1: u64x4 = propagate_carry!(c1_lo, c1_hi);
        let limb2: u64x4 = propagate_carry!(c2_lo, c2_hi);
        let limb3: u64x4 = propagate_carry!(c3_lo, c3_hi);
        let limb4: u64x4 = propagate_carry!(c4_lo, c4_hi);

        limb0 = limb0 + carry * FACTOR_19;
        let carry5 = limb0 >> 51;
        limb0 = limb0 & MASK;
        let limb1: u64x4 = limb1 + carry5;

        let limb0_arr = limb0.to_array();
        let limb1_arr = limb1.to_array();
        let limb2_arr = limb2.to_array();
        let limb3_arr = limb3.to_array();
        let limb4_arr = limb4.to_array();

        [
            Self([limb0_arr[0], limb1_arr[0], limb2_arr[0], limb3_arr[0], limb4_arr[0]]),
            Self([limb0_arr[1], limb1_arr[1], limb2_arr[1], limb3_arr[1], limb4_arr[1]]),
            Self([limb0_arr[2], limb1_arr[2], limb2_arr[2], limb3_arr[2], limb4_arr[2]]),
            Self([limb0_arr[3], limb1_arr[3], limb2_arr[3], limb3_arr[3], limb4_arr[3]]),
        ]
    }

    #[cfg(curve25519_dalek_bits = "64")]
    #[inline(always)]
    fn batch_mul_4way_64bit_ct(a_batch: &[Self; 4], b_batch: &[Self; 4]) -> [Self; 4] {
        const LOW_51: u64 = (1 << 51) - 1;
        const FACTOR_19: u64x4 = u64x4::new([19, 19, 19, 19]);
        const MASK: u64x4 = u64x4::new([LOW_51, LOW_51, LOW_51, LOW_51]);

        #[inline(always)]
        fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
            const MASK_32: u64x4 = u64x4::new([0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
            
            let a_lo = a & MASK_32;
            let a_hi = a >> 32;
            let b_lo = b & MASK_32;
            let b_hi = b >> 32;
            
            let lo_lo = a_lo * b_lo;
            let lo_hi = a_lo * b_hi;
            let hi_lo = a_hi * b_lo;
            let hi_hi = a_hi * b_hi;
            
            let mid = lo_hi + hi_lo;
            let mid_lo = mid << 32;
            let mid_hi = mid >> 32;
            
            let res_lo: u64x4 = lo_lo + mid_lo;
            let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let res_hi: u64x4 = hi_hi + mid_hi + carry;
            
            (res_lo, res_hi)
        }

        #[inline(always)]
        fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
            let sum_lo = a_lo + b_lo;
            let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let sum_hi = a_hi + b_hi + carry;
            (sum_lo, sum_hi)
        }

        let mut a = [u64x4::splat(0); 5];
        let mut b = [u64x4::splat(0); 5];
        
        for i in 0..5 {
            a[i] = u64x4::new([
                a_batch[0].0[i], a_batch[1].0[i], 
                a_batch[2].0[i], a_batch[3].0[i]
            ]);
            b[i] = u64x4::new([
                b_batch[0].0[i], b_batch[1].0[i],
                b_batch[2].0[i], b_batch[3].0[i]
            ]);
        }

        let b1_19 = b[1] * FACTOR_19;
        let b2_19 = b[2] * FACTOR_19;
        let b3_19 = b[3] * FACTOR_19;
        let b4_19 = b[4] * FACTOR_19;

        let a0_b0 = mul64_to_128_simd(a[0], b[0]);
        let a0_b1 = mul64_to_128_simd(a[0], b[1]);
        let a0_b2 = mul64_to_128_simd(a[0], b[2]);
        let a0_b3 = mul64_to_128_simd(a[0], b[3]);
        let a0_b4 = mul64_to_128_simd(a[0], b[4]);

        let a1_b0 = mul64_to_128_simd(a[1], b[0]);
        let a1_b1 = mul64_to_128_simd(a[1], b[1]);
        let a1_b2 = mul64_to_128_simd(a[1], b[2]);
        let a1_b3 = mul64_to_128_simd(a[1], b[3]);
        let a1_b4_19 = mul64_to_128_simd(a[1], b4_19);

        let a2_b0 = mul64_to_128_simd(a[2], b[0]);
        let a2_b1 = mul64_to_128_simd(a[2], b[1]);
        let a2_b2 = mul64_to_128_simd(a[2], b[2]);
        let a2_b3_19 = mul64_to_128_simd(a[2], b3_19);
        let a2_b4_19 = mul64_to_128_simd(a[2], b4_19);

        let a3_b0 = mul64_to_128_simd(a[3], b[0]);
        let a3_b1 = mul64_to_128_simd(a[3], b[1]);
        let a3_b2_19 = mul64_to_128_simd(a[3], b2_19);
        let a3_b3_19 = mul64_to_128_simd(a[3], b3_19);
        let a3_b4_19 = mul64_to_128_simd(a[3], b4_19);

        let a4_b0 = mul64_to_128_simd(a[4], b[0]);
        let a4_b1_19 = mul64_to_128_simd(a[4], b1_19);
        let a4_b2_19 = mul64_to_128_simd(a[4], b2_19);
        let a4_b3_19 = mul64_to_128_simd(a[4], b3_19);
        let a4_b4_19 = mul64_to_128_simd(a[4], b4_19);

        let (c0_lo, c0_hi) = {
            let (lo, hi) = a0_b0;
            let (lo, hi) = add_128_simd(lo, hi, a4_b1_19.0, a4_b1_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a3_b2_19.0, a3_b2_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a2_b3_19.0, a2_b3_19.1);
            add_128_simd(lo, hi, a1_b4_19.0, a1_b4_19.1)
        };

        let (c1_lo, c1_hi) = {
            let (lo, hi) = a1_b0;
            let (lo, hi) = add_128_simd(lo, hi, a0_b1.0, a0_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a4_b2_19.0, a4_b2_19.1);
            let (lo, hi) = add_128_simd(lo, hi, a3_b3_19.0, a3_b3_19.1);
            add_128_simd(lo, hi, a2_b4_19.0, a2_b4_19.1)
        };

        let (c2_lo, c2_hi) = {
            let (lo, hi) = a2_b0;
            let (lo, hi) = add_128_simd(lo, hi, a1_b1.0, a1_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a0_b2.0, a0_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a4_b3_19.0, a4_b3_19.1);
            add_128_simd(lo, hi, a3_b4_19.0, a3_b4_19.1)
        };

        let (c3_lo, c3_hi) = {
            let (lo, hi) = a3_b0;
            let (lo, hi) = add_128_simd(lo, hi, a2_b1.0, a2_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a1_b2.0, a1_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a0_b3.0, a0_b3.1);
            add_128_simd(lo, hi, a4_b4_19.0, a4_b4_19.1)
        };

        let (c4_lo, c4_hi) = {
            let (lo, hi) = a4_b0;
            let (lo, hi) = add_128_simd(lo, hi, a3_b1.0, a3_b1.1);
            let (lo, hi) = add_128_simd(lo, hi, a2_b2.0, a2_b2.1);
            let (lo, hi) = add_128_simd(lo, hi, a1_b3.0, a1_b3.1);
            add_128_simd(lo, hi, a0_b4.0, a0_b4.1)
        };

        let mut limb0 = c0_lo & MASK;
        let mut carry = (c0_hi << 13) | (c0_lo >> 51);

        macro_rules! propagate_carry_ct {
            ($c_lo:expr, $c_hi:expr) => {{
                let acc: u64x4 = $c_lo + carry;
                let limb = acc & MASK;
                
                let overflow_mask = acc.cmp_lt($c_lo);
                let correction = overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                carry = ($c_hi << 13) | (acc >> 51) + correction;
                limb
            }};
        }


        let limb1: u64x4 = propagate_carry_ct!(c1_lo, c1_hi);
        let limb2: u64x4 = propagate_carry_ct!(c2_lo, c2_hi);
        let limb3: u64x4 = propagate_carry_ct!(c3_lo, c3_hi);
        let limb4: u64x4 = propagate_carry_ct!(c4_lo, c4_hi);

        limb0 = limb0 + carry * FACTOR_19;
        let carry5 = limb0 >> 51;
        limb0 = limb0 & MASK;
        let limb1: u64x4 = limb1 + carry5;

        let limb0_arr = limb0.to_array();
        let limb1_arr = limb1.to_array();
        let limb2_arr = limb2.to_array();
        let limb3_arr = limb3.to_array();
        let limb4_arr = limb4.to_array();

        [
            Self([limb0_arr[0], limb1_arr[0], limb2_arr[0], limb3_arr[0], limb4_arr[0]]),
            Self([limb0_arr[1], limb1_arr[1], limb2_arr[1], limb3_arr[1], limb4_arr[1]]),
            Self([limb0_arr[2], limb1_arr[2], limb2_arr[2], limb3_arr[2], limb4_arr[2]]),
            Self([limb0_arr[3], limb1_arr[3], limb2_arr[3], limb3_arr[3], limb4_arr[3]]),
        ]
    }

    /// Batch square 4 field elements simultaneously using SIMD
    #[inline]
    #[cfg(curve25519_dalek_bits = "64")]
    pub(crate) fn batch_square_4way(a: &[Self; 4]) -> [Self; 4] {
        Self::batch_square_4way_64bit(a)
    }

    /// Batch square 4 field elements simultaneously using SIMD (32-bit stub)
    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_square_4way(_a: &[Self; 4]) -> [Self; 4] {
        unimplemented!("32-bit SIMD squaring not yet implemented")
    }

    /// Constant-time batch square 4 field elements simultaneously using SIMD
    #[inline]
    #[cfg(curve25519_dalek_bits = "64")]
    pub(crate) fn batch_square_4way_ct(a: &[Self; 4]) -> [Self; 4] {
        Self::batch_square_4way_64bit_ct(a)
    }

    /// Constant-time batch square 4 field elements simultaneously using SIMD (32-bit stub)
    #[inline]
    #[cfg(curve25519_dalek_bits = "32")]
    pub(crate) fn batch_square_4way_ct(_a: &[Self; 4]) -> [Self; 4] {
        unimplemented!("32-bit SIMD squaring not yet implemented")
    }

    #[cfg(curve25519_dalek_bits = "64")]
    #[inline(always)]
    fn batch_square_4way_64bit(a_batch: &[Self; 4]) -> [Self; 4] {
        const LOW_51: u64 = (1 << 51) - 1;
        const FACTOR_19: u64x4 = u64x4::new([19, 19, 19, 19]);
        const MASK: u64x4 = u64x4::new([LOW_51, LOW_51, LOW_51, LOW_51]);

        #[inline(always)]
        fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
            const MASK_32: u64x4 = u64x4::new([0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
            
            let a_lo = a & MASK_32;
            let a_hi = a >> 32;
            let b_lo = b & MASK_32;
            let b_hi = b >> 32;
            
            let lo_lo = a_lo * b_lo;
            let lo_hi = a_lo * b_hi;
            let hi_lo = a_hi * b_lo;
            let hi_hi = a_hi * b_hi;
            
            let mid = lo_hi + hi_lo;
            let mid_lo = mid << 32;
            let mid_hi = mid >> 32;
            
            let res_lo: u64x4 = lo_lo + mid_lo;
            let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let res_hi: u64x4 = hi_hi + mid_hi + carry;
            
            (res_lo, res_hi)
        }

        #[inline(always)]
        fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
            let sum_lo = a_lo + b_lo;
            let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let sum_hi = a_hi + b_hi + carry;
            (sum_lo, sum_hi)
        }

        // Double a 128-bit value (shift left by 1)
        #[inline(always)]
        fn double_128_simd(lo: u64x4, hi: u64x4) -> (u64x4, u64x4) {
            let new_hi = (hi << 1) | (lo >> 63);
            let new_lo = lo << 1;
            (new_lo, new_hi)
        }

        // Load inputs into SIMD registers
        let mut a = [u64x4::splat(0); 5];
        for i in 0..5 {
            a[i] = u64x4::new([
                a_batch[0].0[i], a_batch[1].0[i], 
                a_batch[2].0[i], a_batch[3].0[i]
            ]);
        }

        // Pre-compute a*19 values
        let a3_19 = a[3] * FACTOR_19;
        let a4_19 = a[4] * FACTOR_19;

        // Compute unique products (exploiting squaring symmetry)
        let a0_sq = mul64_to_128_simd(a[0], a[0]);
        let a1_sq = mul64_to_128_simd(a[1], a[1]);
        let a2_sq = mul64_to_128_simd(a[2], a[2]);
        
        let a0_a1 = mul64_to_128_simd(a[0], a[1]);
        let a0_a2 = mul64_to_128_simd(a[0], a[2]);
        let a0_a3 = mul64_to_128_simd(a[0], a[3]);
        let a0_a4 = mul64_to_128_simd(a[0], a[4]);
        let a1_a2 = mul64_to_128_simd(a[1], a[2]);
        let a1_a3 = mul64_to_128_simd(a[1], a[3]);
        let a1_a4_19 = mul64_to_128_simd(a[1], a4_19);
        let a2_a3_19 = mul64_to_128_simd(a[2], a3_19);
        let a2_a4_19 = mul64_to_128_simd(a[2], a4_19);
        let a3_a3_19 = mul64_to_128_simd(a[3], a3_19);
        let a4_a3_19 = mul64_to_128_simd(a[4], a3_19);
        let a4_a4_19 = mul64_to_128_simd(a[4], a4_19);

        // Double the products that appear with coefficient 2
        let a0_a1_2 = double_128_simd(a0_a1.0, a0_a1.1);
        let a0_a2_2 = double_128_simd(a0_a2.0, a0_a2.1);
        let a0_a3_2 = double_128_simd(a0_a3.0, a0_a3.1);
        let a0_a4_2 = double_128_simd(a0_a4.0, a0_a4.1);
        let a1_a2_2 = double_128_simd(a1_a2.0, a1_a2.1);
        let a1_a3_2 = double_128_simd(a1_a3.0, a1_a3.1);
        let a1_a4_19_2 = double_128_simd(a1_a4_19.0, a1_a4_19.1);
        let a2_a3_19_2 = double_128_simd(a2_a3_19.0, a2_a3_19.1);
        let a2_a4_19_2 = double_128_simd(a2_a4_19.0, a2_a4_19.1);
        let a4_a3_19_2 = double_128_simd(a4_a3_19.0, a4_a3_19.1);

        // Compute c0 = a[0]^2 + 2*(a[1]*a4_19 + a[2]*a3_19)
        let (c0_lo, c0_hi) = {
            let (lo, hi) = a0_sq;
            let (lo, hi) = add_128_simd(lo, hi, a1_a4_19_2.0, a1_a4_19_2.1);
            add_128_simd(lo, hi, a2_a3_19_2.0, a2_a3_19_2.1)
        };

        // Compute c1 = a[3]*a3_19 + 2*(a[0]*a[1] + a[2]*a4_19)
        let (c1_lo, c1_hi) = {
            let (lo, hi) = a3_a3_19;
            let (lo, hi) = add_128_simd(lo, hi, a0_a1_2.0, a0_a1_2.1);
            add_128_simd(lo, hi, a2_a4_19_2.0, a2_a4_19_2.1)
        };

        // Compute c2 = a[1]^2 + 2*(a[0]*a[2] + a[4]*a3_19)
        let (c2_lo, c2_hi) = {
            let (lo, hi) = a1_sq;
            let (lo, hi) = add_128_simd(lo, hi, a0_a2_2.0, a0_a2_2.1);
            add_128_simd(lo, hi, a4_a3_19_2.0, a4_a3_19_2.1)
        };

        // Compute c3 = a[4]*a4_19 + 2*(a[0]*a[3] + a[1]*a[2])
        let (c3_lo, c3_hi) = {
            let (lo, hi) = a4_a4_19;
            let (lo, hi) = add_128_simd(lo, hi, a0_a3_2.0, a0_a3_2.1);
            add_128_simd(lo, hi, a1_a2_2.0, a1_a2_2.1)
        };

        // Compute c4 = a[2]^2 + 2*(a[0]*a[4] + a[1]*a[3])
        let (c4_lo, c4_hi) = {
            let (lo, hi) = a2_sq;
            let (lo, hi) = add_128_simd(lo, hi, a0_a4_2.0, a0_a4_2.1);
            add_128_simd(lo, hi, a1_a3_2.0, a1_a3_2.1)
        };

        // Carry propagation with branch prediction
        let mut limb0 = c0_lo & MASK;
        let mut carry = (c0_hi << 13) | (c0_lo >> 51);

        macro_rules! propagate_carry {
            ($c_lo:expr, $c_hi:expr) => {{
                let acc: u64x4 = $c_lo + carry;
                let limb = acc & MASK;
                let mut new_carry = ($c_hi << 13) | (acc >> 51);
                
                // Check for overflow (rare: ~0.23% of cases)
                let overflow_mask = acc.cmp_lt($c_lo);
                if overflow_mask.to_array() != [0, 0, 0, 0] {
                    new_carry = new_carry + overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                }
                carry = new_carry;
                limb
            }};
        }

        let limb1: u64x4 = propagate_carry!(c1_lo, c1_hi);
        let limb2: u64x4 = propagate_carry!(c2_lo, c2_hi);
        let limb3: u64x4 = propagate_carry!(c3_lo, c3_hi);
        let limb4: u64x4 = propagate_carry!(c4_lo, c4_hi);

        // Final reduction
        limb0 = limb0 + carry * FACTOR_19;
        let carry5 = limb0 >> 51;
        limb0 = limb0 & MASK;
        let limb1: u64x4 = limb1 + carry5;

        // Pack results
        let limb0_arr = limb0.to_array();
        let limb1_arr = limb1.to_array();
        let limb2_arr = limb2.to_array();
        let limb3_arr = limb3.to_array();
        let limb4_arr = limb4.to_array();

        [
            Self([limb0_arr[0], limb1_arr[0], limb2_arr[0], limb3_arr[0], limb4_arr[0]]),
            Self([limb0_arr[1], limb1_arr[1], limb2_arr[1], limb3_arr[1], limb4_arr[1]]),
            Self([limb0_arr[2], limb1_arr[2], limb2_arr[2], limb3_arr[2], limb4_arr[2]]),
            Self([limb0_arr[3], limb1_arr[3], limb2_arr[3], limb3_arr[3], limb4_arr[3]]),
        ]
    }

    #[cfg(curve25519_dalek_bits = "64")]
    #[inline(always)]
    fn batch_square_4way_64bit_ct(a_batch: &[Self; 4]) -> [Self; 4] {
        const LOW_51: u64 = (1 << 51) - 1;
        const FACTOR_19: u64x4 = u64x4::new([19, 19, 19, 19]);
        const MASK: u64x4 = u64x4::new([LOW_51, LOW_51, LOW_51, LOW_51]);

        #[inline(always)]
        fn mul64_to_128_simd(a: u64x4, b: u64x4) -> (u64x4, u64x4) {
            const MASK_32: u64x4 = u64x4::new([0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF]);
            
            let a_lo = a & MASK_32;
            let a_hi = a >> 32;
            let b_lo = b & MASK_32;
            let b_hi = b >> 32;
            
            let lo_lo = a_lo * b_lo;
            let lo_hi = a_lo * b_hi;
            let hi_lo = a_hi * b_lo;
            let hi_hi = a_hi * b_hi;
            
            let mid = lo_hi + hi_lo;
            let mid_lo = mid << 32;
            let mid_hi = mid >> 32;
            
            let res_lo: u64x4 = lo_lo + mid_lo;
            let carry = res_lo.cmp_lt(lo_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let res_hi: u64x4 = hi_hi + mid_hi + carry;
            
            (res_lo, res_hi)
        }

        #[inline(always)]
        fn add_128_simd(a_lo: u64x4, a_hi: u64x4, b_lo: u64x4, b_hi: u64x4) -> (u64x4, u64x4) {
            let sum_lo = a_lo + b_lo;
            let carry = sum_lo.cmp_lt(a_lo).blend(u64x4::splat(1), u64x4::splat(0));
            let sum_hi = a_hi + b_hi + carry;
            (sum_lo, sum_hi)
        }

        // Double a 128-bit value (shift left by 1)
        #[inline(always)]
        fn double_128_simd(lo: u64x4, hi: u64x4) -> (u64x4, u64x4) {
            let new_hi = (hi << 1) | (lo >> 63);
            let new_lo = lo << 1;
            (new_lo, new_hi)
        }

        // Load inputs into SIMD registers
        let mut a = [u64x4::splat(0); 5];
        for i in 0..5 {
            a[i] = u64x4::new([
                a_batch[0].0[i], a_batch[1].0[i], 
                a_batch[2].0[i], a_batch[3].0[i]
            ]);
        }

        // Pre-compute a*19 values
        let a3_19 = a[3] * FACTOR_19;
        let a4_19 = a[4] * FACTOR_19;

        // Compute unique products (exploiting squaring symmetry)
        let a0_sq = mul64_to_128_simd(a[0], a[0]);
        let a1_sq = mul64_to_128_simd(a[1], a[1]);
        let a2_sq = mul64_to_128_simd(a[2], a[2]);
        
        let a0_a1 = mul64_to_128_simd(a[0], a[1]);
        let a0_a2 = mul64_to_128_simd(a[0], a[2]);
        let a0_a3 = mul64_to_128_simd(a[0], a[3]);
        let a0_a4 = mul64_to_128_simd(a[0], a[4]);
        let a1_a2 = mul64_to_128_simd(a[1], a[2]);
        let a1_a3 = mul64_to_128_simd(a[1], a[3]);
        let a1_a4_19 = mul64_to_128_simd(a[1], a4_19);
        let a2_a3_19 = mul64_to_128_simd(a[2], a3_19);
        let a2_a4_19 = mul64_to_128_simd(a[2], a4_19);
        let a3_a3_19 = mul64_to_128_simd(a[3], a3_19);
        let a4_a3_19 = mul64_to_128_simd(a[4], a3_19);
        let a4_a4_19 = mul64_to_128_simd(a[4], a4_19);

        // Double the products that appear with coefficient 2
        let a0_a1_2 = double_128_simd(a0_a1.0, a0_a1.1);
        let a0_a2_2 = double_128_simd(a0_a2.0, a0_a2.1);
        let a0_a3_2 = double_128_simd(a0_a3.0, a0_a3.1);
        let a0_a4_2 = double_128_simd(a0_a4.0, a0_a4.1);
        let a1_a2_2 = double_128_simd(a1_a2.0, a1_a2.1);
        let a1_a3_2 = double_128_simd(a1_a3.0, a1_a3.1);
        let a1_a4_19_2 = double_128_simd(a1_a4_19.0, a1_a4_19.1);
        let a2_a3_19_2 = double_128_simd(a2_a3_19.0, a2_a3_19.1);
        let a2_a4_19_2 = double_128_simd(a2_a4_19.0, a2_a4_19.1);
        let a4_a3_19_2 = double_128_simd(a4_a3_19.0, a4_a3_19.1);

        // Compute c0 = a[0]^2 + 2*(a[1]*a4_19 + a[2]*a3_19)
        let (c0_lo, c0_hi) = {
            let (lo, hi) = a0_sq;
            let (lo, hi) = add_128_simd(lo, hi, a1_a4_19_2.0, a1_a4_19_2.1);
            add_128_simd(lo, hi, a2_a3_19_2.0, a2_a3_19_2.1)
        };

        // Compute c1 = a[3]*a3_19 + 2*(a[0]*a[1] + a[2]*a4_19)
        let (c1_lo, c1_hi) = {
            let (lo, hi) = a3_a3_19;
            let (lo, hi) = add_128_simd(lo, hi, a0_a1_2.0, a0_a1_2.1);
            add_128_simd(lo, hi, a2_a4_19_2.0, a2_a4_19_2.1)
        };

        // Compute c2 = a[1]^2 + 2*(a[0]*a[2] + a[4]*a3_19)
        let (c2_lo, c2_hi) = {
            let (lo, hi) = a1_sq;
            let (lo, hi) = add_128_simd(lo, hi, a0_a2_2.0, a0_a2_2.1);
            add_128_simd(lo, hi, a4_a3_19_2.0, a4_a3_19_2.1)
        };

        // Compute c3 = a[4]*a4_19 + 2*(a[0]*a[3] + a[1]*a[2])
        let (c3_lo, c3_hi) = {
            let (lo, hi) = a4_a4_19;
            let (lo, hi) = add_128_simd(lo, hi, a0_a3_2.0, a0_a3_2.1);
            add_128_simd(lo, hi, a1_a2_2.0, a1_a2_2.1)
        };

        // Compute c4 = a[2]^2 + 2*(a[0]*a[4] + a[1]*a[3])
        let (c4_lo, c4_hi) = {
            let (lo, hi) = a2_sq;
            let (lo, hi) = add_128_simd(lo, hi, a0_a4_2.0, a0_a4_2.1);
            add_128_simd(lo, hi, a1_a3_2.0, a1_a3_2.1)
        };

        // Carry propagation with branch prediction
        let mut limb0 = c0_lo & MASK;
        let mut carry = (c0_hi << 13) | (c0_lo >> 51);
        
        macro_rules! propagate_carry_ct {
            ($c_lo:expr, $c_hi:expr) => {{
                let acc: u64x4 = $c_lo + carry;
                let limb = acc & MASK;
                
                // Always apply overflow correction (branchless)
                let overflow_mask = acc.cmp_lt($c_lo);
                let correction = overflow_mask.blend(u64x4::splat(1 << 13), u64x4::splat(0));
                carry = ($c_hi << 13) | (acc >> 51) + correction;
                limb
            }};
        }
        
        let limb1: u64x4 = propagate_carry_ct!(c1_lo, c1_hi);
        let limb2: u64x4 = propagate_carry_ct!(c2_lo, c2_hi);
        let limb3: u64x4 = propagate_carry_ct!(c3_lo, c3_hi);
        let limb4: u64x4 = propagate_carry_ct!(c4_lo, c4_hi);

        // Final reduction
        limb0 = limb0 + carry * FACTOR_19;
        let carry5 = limb0 >> 51;
        limb0 = limb0 & MASK;
        let limb1: u64x4 = limb1 + carry5;

        // Pack results
        let limb0_arr = limb0.to_array();
        let limb1_arr = limb1.to_array();
        let limb2_arr = limb2.to_array();
        let limb3_arr = limb3.to_array();
        let limb4_arr = limb4.to_array();

        [
            Self([limb0_arr[0], limb1_arr[0], limb2_arr[0], limb3_arr[0], limb4_arr[0]]),
            Self([limb0_arr[1], limb1_arr[1], limb2_arr[1], limb3_arr[1], limb4_arr[1]]),
            Self([limb0_arr[2], limb1_arr[2], limb2_arr[2], limb3_arr[2], limb4_arr[2]]),
            Self([limb0_arr[3], limb1_arr[3], limb2_arr[3], limb3_arr[3], limb4_arr[3]]),
        ]
    }

    #[inline]
    pub fn batch_invert_4(elements: &mut [Self; 4]) {
        let mut acc = Self::ONE;
        let mut scratch = [Self::ONE; 4];
        
        for i in 0..4 {
            scratch[i] = acc;
            acc = &acc * &elements[i];
        }
        
        acc = acc.invert();
        
        for i in (0..4).rev() {
            let tmp = &acc * &scratch[i];
            acc = &acc * &elements[i];
            elements[i] = tmp;
        }
    }
}