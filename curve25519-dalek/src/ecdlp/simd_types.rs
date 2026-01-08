//! Compile-time dispatched SIMD types.
//! These types use #[cfg(target_feature)] to select implementations.
//! Runtime dispatch happens at the algorithm level via #[multiversion].

use cfg_if::cfg_if;

// ============================================================================
// U64x4 - 4-lane u64 vector (unsigned)
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct U64x4(pub [u64; 4]);

#[allow(dead_code)]
impl U64x4 {
    pub const ZERO: Self = Self([0; 4]);

    #[inline(always)]
    pub const fn new(arr: [u64; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self([v, v, v, v])
    }

    #[inline(always)]
    pub fn broadcast(v: u64) -> Self {
        Self([v, v, v, v])
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 4] {
        self.0
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn cmp_lt(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn cmp_lt_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let sign_bit = _mm256_set1_epi64x(i64::MIN);
                let a_flipped = _mm256_xor_si256(a_val, sign_bit);
                let b_flipped = _mm256_xor_si256(b_val, sign_bit);
                let r = _mm256_cmpgt_epi64(b_flipped, a_flipped);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_lt_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;

            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = vcltq_u64(a0, b0);
                let r1 = vcltq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_lt_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_lt_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] < other.0[0] { u64::MAX } else { 0 },
                    if self.0[1] < other.0[1] { u64::MAX } else { 0 },
                    if self.0[2] < other.0[2] { u64::MAX } else { 0 },
                    if self.0[3] < other.0[3] { u64::MAX } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    pub fn cmp_eq(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn cmp_eq_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_cmpeq_epi64(a_val, b_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_eq_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;

            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = vceqq_u64(a0, b0);
                let r1 = vceqq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_eq_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_eq_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] == other.0[0] { u64::MAX } else { 0 },
                    if self.0[1] == other.0[1] { u64::MAX } else { 0 },
                    if self.0[2] == other.0[2] { u64::MAX } else { 0 },
                    if self.0[3] == other.0[3] { u64::MAX } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    /// Blend: self is the mask, t is selected when mask is true, f when false
    /// Signature matches `wide` library: mask.blend(true_val, false_val)
    pub fn blend(self, t: Self, f: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn blend_avx2(mask: &U64x4, t: &U64x4, f: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let mask_val = _mm256_loadu_si256(mask.0.as_ptr() as *const __m256i);
                let true_val = _mm256_loadu_si256(t.0.as_ptr() as *const __m256i);
                let false_val = _mm256_loadu_si256(f.0.as_ptr() as *const __m256i);
                // _mm256_blendv_epi8(a, b, mask): selects from b when mask bit is 1, from a when 0
                let r = _mm256_blendv_epi8(false_val, true_val, mask_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn blend_neon(mask: &U64x4, t: &U64x4, f: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;

            unsafe {
                let m0 = vld1q_u64(mask.0.as_ptr());
                let m1 = vld1q_u64(mask.0.as_ptr().add(2));
                let t0 = vld1q_u64(t.0.as_ptr());
                let t1 = vld1q_u64(t.0.as_ptr().add(2));
                let f0 = vld1q_u64(f.0.as_ptr());
                let f1 = vld1q_u64(f.0.as_ptr().add(2));
                // vbslq_u64(mask, true, false)
                let r0 = vbslq_u64(m0, t0, f0);
                let r1 = vbslq_u64(m1, t1, f1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { blend_avx2(&self, &t, &f) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { blend_neon(&self, &t, &f) }
            } else {
                Self([
                    if self.0[0] != 0 { t.0[0] } else { f.0[0] },
                    if self.0[1] != 0 { t.0[1] } else { f.0[1] },
                    if self.0[2] != 0 { t.0[2] } else { f.0[2] },
                    if self.0[3] != 0 { t.0[3] } else { f.0[3] },
                ])
            }
        }
    }
}

impl std::ops::Add for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn add_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_add_epi64(a_val, b_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn add_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = vaddq_u64(a0, b0);
                let r1 = vaddq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
            }
            out
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { add_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { add_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_add(other.0[0]),
                    self.0[1].wrapping_add(other.0[1]),
                    self.0[2].wrapping_add(other.0[2]),
                    self.0[3].wrapping_add(other.0[3]),
                ])
            }
        }
    }
}

impl std::ops::Sub for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn sub_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_sub_epi64(a_val, b_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn sub_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = vsubq_u64(a0, b0);
                let r1 = vsubq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { sub_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { sub_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_sub(other.0[0]),
                    self.0[1].wrapping_sub(other.0[1]),
                    self.0[2].wrapping_sub(other.0[2]),
                    self.0[3].wrapping_sub(other.0[3]),
                ])
            }
        }
    }
}


impl std::ops::Mul for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // wide library uses scalar multiplication even with AVX2 enabled
        // because there's no efficient 64x64 multiply in AVX2
        Self([
            self.0[0].wrapping_mul(other.0[0]),
            self.0[1].wrapping_mul(other.0[1]),
            self.0[2].wrapping_mul(other.0[2]),
            self.0[3].wrapping_mul(other.0[3]),
        ])
    }
}

impl std::ops::BitAnd for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitand_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_and_si256(a_val, b_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitand_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = vandq_u64(a0, b0);
                let r1 = vandq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitand_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitand_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] & other.0[0],
                    self.0[1] & other.0[1],
                    self.0[2] & other.0[2],
                    self.0[3] & other.0[3],
                ])
            }
        }
    }
}

impl std::ops::BitOr for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitor_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_or_si256(a_val, b_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitor_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = vorrq_u64(a0, b0);
                let r1 = vorrq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitor_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitor_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] | other.0[0],
                    self.0[1] | other.0[1],
                    self.0[2] | other.0[2],
                    self.0[3] | other.0[3],
                ])
            }
        }
    }
}

impl std::ops::BitXor for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitxor_avx2(a: &U64x4, b: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_xor_si256(a_val, b_val);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitxor_neon(a: &U64x4, b: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let b0 = vld1q_u64(b.0.as_ptr());
                let b1 = vld1q_u64(b.0.as_ptr().add(2));
                let r0 = veorq_u64(a0, b0);
                let r1 = veorq_u64(a1, b1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitxor_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitxor_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] ^ other.0[0],
                    self.0[1] ^ other.0[1],
                    self.0[2] ^ other.0[2],
                    self.0[3] ^ other.0[3],
                ])
            }
        }
    }
}

impl std::ops::Not for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn not_avx2(a: &U64x4) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let ones = _mm256_set1_epi64x(-1i64);
                let r = _mm256_xor_si256(a_val, ones);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn not_neon(a: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let r0 = vmvnq_u64(a0);
                let r1 = vmvnq_u64(a1);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn not_neon_macos(a: &U64x4) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                // Use XOR with all 1s as fallback for macOS
                let ones = vdupq_n_u64(u64::MAX);
                let r0 = veorq_u64(a0, ones);
                let r1 = veorq_u64(a1, ones);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { not_avx2(&self) }
            } else if #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))] {
                unsafe { not_neon(&self) }
            } else if #[cfg(all(target_arch = "aarch64", target_os = "macos"))] {
                unsafe { not_neon_macos(&self) }
            } else {
                Self([
                    !self.0[0],
                    !self.0[1],
                    !self.0[2],
                    !self.0[3],
                ])
            }
        }
    }
}

impl std::ops::Shl<i32> for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, shift: i32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn shl_avx2(a: &U64x4, shift: i32) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let shift_vec = _mm256_set1_epi64x(shift as i64);
                let r = _mm256_sllv_epi64(a_val, shift_vec);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn shl_neon(a: &U64x4, shift: i32) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let shift_vec = vdupq_n_s64(shift as i64);
                let r0 = vshlq_u64(a0, shift_vec);
                let r1 = vshlq_u64(a1, shift_vec);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { shl_avx2(&self, shift) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { shl_neon(&self, shift) }
            } else {
                Self([
                    self.0[0] << shift,
                    self.0[1] << shift,
                    self.0[2] << shift,
                    self.0[3] << shift,
                ])
            }
        }
    }
}

impl std::ops::Shr<i32> for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, shift: i32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn shr_avx2(a: &U64x4, shift: i32) -> U64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let shift_vec = _mm256_set1_epi64x(shift as i64);
                let r = _mm256_srlv_epi64(a_val, shift_vec);
                let mut out = U64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn shr_neon(a: &U64x4, shift: i32) -> U64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u64(a.0.as_ptr());
                let a1 = vld1q_u64(a.0.as_ptr().add(2));
                let shift_vec = vdupq_n_s64(-(shift as i64));
                let r0 = vshlq_u64(a0, shift_vec);
                let r1 = vshlq_u64(a1, shift_vec);
                let mut out = U64x4::ZERO;
                vst1q_u64(out.0.as_mut_ptr(), r0);
                vst1q_u64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { shr_avx2(&self, shift) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { shr_neon(&self, shift) }
            } else {
                Self([
                    self.0[0] >> shift,
                    self.0[1] >> shift,
                    self.0[2] >> shift,
                    self.0[3] >> shift,
                ])
            }
        }
    }
}


#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct I64x4(pub [i64; 4]);

#[allow(dead_code)]
impl I64x4 {
    pub const ZERO: Self = Self([0; 4]);

    #[inline(always)]
    pub const fn new(arr: [i64; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self([v, v, v, v])
    }

    #[inline(always)]
    pub fn broadcast(v: i64) -> Self {
        Self([v, v, v, v])
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 4] {
        self.0
    }

    #[inline(always)]
    pub fn from_array(arr: [i64; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn cmp_gt_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_cmpgt_epi64(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_gt_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = vcgtq_s64(a0, b0);
                let r1 = vcgtq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), vreinterpretq_s64_u64(r0));
                vst1q_s64(out.0.as_mut_ptr().add(2), vreinterpretq_s64_u64(r1));
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_gt_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_gt_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] > other.0[0] { -1 } else { 0 },
                    if self.0[1] > other.0[1] { -1 } else { 0 },
                    if self.0[2] > other.0[2] { -1 } else { 0 },
                    if self.0[3] > other.0[3] { -1 } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    pub fn cmp_lt(self, other: Self) -> Self {
        other.cmp_gt(self)
    }

    #[inline(always)]
    pub fn cmp_eq(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn cmp_eq_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_cmpeq_epi64(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_eq_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = vceqq_s64(a0, b0);
                let r1 = vceqq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), vreinterpretq_s64_u64(r0));
                vst1q_s64(out.0.as_mut_ptr().add(2), vreinterpretq_s64_u64(r1));
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_eq_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_eq_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] == other.0[0] { -1 } else { 0 },
                    if self.0[1] == other.0[1] { -1 } else { 0 },
                    if self.0[2] == other.0[2] { -1 } else { 0 },
                    if self.0[3] == other.0[3] { -1 } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    /// Blend: self is the mask, t is selected when mask is true, f when false
    /// Signature matches `wide` library: mask.blend(true_val, false_val)
    pub fn blend(self, t: Self, f: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn blend_avx2(mask: &I64x4, t: &I64x4, f: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let mask_val = _mm256_loadu_si256(mask.0.as_ptr() as *const __m256i);
                let true_val = _mm256_loadu_si256(t.0.as_ptr() as *const __m256i);
                let false_val = _mm256_loadu_si256(f.0.as_ptr() as *const __m256i);
                let r = _mm256_blendv_epi8(false_val, true_val, mask_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn blend_neon(mask: &I64x4, t: &I64x4, f: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let m0 = vld1q_u64(mask.0.as_ptr() as *const u64);
                let m1 = vld1q_u64(mask.0.as_ptr().add(2) as *const u64);
                let t0 = vld1q_s64(t.0.as_ptr());
                let t1 = vld1q_s64(t.0.as_ptr().add(2));
                let f0 = vld1q_s64(f.0.as_ptr());
                let f1 = vld1q_s64(f.0.as_ptr().add(2));
                let r0 = vbslq_s64(m0, t0, f0);
                let r1 = vbslq_s64(m1, t1, f1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { blend_avx2(&self, &t, &f) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { blend_neon(&self, &t, &f) }
            } else {
                Self([
                    if self.0[0] != 0 { t.0[0] } else { f.0[0] },
                    if self.0[1] != 0 { t.0[1] } else { f.0[1] },
                    if self.0[2] != 0 { t.0[2] } else { f.0[2] },
                    if self.0[3] != 0 { t.0[3] } else { f.0[3] },
                ])
            }
        }
    }
}

impl std::ops::Add for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn add_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_add_epi64(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn add_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = vaddq_s64(a0, b0);
                let r1 = vaddq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { add_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { add_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_add(other.0[0]),
                    self.0[1].wrapping_add(other.0[1]),
                    self.0[2].wrapping_add(other.0[2]),
                    self.0[3].wrapping_add(other.0[3]),
                ])
            }
        }
    }
}

impl std::ops::Sub for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn sub_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_sub_epi64(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn sub_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = vsubq_s64(a0, b0);
                let r1 = vsubq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { sub_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { sub_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_sub(other.0[0]),
                    self.0[1].wrapping_sub(other.0[1]),
                    self.0[2].wrapping_sub(other.0[2]),
                    self.0[3].wrapping_sub(other.0[3]),
                ])
            }
        }
    }
}

impl std::ops::Mul for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn mul_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let a_hi = _mm256_srli_epi64(a_val, 32);
                let b_hi = _mm256_srli_epi64(b_val, 32);
                let lo_lo = _mm256_mul_epu32(a_val, b_val);
                let lo_hi = _mm256_mul_epu32(a_val, b_hi);
                let hi_lo = _mm256_mul_epu32(a_hi, b_val);
                let mid = _mm256_add_epi64(lo_hi, hi_lo);
                let mid_shifted = _mm256_slli_epi64(mid, 32);
                let r = _mm256_add_epi64(lo_lo, mid_shifted);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { mul_avx2(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_mul(other.0[0]),
                    self.0[1].wrapping_mul(other.0[1]),
                    self.0[2].wrapping_mul(other.0[2]),
                    self.0[3].wrapping_mul(other.0[3]),
                ])
            }
        }
    }
}

impl std::ops::Neg for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn neg_avx2(a: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let zero = _mm256_setzero_si256();
                let r = _mm256_sub_epi64(zero, a_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn neg_neon(a: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let r0 = vnegq_s64(a0);
                let r1 = vnegq_s64(a1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { neg_avx2(&self) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { neg_neon(&self) }
            } else {
                Self([
                    self.0[0].wrapping_neg(),
                    self.0[1].wrapping_neg(),
                    self.0[2].wrapping_neg(),
                    self.0[3].wrapping_neg(),
                ])
            }
        }
    }
}

impl std::ops::BitAnd for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitand_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_and_si256(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitand_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = vandq_s64(a0, b0);
                let r1 = vandq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitand_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitand_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] & other.0[0],
                    self.0[1] & other.0[1],
                    self.0[2] & other.0[2],
                    self.0[3] & other.0[3],
                ])
            }
        }
    }
}

impl std::ops::BitOr for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitor_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_or_si256(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitor_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = vorrq_s64(a0, b0);
                let r1 = vorrq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitor_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitor_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] | other.0[0],
                    self.0[1] | other.0[1],
                    self.0[2] | other.0[2],
                    self.0[3] | other.0[3],
                ])
            }
        }
    }
}

impl std::ops::BitXor for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitxor_avx2(a: &I64x4, b: &I64x4) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_xor_si256(a_val, b_val);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitxor_neon(a: &I64x4, b: &I64x4) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let b0 = vld1q_s64(b.0.as_ptr());
                let b1 = vld1q_s64(b.0.as_ptr().add(2));
                let r0 = veorq_s64(a0, b0);
                let r1 = veorq_s64(a1, b1);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitxor_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitxor_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] ^ other.0[0],
                    self.0[1] ^ other.0[1],
                    self.0[2] ^ other.0[2],
                    self.0[3] ^ other.0[3],
                ])
            }
        }
    }
}

impl std::ops::Shr<i32> for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, amt: i32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn shr_avx2(a: &I64x4, amt: i32) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let shift_vec = _mm256_set1_epi64x(amt as i64);
                let zero = _mm256_setzero_si256();
                let sign = _mm256_cmpgt_epi64(zero, a_val);
                let shifted = _mm256_srlv_epi64(a_val, shift_vec);
                let sign_shift = _mm256_set1_epi64x((64 - amt) as i64);
                let sign_mask = _mm256_sllv_epi64(sign, sign_shift);
                let r = _mm256_or_si256(shifted, sign_mask);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn shr_neon(a: &I64x4, amt: i32) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let shift = vdupq_n_s64(-(amt as i64));
                let r0 = vshlq_s64(a0, shift);
                let r1 = vshlq_s64(a1, shift);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { shr_avx2(&self, amt) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { shr_neon(&self, amt) }
            } else {
                Self([
                    self.0[0] >> amt,
                    self.0[1] >> amt,
                    self.0[2] >> amt,
                    self.0[3] >> amt,
                ])
            }
        }
    }
}

impl std::ops::Shl<i32> for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, amt: i32) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn shl_avx2(a: &I64x4, amt: i32) -> I64x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let shift = _mm256_set1_epi64x(amt as i64);
                let r = _mm256_sllv_epi64(a_val, shift);
                let mut out = I64x4::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn shl_neon(a: &I64x4, amt: i32) -> I64x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_s64(a.0.as_ptr());
                let a1 = vld1q_s64(a.0.as_ptr().add(2));
                let shift = vdupq_n_s64(amt as i64);
                let r0 = vshlq_s64(a0, shift);
                let r1 = vshlq_s64(a1, shift);
                let mut out = I64x4::ZERO;
                vst1q_s64(out.0.as_mut_ptr(), r0);
                vst1q_s64(out.0.as_mut_ptr().add(2), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { shl_avx2(&self, amt) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { shl_neon(&self, amt) }
            } else {
                Self([
                    self.0[0] << amt,
                    self.0[1] << amt,
                    self.0[2] << amt,
                    self.0[3] << amt,
                ])
            }
        }
    }
}

impl From<[i64; 4]> for I64x4 {
    #[inline(always)]
    fn from(arr: [i64; 4]) -> Self {
        Self(arr)
    }
}

// ============================================================================
// U32x4 - 4-lane u32 vector
// ============================================================================

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct U32x4(pub [u32; 4]);

#[allow(dead_code)]
impl U32x4 {
    pub const ZERO: Self = Self([0; 4]);

    #[inline(always)]
    pub const fn new(arr: [u32; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self([v, v, v, v])
    }

    #[inline(always)]
    pub fn broadcast(v: u32) -> Self {
        Self([v, v, v, v])
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        self.0
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 4]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn cmp_eq(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn cmp_eq_sse(a: &U32x4, b: &U32x4) -> U32x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm_loadu_si128(a.0.as_ptr() as *const __m128i);
                let b_val = _mm_loadu_si128(b.0.as_ptr() as *const __m128i);
                let r = _mm_cmpeq_epi32(a_val, b_val);
                let mut out = U32x4::ZERO;
                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_eq_neon(a: &U32x4, b: &U32x4) -> U32x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a_val = vld1q_u32(a.0.as_ptr());
                let b_val = vld1q_u32(b.0.as_ptr());
                let r = vceqq_u32(a_val, b_val);
                let mut out = U32x4::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_eq_sse(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_eq_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] == other.0[0] { u32::MAX } else { 0 },
                    if self.0[1] == other.0[1] { u32::MAX } else { 0 },
                    if self.0[2] == other.0[2] { u32::MAX } else { 0 },
                    if self.0[3] == other.0[3] { u32::MAX } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    pub fn cmp_lt(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "sse2")]
        #[inline]
        unsafe fn cmp_lt_sse2(a: &U32x4, b: &U32x4) -> U32x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm_loadu_si128(a.0.as_ptr() as *const __m128i);
                let b_val = _mm_loadu_si128(b.0.as_ptr() as *const __m128i);
                // For unsigned comparison, we need to subtract 2^31 from both values
                let sign_bit = _mm_set1_epi32(i32::MIN);
                let a_flipped = _mm_xor_si128(a_val, sign_bit);
                let b_flipped = _mm_xor_si128(b_val, sign_bit);
                let r = _mm_cmplt_epi32(a_flipped, b_flipped);
                let mut out = U32x4::ZERO;
                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_lt_neon(a: &U32x4, b: &U32x4) -> U32x4 {
            use std::arch::aarch64::*;

            unsafe {
                let a_val = vld1q_u32(a.0.as_ptr());
                let b_val = vld1q_u32(b.0.as_ptr());
                let r = vcltq_u32(a_val, b_val);
                let mut out = U32x4::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_lt_sse2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_lt_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] < other.0[0] { u32::MAX } else { 0 },
                    if self.0[1] < other.0[1] { u32::MAX } else { 0 },
                    if self.0[2] < other.0[2] { u32::MAX } else { 0 },
                    if self.0[3] < other.0[3] { u32::MAX } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    pub fn blend(self, true_val: Self, false_val: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "sse4.1")]
        #[inline]
        unsafe fn blend_sse41(mask: &U32x4, true_val: &U32x4, false_val: &U32x4) -> U32x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let mask_val = _mm_loadu_si128(mask.0.as_ptr() as *const __m128i);
                let true_val_vec = _mm_loadu_si128(true_val.0.as_ptr() as *const __m128i);
                let false_val_vec = _mm_loadu_si128(false_val.0.as_ptr() as *const __m128i);
                let r = _mm_blendv_epi8(false_val_vec, true_val_vec, mask_val);
                let mut out = U32x4::ZERO;
                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn blend_neon(mask: &U32x4, true_val: &U32x4, false_val: &U32x4) -> U32x4 {
            use std::arch::aarch64::*;

            unsafe {
                let mask_val = vld1q_u32(mask.0.as_ptr());
                let true_val_vec = vld1q_u32(true_val.0.as_ptr());
                let false_val_vec = vld1q_u32(false_val.0.as_ptr());
                let r = vbslq_u32(mask_val, true_val_vec, false_val_vec);
                let mut out = U32x4::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { blend_sse41(&self, &true_val, &false_val) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { blend_neon(&self, &true_val, &false_val) }
            } else {
                Self([
                    if self.0[0] != 0 { true_val.0[0] } else { false_val.0[0] },
                    if self.0[1] != 0 { true_val.0[1] } else { false_val.0[1] },
                    if self.0[2] != 0 { true_val.0[2] } else { false_val.0[2] },
                    if self.0[3] != 0 { true_val.0[3] } else { false_val.0[3] },
                ])
            }
        }
    }
}

impl std::ops::Add for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn add_sse(a: &U32x4, b: &U32x4) -> U32x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm_loadu_si128(a.0.as_ptr() as *const __m128i);
                let b_val = _mm_loadu_si128(b.0.as_ptr() as *const __m128i);
                let r = _mm_add_epi32(a_val, b_val);
                let mut out = U32x4::ZERO;
                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn add_neon(a: &U32x4, b: &U32x4) -> U32x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a_val = vld1q_u32(a.0.as_ptr());
                let b_val = vld1q_u32(b.0.as_ptr());
                let r = vaddq_u32(a_val, b_val);
                let mut out = U32x4::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { add_sse(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { add_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_add(other.0[0]),
                    self.0[1].wrapping_add(other.0[1]),
                    self.0[2].wrapping_add(other.0[2]),
                    self.0[3].wrapping_add(other.0[3]),
                ])
            }
        }
    }
}

impl std::ops::Sub for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn sub_sse(a: &U32x4, b: &U32x4) -> U32x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm_loadu_si128(a.0.as_ptr() as *const __m128i);
                let b_val = _mm_loadu_si128(b.0.as_ptr() as *const __m128i);
                let r = _mm_sub_epi32(a_val, b_val);
                let mut out = U32x4::ZERO;
                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn sub_neon(a: &U32x4, b: &U32x4) -> U32x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a_val = vld1q_u32(a.0.as_ptr());
                let b_val = vld1q_u32(b.0.as_ptr());
                let r = vsubq_u32(a_val, b_val);
                let mut out = U32x4::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { sub_sse(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { sub_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_sub(other.0[0]),
                    self.0[1].wrapping_sub(other.0[1]),
                    self.0[2].wrapping_sub(other.0[2]),
                    self.0[3].wrapping_sub(other.0[3]),
                ])
            }
        }
    }
}

impl std::ops::Mul for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // Use scalar multiplication - no efficient 32x32 SIMD multiply in SSE2
        Self([
            self.0[0].wrapping_mul(other.0[0]),
            self.0[1].wrapping_mul(other.0[1]),
            self.0[2].wrapping_mul(other.0[2]),
            self.0[3].wrapping_mul(other.0[3]),
        ])
    }
}

impl std::ops::BitAnd for U32x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitand_sse(a: &U32x4, b: &U32x4) -> U32x4 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm_loadu_si128(a.0.as_ptr() as *const __m128i);
                let b_val = _mm_loadu_si128(b.0.as_ptr() as *const __m128i);
                let r = _mm_and_si128(a_val, b_val);
                let mut out = U32x4::ZERO;
                _mm_storeu_si128(out.0.as_mut_ptr() as *mut __m128i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitand_neon(a: &U32x4, b: &U32x4) -> U32x4 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a_val = vld1q_u32(a.0.as_ptr());
                let b_val = vld1q_u32(b.0.as_ptr());
                let r = vandq_u32(a_val, b_val);
                let mut out = U32x4::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitand_sse(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitand_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] & other.0[0],
                    self.0[1] & other.0[1],
                    self.0[2] & other.0[2],
                    self.0[3] & other.0[3],
                ])
            }
        }
    }
}

impl From<[u32; 4]> for U32x4 {
    #[inline(always)]
    fn from(arr: [u32; 4]) -> Self {
        Self(arr)
    }
}

// ============================================================================
// U32x8 - 8-lane u32 vector
// ============================================================================

#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct U32x8(pub [u32; 8]);

#[allow(dead_code)]
impl U32x8 {
    pub const ZERO: Self = Self([0; 8]);

    #[inline(always)]
    pub const fn new(arr: [u32; 8]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self([v, v, v, v, v, v, v, v])
    }

    #[inline(always)]
    pub fn broadcast(v: u32) -> Self {
        Self([v, v, v, v, v, v, v, v])
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 8] {
        self.0
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 8]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn cmp_eq(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn cmp_eq_avx2(a: &U32x8, b: &U32x8) -> U32x8 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_cmpeq_epi32(a_val, b_val);
                let mut out = U32x8::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn cmp_eq_neon(a: &U32x8, b: &U32x8) -> U32x8 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u32(a.0.as_ptr());
                let a1 = vld1q_u32(a.0.as_ptr().add(4));
                let b0 = vld1q_u32(b.0.as_ptr());
                let b1 = vld1q_u32(b.0.as_ptr().add(4));
                let r0 = vceqq_u32(a0, b0);
                let r1 = vceqq_u32(a1, b1);
                let mut out = U32x8::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r0);
                vst1q_u32(out.0.as_mut_ptr().add(4), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { cmp_eq_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { cmp_eq_neon(&self, &other) }
            } else {
                Self([
                    if self.0[0] == other.0[0] { u32::MAX } else { 0 },
                    if self.0[1] == other.0[1] { u32::MAX } else { 0 },
                    if self.0[2] == other.0[2] { u32::MAX } else { 0 },
                    if self.0[3] == other.0[3] { u32::MAX } else { 0 },
                    if self.0[4] == other.0[4] { u32::MAX } else { 0 },
                    if self.0[5] == other.0[5] { u32::MAX } else { 0 },
                    if self.0[6] == other.0[6] { u32::MAX } else { 0 },
                    if self.0[7] == other.0[7] { u32::MAX } else { 0 },
                ])
            }
        }
    }

    #[inline(always)]
    pub fn movemask(self) -> u32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn movemask_avx2(a: &U32x8) -> u32 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let a_ps = _mm256_castsi256_ps(a_val);
                _mm256_movemask_ps(a_ps) as u32
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { movemask_avx2(&self) }
            } else {
                let mut mask = 0u32;
                for i in 0..8 {
                    if self.0[i] & 0x80000000 != 0 {
                        mask |= 1 << i;
                    }
                }
                mask
            }
        }
    }

    #[inline(always)]
    pub fn any_nonzero(self) -> bool {
        self.0.iter().any(|&x| x != 0)
    }

    #[inline(always)]
    pub fn blend(self, true_val: Self, false_val: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn blend_avx2(mask: &U32x8, true_val: &U32x8, false_val: &U32x8) -> U32x8 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let mask_val = _mm256_loadu_si256(mask.0.as_ptr() as *const __m256i);
                let true_val_vec = _mm256_loadu_si256(true_val.0.as_ptr() as *const __m256i);
                let false_val_vec = _mm256_loadu_si256(false_val.0.as_ptr() as *const __m256i);
                let r = _mm256_blendv_epi8(false_val_vec, true_val_vec, mask_val);
                let mut out = U32x8::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn blend_neon(mask: &U32x8, true_val: &U32x8, false_val: &U32x8) -> U32x8 {
            use std::arch::aarch64::*;

            unsafe {
                let mask0 = vld1q_u32(mask.0.as_ptr());
                let mask1 = vld1q_u32(mask.0.as_ptr().add(4));
                let true_val0 = vld1q_u32(true_val.0.as_ptr());
                let true_val1 = vld1q_u32(true_val.0.as_ptr().add(4));
                let false_val0 = vld1q_u32(false_val.0.as_ptr());
                let false_val1 = vld1q_u32(false_val.0.as_ptr().add(4));
                let r0 = vbslq_u32(mask0, true_val0, false_val0);
                let r1 = vbslq_u32(mask1, true_val1, false_val1);
                let mut out = U32x8::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r0);
                vst1q_u32(out.0.as_mut_ptr().add(4), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { blend_avx2(&self, &true_val, &false_val) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { blend_neon(&self, &true_val, &false_val) }
            } else {
                Self([
                    if self.0[0] != 0 { true_val.0[0] } else { false_val.0[0] },
                    if self.0[1] != 0 { true_val.0[1] } else { false_val.0[1] },
                    if self.0[2] != 0 { true_val.0[2] } else { false_val.0[2] },
                    if self.0[3] != 0 { true_val.0[3] } else { false_val.0[3] },
                    if self.0[4] != 0 { true_val.0[4] } else { false_val.0[4] },
                    if self.0[5] != 0 { true_val.0[5] } else { false_val.0[5] },
                    if self.0[6] != 0 { true_val.0[6] } else { false_val.0[6] },
                    if self.0[7] != 0 { true_val.0[7] } else { false_val.0[7] },
                ])
            }
        }
    }
}

impl std::ops::Add for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn add_avx2(a: &U32x8, b: &U32x8) -> U32x8 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_add_epi32(a_val, b_val);
                let mut out = U32x8::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn add_neon(a: &U32x8, b: &U32x8) -> U32x8 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u32(a.0.as_ptr());
                let a1 = vld1q_u32(a.0.as_ptr().add(4));
                let b0 = vld1q_u32(b.0.as_ptr());
                let b1 = vld1q_u32(b.0.as_ptr().add(4));
                let r0 = vaddq_u32(a0, b0);
                let r1 = vaddq_u32(a1, b1);
                let mut out = U32x8::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r0);
                vst1q_u32(out.0.as_mut_ptr().add(4), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { add_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { add_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_add(other.0[0]),
                    self.0[1].wrapping_add(other.0[1]),
                    self.0[2].wrapping_add(other.0[2]),
                    self.0[3].wrapping_add(other.0[3]),
                    self.0[4].wrapping_add(other.0[4]),
                    self.0[5].wrapping_add(other.0[5]),
                    self.0[6].wrapping_add(other.0[6]),
                    self.0[7].wrapping_add(other.0[7]),
                ])
            }
        }
    }
}

impl std::ops::Sub for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn sub_avx2(a: &U32x8, b: &U32x8) -> U32x8 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_sub_epi32(a_val, b_val);
                let mut out = U32x8::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn sub_neon(a: &U32x8, b: &U32x8) -> U32x8 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u32(a.0.as_ptr());
                let a1 = vld1q_u32(a.0.as_ptr().add(4));
                let b0 = vld1q_u32(b.0.as_ptr());
                let b1 = vld1q_u32(b.0.as_ptr().add(4));
                let r0 = vsubq_u32(a0, b0);
                let r1 = vsubq_u32(a1, b1);
                let mut out = U32x8::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r0);
                vst1q_u32(out.0.as_mut_ptr().add(4), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { sub_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { sub_neon(&self, &other) }
            } else {
                Self([
                    self.0[0].wrapping_sub(other.0[0]),
                    self.0[1].wrapping_sub(other.0[1]),
                    self.0[2].wrapping_sub(other.0[2]),
                    self.0[3].wrapping_sub(other.0[3]),
                    self.0[4].wrapping_sub(other.0[4]),
                    self.0[5].wrapping_sub(other.0[5]),
                    self.0[6].wrapping_sub(other.0[6]),
                    self.0[7].wrapping_sub(other.0[7]),
                ])
            }
        }
    }
}

impl std::ops::Mul for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        // Use scalar multiplication - no efficient 32x32 SIMD multiply
        Self([
            self.0[0].wrapping_mul(other.0[0]),
            self.0[1].wrapping_mul(other.0[1]),
            self.0[2].wrapping_mul(other.0[2]),
            self.0[3].wrapping_mul(other.0[3]),
            self.0[4].wrapping_mul(other.0[4]),
            self.0[5].wrapping_mul(other.0[5]),
            self.0[6].wrapping_mul(other.0[6]),
            self.0[7].wrapping_mul(other.0[7]),
        ])
    }
}

impl std::ops::BitAnd for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitand_avx2(a: &U32x8, b: &U32x8) -> U32x8 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_and_si256(a_val, b_val);
                let mut out = U32x8::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitand_neon(a: &U32x8, b: &U32x8) -> U32x8 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u32(a.0.as_ptr());
                let a1 = vld1q_u32(a.0.as_ptr().add(4));
                let b0 = vld1q_u32(b.0.as_ptr());
                let b1 = vld1q_u32(b.0.as_ptr().add(4));
                let r0 = vandq_u32(a0, b0);
                let r1 = vandq_u32(a1, b1);
                let mut out = U32x8::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r0);
                vst1q_u32(out.0.as_mut_ptr().add(4), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitand_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitand_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] & other.0[0],
                    self.0[1] & other.0[1],
                    self.0[2] & other.0[2],
                    self.0[3] & other.0[3],
                    self.0[4] & other.0[4],
                    self.0[5] & other.0[5],
                    self.0[6] & other.0[6],
                    self.0[7] & other.0[7],
                ])
            }
        }
    }
}

impl std::ops::BitOr for U32x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, other: Self) -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn bitor_avx2(a: &U32x8, b: &U32x8) -> U32x8 {
            #[cfg(target_arch = "x86_64")]
            use std::arch::x86_64::*;
            #[cfg(target_arch = "x86")]
            use std::arch::x86::*;

            unsafe {
                let a_val = _mm256_loadu_si256(a.0.as_ptr() as *const __m256i);
                let b_val = _mm256_loadu_si256(b.0.as_ptr() as *const __m256i);
                let r = _mm256_or_si256(a_val, b_val);
                let mut out = U32x8::ZERO;
                _mm256_storeu_si256(out.0.as_mut_ptr() as *mut __m256i, r);
                out
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn bitor_neon(a: &U32x8, b: &U32x8) -> U32x8 {
            use std::arch::aarch64::*;
            
            unsafe {
                let a0 = vld1q_u32(a.0.as_ptr());
                let a1 = vld1q_u32(a.0.as_ptr().add(4));
                let b0 = vld1q_u32(b.0.as_ptr());
                let b1 = vld1q_u32(b.0.as_ptr().add(4));
                let r0 = vorrq_u32(a0, b0);
                let r1 = vorrq_u32(a1, b1);
                let mut out = U32x8::ZERO;
                vst1q_u32(out.0.as_mut_ptr(), r0);
                vst1q_u32(out.0.as_mut_ptr().add(4), r1);
                out
            }
        }

        cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                unsafe { bitor_avx2(&self, &other) }
            } else if #[cfg(target_arch = "aarch64")] {
                unsafe { bitor_neon(&self, &other) }
            } else {
                Self([
                    self.0[0] | other.0[0],
                    self.0[1] | other.0[1],
                    self.0[2] | other.0[2],
                    self.0[3] | other.0[3],
                    self.0[4] | other.0[4],
                    self.0[5] | other.0[5],
                    self.0[6] | other.0[6],
                    self.0[7] | other.0[7],
                ])
            }
        }
    }
}

impl From<[u32; 8]> for U32x8 {
    #[inline(always)]
    fn from(arr: [u32; 8]) -> Self {
        Self(arr)
    }
}

// ============================================================================
// CmpGt trait for wide compatibility
// ============================================================================

#[allow(dead_code)]
pub trait CmpGt<Rhs = Self> {
    type Output;
    fn cmp_gt(self, other: Rhs) -> Self::Output;
}

impl CmpGt for I64x4 {
    type Output = Self;
    #[inline(always)]
    fn cmp_gt(self, other: Self) -> Self::Output {
        I64x4::cmp_gt(self, other)
    }
}

// ============================================================================
// Type aliases for wide compatibility
// ============================================================================

pub use I64x4 as i64x4;
pub use U32x4 as u32x4;
pub use U32x8 as u32x8;
pub use U64x4 as u64x4;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_u64x4_basic() {
        let a = U64x4::new([1, 2, 3, 4]);
        let b = U64x4::new([10, 20, 30, 40]);
        
        let sum = a + b;
        assert_eq!(sum.to_array(), [11, 22, 33, 44]);
        
        let diff = b - a;
        assert_eq!(diff.to_array(), [9, 18, 27, 36]);
        
        let prod = a * b;
        assert_eq!(prod.to_array(), [10, 40, 90, 160]);
    }

    #[test]
    fn test_u64x4_bitops() {
        let a = U64x4::new([0xFF, 0xFF00, 0xFF0000, 0xFF000000]);
        let b = U64x4::new([0x0F, 0x0F00, 0x0F0000, 0x0F000000]);
        
        assert_eq!((a & b).to_array(), [0x0F, 0x0F00, 0x0F0000, 0x0F000000]);
        assert_eq!((a | b).to_array(), [0xFF, 0xFF00, 0xFF0000, 0xFF000000]);
    }

    #[test]
    fn test_u64x4_shifts() {
        let a = U64x4::new([0x100, 0x200, 0x300, 0x400]);
        assert_eq!((a >> 4).to_array(), [0x10, 0x20, 0x30, 0x40]);
        assert_eq!((a << 4).to_array(), [0x1000, 0x2000, 0x3000, 0x4000]);
    }

    #[test]
    fn test_i64x4_basic() {
        let a = I64x4::new([1, -2, 3, -4]);
        let b = I64x4::new([10, 20, -30, -40]);
        
        let sum = a + b;
        assert_eq!(sum.to_array(), [11, 18, -27, -44]);
        
        let neg = -a;
        assert_eq!(neg.to_array(), [-1, 2, -3, 4]);
    }

    #[test]
    fn test_i64x4_cmp() {
        let a = I64x4::new([1, 20, -3, -40]);
        let b = I64x4::new([10, 2, -30, -4]);
        
        let gt = a.cmp_gt(b);
        assert_eq!(gt.to_array()[0], 0);  // 1 <= 10
        assert_eq!(gt.to_array()[1], -1); // 20 > 2
        assert_eq!(gt.to_array()[2], -1); // -3 > -30
        assert_eq!(gt.to_array()[3], 0);  // -40 <= -4
    }

    #[test]
    fn test_u32x8_basic() {
        let a = U32x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = U32x8::new([10, 20, 30, 40, 50, 60, 70, 80]);
        
        let sum = a + b;
        assert_eq!(sum.to_array(), [11, 22, 33, 44, 55, 66, 77, 88]);
    }

    #[test]
    fn test_u32x8_cmp_eq() {
        let a = U32x8::new([1, 2, 3, 4, 5, 6, 7, 8]);
        let b = U32x8::new([1, 20, 3, 40, 5, 60, 7, 80]);
        
        let eq = a.cmp_eq(b);
        let arr = eq.to_array();
        assert_eq!(arr[0], u32::MAX);
        assert_eq!(arr[1], 0);
        assert_eq!(arr[2], u32::MAX);
        assert_eq!(arr[3], 0);
    }
}