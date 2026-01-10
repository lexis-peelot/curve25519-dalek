use crate::{EdwardsPoint, constants::MONTGOMERY_A, field::FieldElement};

/// An affine point on a Montgomery curve in (u, v) coordinates.
#[derive(Clone, Copy, Debug)]
pub struct AffineMontgomeryPoint {
    pub(crate) u: FieldElement,
    pub(crate) v: FieldElement,
}

impl AffineMontgomeryPoint {
    /// Check if the point is the identity point (point at infinity) in non-constant time.
    pub fn is_identity_not_ct(&self) -> bool {
        self.u == FieldElement::ZERO && self.v == FieldElement::ZERO
    }

    /// Return the identity point (point at infinity).
    #[inline(always)]
    pub fn identity() -> Self {
        Self {
            u: FieldElement::ZERO,
            v: FieldElement::ZERO,
        }
    }

    /// Create an `AffineMontgomeryPoint` from byte arrays representing the u and v coordinates.
    pub fn from_bytes(u: &[u8; 32], v: &[u8; 32]) -> Self {
        Self {
            u: FieldElement::from_bytes(u),
            v: FieldElement::from_bytes(v),
        }
    }

    /// Convert a batch of `EdwardsPoint`s to `AffineMontgomeryPoint`s.
    pub fn from_points<const N: usize>(points: [&EdwardsPoint; N]) -> [Self; N] {
        // u = (1+y)/(1-y) = (Z+Y)/(Z-Y),
        // v = (1+y)/(x(1-y)) * alpha = (Z+Y)/(X-T) * alpha.
        let mut t = [FieldElement::ZERO; N];
        let mut x_minus_t = [FieldElement::ZERO; N];
        let mut y = [FieldElement::ZERO; N];
        let mut z_minus_y = [FieldElement::ZERO; N];
        let mut z_plus_y = [FieldElement::ZERO; N];

        for (i, point) in points.iter().enumerate() {
            t[i] = point.T;
            y[i] = point.Y;

            x_minus_t[i] = point.X;
            z_minus_y[i] = point.Z;
            z_plus_y[i] = point.Z;
        }

        FieldElement::batch_add_n(&mut z_plus_y, &y);
        FieldElement::batch_subtract_n(&mut z_minus_y, &y);
        FieldElement::batch_subtract_n(&mut x_minus_t, &t);

        FieldElement::invert_batch(&mut z_minus_y);
        FieldElement::invert_batch(&mut x_minus_t);

        let mut u_coords = z_plus_y;
        FieldElement::batch_mul(&mut u_coords, &z_minus_y);

        let mut v_coords = z_plus_y;
        FieldElement::batch_mul(&mut v_coords, &x_minus_t);
        FieldElement::batch_mul(&mut v_coords, &[ALPHA; N]);

        let mut result = [Self::identity(); N];
        for (i, (u, v)) in u_coords.into_iter().zip(v_coords.into_iter()).enumerate() {
            result[i] = Self { u, v };
        }

        result
    }

    /// Add two `AffineMontgomeryPoint` together.
    pub fn addition_not_ct(&self, p2: &Self) -> AffineMontgomeryPoint {
        let p1 = self;
        if p1.is_identity_not_ct() {
            // p2 + P_inf = p2
            *p2
        } else if p2.is_identity_not_ct() {
            // p1 + P_inf = p1
            *p1
        } else if p1.u == p2.u && p1.v == -&p2.v {
            // p1 = -p2 = (u1, -v1), meaning p1 + p2 = P_inf
            Self::identity()
        } else {
            let lambda = if p1.u == p2.u {
                // doubling case

                // (3*u1^2 + 2*A*u1 + 1) / (2*v1)
                // todo this is ugly
                let u1_sq = p1.u.square();
                let u1_sq_3 = &(&u1_sq + &u1_sq) + &u1_sq;
                let u1_ta = &MONTGOMERY_A * &p1.u;
                let u1_ta_2 = &u1_ta + &u1_ta;
                let den = &p1.v + &p1.v;
                let num = &(&u1_sq_3 + &u1_ta_2) + &FieldElement::ONE;

                &num * &den.invert()
            } else {
                // (v1 - v2) / (u1 - u2)
                &(&p1.v - &p2.v) * &(&p1.u - &p2.u).invert()
            };

            // u3 = lambda^2 - A - u1 - u2
            // v3 = lambda * (u1 - u3) - v1
            let new_u = &(&lambda.square() - &MONTGOMERY_A) - &(&p1.u + &p2.u);
            let new_v = &(&lambda * &(&p1.u - &new_u)) - &p1.v;

            AffineMontgomeryPoint { u: new_u, v: new_v }
        }
    }

    /// Add the same point to N different points simultaneously
    pub fn batch_addition_not_ct<const N: usize>(points: &[Self; N], addend: &Self) -> [Self; N] {
        // Early exit checks for identity
        if addend.is_identity_not_ct() {
            return *points;
        }

        // Check if any input points are identity
        let mut results = [Self::identity(); N];
        let mut masks = [false; N];

        // Extract u and v coordinates for batch operations
        let mut u_coords_origin = [FieldElement::ZERO; N];
        let mut v_coords_origin = [FieldElement::ZERO; N];

        for (i, point) in points.iter().enumerate() {
            u_coords_origin[i] = point.u;
            v_coords_origin[i] = point.v;
        }

        // Check for inverse points (u1 == u2 && v1 == -v2)
        let mut u_coords = u_coords_origin;
        FieldElement::batch_subtract(&mut u_coords, &addend.u);

        let mut v_coords = v_coords_origin;
        FieldElement::batch_add(&mut v_coords, &addend.v);

        // Compute denominators for lambda
        let mut denominators = [FieldElement::ZERO; N];
        let mut numerators = [FieldElement::ZERO; N];

        for ((i, u_coord), v_coord) in u_coords.iter().enumerate().zip(v_coords.iter()) {
            let point = &points[i];
            if point.is_identity_not_ct() {
                masks[i] = true;
                results[i] = *addend;
                continue;
            }

            if *u_coord == FieldElement::ZERO {
                if *v_coord == FieldElement::ZERO {
                    // Point at infinity case
                    masks[i] = true;
                } else {
                    // Doubling case
                    // (3*u1^2 + 2*A*u1 + 1)
                    let point = &points[i];
                    let u_ta = &MONTGOMERY_A * &point.u;

                    let u_sq = point.u.square();

                    // last item is dummy, just in case we support SIMD
                    let mut tmp = [u_sq, u_ta, point.v, FieldElement::ZERO];
                    FieldElement::batch_add_n(&mut tmp, &[u_sq, u_ta, point.v, FieldElement::ZERO]);

                    let [u_sq_2, u_ta_2, point_v_2, _] = tmp;
                    let u_sq_3 = &u_sq_2 + &u_sq;

                    numerators[i] = &(&u_sq_3 + &u_ta_2) + &FieldElement::ONE;

                    // Denominator is 2*v1
                    denominators[i] = point_v_2;
                }
            } else {
                // (v1 - v2)
                numerators[i] = &points[i].v - &addend.v;
                // Regular addition case
                denominators[i] = *u_coord;
            }
        }

        // Batch invert denominators
        let mut inv_denominators = denominators;
        FieldElement::invert_batch(&mut inv_denominators);

        // Compute lambdas using batch multiplication
        FieldElement::batch_mul(&mut numerators, &inv_denominators);

        let mut lambdas = numerators;

        // Square lambdas
        FieldElement::batch_square(&mut numerators);

        // Compute lambda^2 - A
        FieldElement::batch_subtract(&mut numerators, &MONTGOMERY_A);

        let mut u_coords_plus_addend = u_coords_origin;
        FieldElement::batch_add(&mut u_coords_plus_addend, &addend.u);
        FieldElement::batch_subtract_n(&mut numerators, &u_coords_plus_addend);

        // Compute u1 - u3 for each point
        let mut u_diffs_for_v = u_coords_origin;
        FieldElement::batch_subtract_n(&mut u_diffs_for_v, &numerators);

        // Compute new v coordinates: lambda * (u1 - u3) - v1
        FieldElement::batch_mul(&mut lambdas, &u_diffs_for_v);

        // Compute then subtract v1
        FieldElement::batch_subtract_n(&mut lambdas, &v_coords_origin);

        // Assemble results
        for (i, ((u, v), is_masked)) in numerators
            .into_iter()
            .zip(lambdas)
            .zip(masks)
            .enumerate()
        {
            if !is_masked {
                results[i] = Self {
                    u,
                    v,
                };
            }
        }

        results
    }
}

// see test for correctness of this const
// Constant comes from https://ristretto.group/details/isogenies.html (birational mapping from E2 = E_(a2,d2) to M_(B,A))
// alpha = sqrt((A + 2) / (B * a_2)) with B = 1 and a_2 = -1.
const ALPHA: FieldElement = FieldElement::from_bytes(&[
    6, 126, 69, 255, 170, 4, 110, 204, 130, 26, 125, 75, 209, 211, 161, 197, 126, 79, 252, 3, 220,
    8, 123, 210, 187, 6, 160, 96, 244, 237, 38, 15,
]);

impl From<&'_ EdwardsPoint> for AffineMontgomeryPoint {
    #[allow(non_snake_case)]
    fn from(eddy: &EdwardsPoint) -> Self {
        // u = (1+y)/(1-y) = (Z+Y)/(Z-Y),
        // v = (1+y)/(x(1-y)) * alpha = (Z+Y)/(X-T) * alpha.
        let Z_plus_Y = &eddy.Z + &eddy.Y;
        let Z_minus_Y = &eddy.Z - &eddy.Y;
        let X_minus_T = &eddy.X - &eddy.T;

        let mut tmp = [Z_minus_Y, X_minus_T];
        FieldElement::invert_batch(&mut tmp);

        Self {
            u: &Z_plus_Y * &tmp[0],
            v: &(&Z_plus_Y * &tmp[1]) * &ALPHA,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, rng};

    use super::*;
    use crate::Scalar;

    #[test]
    fn test_const_alpha() {
        // Constant comes from https://ristretto.group/details/isogenies.html (birational mapping from E2 = E_(a2,d2) to M_(B,A))
        // alpha = sqrt((A + 2) / (B * a_2)) with B = 1 and a_2 = -1.
        let two = &FieldElement::ONE + &FieldElement::ONE;
        let (is_sq, v) =
            FieldElement::sqrt_ratio_i(&(&MONTGOMERY_A + &two), &FieldElement::MINUS_ONE);
        assert!(bool::from(is_sq));

        assert_eq!(ALPHA.to_bytes(), v.to_bytes());
    }

    #[test]
    fn test_batch_addition_not_ct() {
        // Create test points by converting from Edwards points
        let ed_p1 = EdwardsPoint::mul_base(&Scalar::from(2u64));
        let ed_p2 = EdwardsPoint::mul_base(&Scalar::from(3u64));
        let ed_p3 = EdwardsPoint::mul_base(&Scalar::from(5u64));
        let ed_p4 = EdwardsPoint::mul_base(&Scalar::from(7u64));
        let ed_addend = EdwardsPoint::mul_base(&Scalar::from(11u64));
        let p1 = AffineMontgomeryPoint::from(&ed_p1);
        let p2 = AffineMontgomeryPoint::from(&ed_p2);
        let p3 = AffineMontgomeryPoint::from(&ed_p3);
        let p4 = AffineMontgomeryPoint::from(&ed_p4);
        let addend = AffineMontgomeryPoint::from(&ed_addend);

        // Test batch addition
        let points = [p1, p2, p3, p4, p4];
        let batch_results = AffineMontgomeryPoint::batch_addition_not_ct(&points, &addend);

        // Compare with individual additions
        let individual_results = [
            p1.addition_not_ct(&addend),
            p2.addition_not_ct(&addend),
            p3.addition_not_ct(&addend),
            p4.addition_not_ct(&addend),
            p4.addition_not_ct(&addend),
        ];

        for i in 0..5 {
            assert_eq!(
                batch_results[i].u.to_bytes(),
                individual_results[i].u.to_bytes(),
                "batch_addition u mismatch at index {}",
                i
            );
            assert_eq!(
                batch_results[i].v.to_bytes(),
                individual_results[i].v.to_bytes(),
                "batch_addition v mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_batch_addition_identity_cases() {
        let ed_p1 = EdwardsPoint::mul_base(&Scalar::from(2u64));
        let ed_p2 = EdwardsPoint::mul_base(&Scalar::from(3u64));

        let p1 = AffineMontgomeryPoint::from(&ed_p1);
        let p2 = AffineMontgomeryPoint::from(&ed_p2);
        let identity = AffineMontgomeryPoint::identity();

        // Test adding identity to points
        let points = [p1, p2, identity, p1];
        let batch_results = AffineMontgomeryPoint::batch_addition_not_ct(&points, &identity);

        assert_eq!(batch_results[0].u.to_bytes(), p1.u.to_bytes());
        assert_eq!(batch_results[1].u.to_bytes(), p2.u.to_bytes());
        assert_eq!(batch_results[2].u.to_bytes(), identity.u.to_bytes());

        // Test adding to identity points
        let addend = AffineMontgomeryPoint::from(&ed_p1);
        let points_with_identity = [identity, p2, identity, p1];
        let batch_results2 =
            AffineMontgomeryPoint::batch_addition_not_ct(&points_with_identity, &addend);

        assert_eq!(batch_results2[0].u.to_bytes(), addend.u.to_bytes());
        assert_eq!(batch_results2[0].v.to_bytes(), addend.v.to_bytes());
    }

    #[test]
    fn test_from_points() {
        let ed_p1 = EdwardsPoint::mul_base(&Scalar::from(2u64));
        let ed_p2 = EdwardsPoint::mul_base(&Scalar::from(3u64));
        let ed_p3 = EdwardsPoint::mul_base(&Scalar::from(5u64));
        let ed_p4 = EdwardsPoint::mul_base(&Scalar::from(rng().random::<u64>()));

        let points = [&ed_p1, &ed_p2, &ed_p3, &ed_p4];
        let affine_points = AffineMontgomeryPoint::from_points(points);

        for (i, ed_point) in points.iter().enumerate() {
            let expected = AffineMontgomeryPoint::from(*ed_point);
            assert_eq!(
                affine_points[i].u.to_bytes(),
                expected.u.to_bytes(),
                "from_points u mismatch at index {}",
                i
            );
            assert_eq!(
                affine_points[i].v.to_bytes(),
                expected.v.to_bytes(),
                "from_points v mismatch at index {}",
                i
            );
        }
    }
}
