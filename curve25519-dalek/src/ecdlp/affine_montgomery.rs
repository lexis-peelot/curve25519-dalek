use crate::{EdwardsPoint, constants::MONTGOMERY_A, field::FieldElement};

#[derive(Clone, Copy, Debug)]
pub(crate) struct AffineMontgomeryPoint {
    pub u: FieldElement,
    pub v: FieldElement,
}

impl AffineMontgomeryPoint {
    pub fn is_identity_not_ct(&self) -> bool {
        self.u == FieldElement::ZERO && self.v == FieldElement::ZERO
    }

    pub fn identity() -> Self {
        AffineMontgomeryPoint {
            u: FieldElement::ZERO,
            v: FieldElement::ZERO,
        }
    }

    pub fn from_bytes(u: &[u8; 32], v: &[u8; 32]) -> Self {
        Self {
            u: FieldElement::from_bytes(u),
            v: FieldElement::from_bytes(v),
        }
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

    /// Add the same point to 4 different points simultaneously
    pub fn batch_addition_not_ct_4way(
        points: &[Self; 4],
        addend: &Self,
    ) -> [Self; 4] {
        // Early exit checks for identity
        if addend.is_identity_not_ct() {
            return *points;
        }
        
        // Check if any input points are identity
        let mut results = [Self::identity(); 4];
        let mut mask = [false; 4];
        for i in 0..4 {
            if points[i].is_identity_not_ct() {
                results[i] = *addend;
                mask[i] = true;
            }
        }
        
        // Extract u and v coordinates for batch operations
        let u_coords = [points[0].u, points[1].u, points[2].u, points[3].u];
        let v_coords = [points[0].v, points[1].v, points[2].v, points[3].v];
        
        // Check for inverse points (u1 == u2 && v1 == -v2)
        let u_diffs = FieldElement::batch_subtract_4way(&u_coords, &addend.u);
        let v_sums = FieldElement::batch_add_4way(&v_coords, &addend.v);
        
        // Compute denominators for lambda
        let mut denominators = [FieldElement::ZERO; 4];
        let mut is_doubling = [false; 4];
        
        for i in 0..4 {
            if mask[i] {
                continue;
            }
            
            if u_diffs[i] == FieldElement::ZERO {
                if v_sums[i] == FieldElement::ZERO {
                    // Point at infinity case
                    results[i] = Self::identity();
                    mask[i] = true;
                } else {
                    // Doubling case
                    is_doubling[i] = true;
                    denominators[i] = &points[i].v + &points[i].v;
                }
            } else {
                // Regular addition case
                denominators[i] = u_diffs[i];
            }
        }
        
        // Batch invert denominators
        let mut inv_denominators = denominators;
        FieldElement::batch_invert_4(&mut inv_denominators);
        
        // Compute numerators based on doubling vs addition
        let mut numerators = [FieldElement::ZERO; 4];
        for i in 0..4 {
            if mask[i] {
                continue;
            }
            
            if is_doubling[i] {
                // (3*u1^2 + 2*A*u1 + 1)
                let u_sq = points[i].u.square();
                let u_sq_3 = &(&u_sq + &u_sq) + &u_sq;
                let u_ta = &MONTGOMERY_A * &points[i].u;
                let u_ta_2 = &u_ta + &u_ta;
                numerators[i] = &(&u_sq_3 + &u_ta_2) + &FieldElement::ONE;
            } else {
                // (v1 - v2)
                numerators[i] = &points[i].v - &addend.v;
            }
        }
        
        // Compute lambdas using batch multiplication
        let lambdas = FieldElement::batch_mul_4way(&numerators, &inv_denominators);
        
        // Square lambdas
        let lambda_squared = FieldElement::batch_square_4way(&lambdas);
        
        // Compute new u coordinates: lambda^2 - A - u1 - u2
        let mut new_u_values = lambda_squared;
        for i in 0..4 {
            if !mask[i] {
                new_u_values[i] = &(&new_u_values[i] - &MONTGOMERY_A) - &(&points[i].u + &addend.u);
            }
        }
        
        // Compute u1 - u3 for each point
        let u_diffs_for_v = [
            &points[0].u - &new_u_values[0],
            &points[1].u - &new_u_values[1],
            &points[2].u - &new_u_values[2],
            &points[3].u - &new_u_values[3],
        ];
        
        // Compute new v coordinates: lambda * (u1 - u3) - v1
        let lambda_times_diff = FieldElement::batch_mul_4way(&lambdas, &u_diffs_for_v);
        let new_v_values = [
            &lambda_times_diff[0] - &points[0].v,
            &lambda_times_diff[1] - &points[1].v,
            &lambda_times_diff[2] - &points[2].v,
            &lambda_times_diff[3] - &points[3].v,
        ];
        
        // Assemble results
        for i in 0..4 {
            if !mask[i] {
                results[i] = Self {
                    u: new_u_values[i],
                    v: new_v_values[i],
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
        AffineMontgomeryPoint {
            u: &Z_plus_Y * &Z_minus_Y.invert(),
            v: &(&Z_plus_Y * &X_minus_T.invert()) * &ALPHA,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
