use {
    crate::field::Field,
    num_traits::{FromPrimitive, ToPrimitive},
    rand::distr::{Distribution, StandardUniform},
};

/// Iterator over all vectors of length N.
pub struct VecIter<F: Field>
where
    StandardUniform: Distribution<F>,
{
    pub current: Vec<F>,
}

/// Iterator over all vectors of length N modulo a scaling factor.
pub struct ProjectiveIter<F: Field>
where
    StandardUniform: Distribution<F>,
{
    pub current: Vec<F>,
}

/// Iterator over all size N subsets of a field F.
pub struct SetIter<F: Field>
where
    StandardUniform: Distribution<F>,
{
    pub current: Vec<F>,
}

impl<F: Field> VecIter<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn new(size: usize) -> Self {
        Self {
            current: vec![F::ZERO; size],
        }
    }
}

impl<F: Field> Iterator for VecIter<F>
where
    StandardUniform: Distribution<F>,
{
    type Item = Vec<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_empty() {
            return None;
        }
        let value = self.current.clone();
        // Increment current
        let mut i = self.current.len() - 1;
        loop {
            self.current[i] += F::ONE;
            if self.current[i] == F::ZERO {
                // Wrapped around
                if i == 0 {
                    // Finished all vectors
                    self.current.clear();
                    break;
                }
                i -= 1;
            } else {
                break;
            }
        }
        Some(value)
    }
}

impl<F: Field> ProjectiveIter<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn new(size: usize) -> Self {
        Self {
            current: vec![F::ZERO; size],
        }
    }
}

impl<F: Field> Iterator for ProjectiveIter<F>
where
    StandardUniform: Distribution<F>,
{
    type Item = Vec<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_empty() {
            return None;
        }
        let value = self.current.clone();
        // Increment current, ensuring the first non-zero element (if any) is
        // always one.
        let mut i = self.current.len() - 1;
        let first_nonzero = self.current.iter().position(|x| *x != F::ZERO);
        loop {
            if Some(i) == first_nonzero {
                debug_assert!(self.current[i] == F::ONE);
                self.current[i] = F::ZERO;
                if i == 0 {
                    // Finished all vectors
                    self.current.clear();
                    break;
                }
                i -= 1;
            } else {
                self.current[i] += F::ONE;
                if self.current[i] == F::ZERO {
                    // Wrapped around
                    if i == 0 {
                        // Finished all vectors
                        self.current.clear();
                        break;
                    }
                    i -= 1;
                } else {
                    break;
                }
            }
        }
        Some(value)
    }
}

impl<F: Field> SetIter<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn new(size: usize) -> Self {
        assert!(size <= F::MODULUS.to_usize().unwrap());
        Self {
            current: (0..size)
                .map(|i| F::from(F::UInt::from_usize(i).unwrap()))
                .collect(),
        }
    }
}

impl<F: Field> Iterator for SetIter<F>
where
    StandardUniform: Distribution<F>,
{
    type Item = Vec<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_empty() {
            return None;
        }
        let value = self.current.clone();
        let n = self.current.len();
        let p = F::MODULUS.to_usize().unwrap();
        // Increment current
        let mut i = n - 1;
        loop {
            let cur = self.current[i].to_u64() as usize;
            let max = p - (n - i);
            if cur < max {
                let next_val = cur + 1;
                self.current[i] = F::from_u64(next_val as u64);
                for j in (i + 1)..n {
                    let prev = self.current[j - 1].to_u64() as usize + 1;
                    self.current[j] = F::from_u64(prev as u64);
                }
                break;
            }
            if i == 0 {
                self.current.clear();
                break;
            }
            i -= 1;
        }
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{const_for, field::Fu8, util::binomial_coefficient},
        std::cmp::min,
    };

    #[test]
    fn test_vec_iter() {
        const_for!(P in [2,3,5,7,13] {
            type Fp = Fu8<{P as u8}>;
            println!("Testing F_{P}");
            for n in 1..=4 {
                assert_eq!(VecIter::<Fp>::new(n).count(), P.pow(n as u32), "All vectors of length {n} over F_{P}");
            }
        });
    }

    #[test]
    fn test_proj_iter() {
        const_for!(P in [2,3,5,7,13] {
            type Fp = Fu8<{P as u8}>;
            println!("Testing F_{P}");
            for n in 1..=4 {
                assert_eq!(ProjectiveIter::<Fp>::new(n).count(), 1 + (P.pow(n as u32) - 1) / (P - 1), "All vectors of length {n} over F_{P} with first non-zero coordinate one");
            }
        });
    }

    #[test]
    fn test_set_iter() {
        const_for!(P in [2,3,5,7,13] {
            type Fp = Fu8<{P as u8}>;
            println!("Testing F_{P}");
            for n in 1..min(5, P) {
                let expected = binomial_coefficient(P, n);
                assert_eq!(SetIter::<Fp>::new(n).count(), expected, "All size {n} subsets of F_{P}");
            }
        });
    }
}
