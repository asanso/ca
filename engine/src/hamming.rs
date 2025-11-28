use {
    crate::Field,
    num_traits::Zero,
    rand::distr::{Distribution, StandardUniform},
};

pub fn weight<T: Zero, const N: usize>(a: [T; N]) -> usize {
    let mut w = 0;
    for i in 0..N {
        if !a[i].is_zero() {
            w += 1;
        }
    }
    w
}

pub fn dist<T: Eq, const N: usize>(a: [T; N], b: [T; N]) -> usize {
    let mut d = 0;
    for i in 0..N {
        if a[i] != b[i] {
            d += 1;
        }
    }
    d
}

/// Iterator over all vectors of length N and Hamming weight at most
/// `max_weight`.
pub struct HammingIter<F: Field, const N: usize>
where
    StandardUniform: Distribution<F>,
{
    pub max_weight: usize,
    pub index:      usize,
    pub weight:     usize,
    pub current:    [F; N],
}

impl<F: Field, const N: usize> HammingIter<F, N>
where
    StandardUniform: Distribution<F>,
{
    pub fn new(max_weight: usize) -> Self {
        Self {
            max_weight,
            index: 0,
            weight: 0,
            current: [F::ZERO; N],
        }
    }
}

impl<F: Field, const N: usize> Iterator for HammingIter<F, N>
where
    StandardUniform: Distribution<F>,
{
    type Item = [F; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == N {
            return None;
        }
        let val = self.current;
        loop {
            if self.weight == self.max_weight && self.current[self.index] == F::ZERO {
                // Skip to next index
                self.index += 1;
                if self.index == N {
                    break;
                }
                continue;
            }
            // Increment current
            if self.current[self.index] == F::ZERO {
                self.weight += 1;
            }
            self.current[self.index] += F::ONE;
            if self.current[self.index] == F::ZERO {
                // Wrapped around
                self.weight -= 1;
                self.index += 1;
                if self.index == N {
                    break;
                }
                continue;
            } else {
                self.index = 0;
                break;
            }
        }
        Some(val)
    }
}
