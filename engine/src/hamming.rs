use {
    crate::Field,
    num_traits::Zero,
    rand::distr::{Distribution, StandardUniform},
};

pub fn weight<T: Zero>(a: &[T]) -> usize {
    a.iter().filter(|x| !x.is_zero()).count()
}

pub fn dist<T: Eq>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
}

/// Iterator over all vectors of length N and Hamming weight at most
/// `max_weight`.
pub struct HammingIter<F: Field>
where
    StandardUniform: Distribution<F>,
{
    pub max_weight: usize,
    pub index:      usize,
    pub weight:     usize,
    pub current:    Vec<F>,
}

impl<F: Field> HammingIter<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn new(size: usize, max_weight: usize) -> Self {
        Self {
            max_weight,
            index: 0,
            weight: 0,
            current: vec![F::ZERO; size],
        }
    }
}

impl<F: Field> Iterator for HammingIter<F>
where
    StandardUniform: Distribution<F>,
{
    type Item = Vec<F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.current.len() {
            return None;
        }
        let value = self.current.clone();
        loop {
            if self.weight == self.max_weight && self.current[self.index] == F::ZERO {
                // Skip to next index
                self.index += 1;
                if self.index == self.current.len() {
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
                if self.index == self.current.len() {
                    break;
                }
                continue;
            } else {
                self.index = 0;
                break;
            }
        }
        Some(value)
    }
}
