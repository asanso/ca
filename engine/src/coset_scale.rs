use {
    crate::{Field, mat::Mat},
    num_traits::ToPrimitive,
    rand::distr::{Distribution, StandardUniform},
};

/// A scale is an instrument for weighing something.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CosetScale<F: Field>
where
    StandardUniform: Distribution<F>,
{
    q:            usize,
    n:            usize,
    k:            usize,
    space:        Mat<F>,
    counts:       Vec<usize>,
    accumulators: Vec<Option<Vec<F>>>,
    rows:         Vec<Option<Vec<F>>>,
    inv_last_dim: Vec<Option<F>>,
    alpha_counts: Vec<usize>,
}

impl<F: Field> CosetScale<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn new(space: &Mat<F>) -> Self {
        let q = F::MODULUS.to_usize().unwrap();
        let n = space.cols();
        let k = space.rows();
        let space = space.normal_form().expect("Space matrix must be full rank");
        let rows = (0..k).map(|i| space.row(i).to_vec()).map(Some).collect();
        let inv_last_dim = space
            .row(k - 1)
            .iter()
            .map(|c| if c.is_zero() { None } else { Some(c.inv()) })
            .collect::<Vec<_>>();
        Self {
            q,
            n,
            k,
            space,
            inv_last_dim,
            accumulators: vec![Some(vec![F::ZERO; n]); k],
            rows,
            counts: vec![0; n + 1],
            alpha_counts: vec![0; q],
        }
    }

    pub fn weigh(&mut self, offset: &[F]) -> Vec<usize> {
        assert!(
            offset.len() == self.n,
            "Offset length must match space dimension"
        );
        self.counts.fill(0);
        self.affine_space(offset, 0);
        self.counts.clone()
    }

    /// Runs the scale on the affine space defined by the given index.
    fn affine_space(&mut self, offset: &[F], i: usize) {
        assert_eq!(offset.len(), self.n);
        if i == self.k - 1 {
            return self.affine_line(offset);
        }
        let mut accumulator = self.accumulators[i].take().unwrap();
        let row = self.rows[i].take().unwrap();
        accumulator.copy_from_slice(offset);
        for a in 0..self.q {
            self.affine_space(&accumulator, i + 1);
            if a < self.q - 1 {
                accumulator[i] += F::ONE;
                for (a, c) in accumulator.iter_mut().zip(row.iter()).skip(self.k) {
                    *a += *c;
                }
            }
        }
        self.accumulators[i] = Some(accumulator);
        self.rows[i] = Some(row);
    }

    /// Affine line in the last dimension.
    fn affine_line(&mut self, offset: &[F]) {
        assert_eq!(offset.len(), self.n);
        assert_eq!(self.alpha_counts.len(), self.q);
        assert_eq!(self.inv_last_dim.len(), self.n);
        assert_eq!(self.counts.len(), self.n + 1);
        let mut zeros = 0;
        self.alpha_counts.fill(0);
        for (&r, c_inv) in offset.iter().zip(self.inv_last_dim.iter()) {
            if let Some(c_inv) = c_inv {
                let alpha = r * *c_inv;
                self.alpha_counts[alpha.to_u64() as usize] += 1;
            } else {
                // c_j is zero
                if r.is_zero() {
                    // Coordinate is always zero
                    zeros += 1;
                }
            }
        }
        for &count in self.alpha_counts.iter() {
            let zero_count = zeros + count;
            debug_assert!(zero_count <= self.n);
            let weight = self.n - zero_count;
            self.counts[weight] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::{field::Fu8, hamming::weight, iter::VecIter},
    };

    fn all_vectors<F: Field>(len: usize) -> Vec<Vec<F>>
    where
        StandardUniform: Distribution<F>,
    {
        if len == 0 {
            vec![Vec::new()]
        } else {
            let mut result = Vec::new();
            for vec in VecIter::<F>::new(len) {
                result.push(vec);
            }
            result
        }
    }

    fn space_with_tail<F: Field>(k: usize, n: usize, tail: &[F]) -> Mat<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut space = Mat::zero(k, n);
        for i in 0..k {
            space[(i, i)] = F::ONE;
        }
        if n > k {
            let width = n - k;
            for (idx, value) in tail.iter().enumerate() {
                let row = idx / width;
                let col = k + (idx % width);
                space[(row, col)] = *value;
            }
        }
        space
    }

    fn brute_force_weights<F: Field>(offset: &[F], space: &Mat<F>) -> Vec<usize>
    where
        StandardUniform: Distribution<F>,
    {
        let n = space.cols();
        let k = space.rows();
        assert!(offset.len() == n);
        assert!(offset[..k].iter().all(|x| x.is_zero()));

        let mut histogram = vec![0usize; n + 1];
        let mut counter = vec![F::ZERO; k];
        let mut counter_weight = 0;
        let mut active = offset[k..].to_vec();
        'outer: loop {
            let w = counter_weight + weight(&active);
            histogram[w] += 1;

            let mut index = k - 1;
            loop {
                active
                    .iter_mut()
                    .zip(space.row(index)[k..].iter())
                    .for_each(|(c, s)| {
                        *c += *s;
                    });
                if counter[index].is_zero() {
                    counter_weight += 1;
                }
                counter[index] += F::ONE;
                if counter[index].is_zero() {
                    counter_weight -= 1;
                    if index == 0 {
                        break 'outer;
                    }
                    index -= 1;
                } else {
                    break;
                }
            }
        }

        histogram
    }

    fn check_field<F: Field>()
    where
        StandardUniform: Distribution<F>,
    {
        // Exhaustively enumerate small matrices that are already in normal form
        // (identity left block with arbitrary tail) and verify that the coset
        // scale matches brute force weight enumeration for every offset.
        const MAX_N: usize = 3;
        for n in 1..=MAX_N {
            for k in 1..=n {
                let tail_len = k * (n - k);
                let tails = all_vectors::<F>(tail_len);
                for tail in tails {
                    let space = space_with_tail::<F>(k, n, &tail);
                    let space = space.normal_form().expect("matrix should be full rank");
                    let offsets = all_vectors::<F>(n - k);
                    for offset_tail in offsets {
                        let mut offset = vec![F::ZERO; n];
                        offset[k..].copy_from_slice(&offset_tail);
                        let expected = brute_force_weights::<F>(&offset, &space);
                        let mut scale = CosetScale::new(&space);
                        let actual = scale.weigh(&offset);
                        assert_eq!(
                            expected, actual,
                            "Mismatch for n={n}, k={k}, tail={tail:?}, offset={offset_tail:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn coset_scale_matches_enumeration_f2() {
        check_field::<Fu8<2>>();
    }

    #[test]
    fn coset_scale_matches_enumeration_f3() {
        check_field::<Fu8<3>>();
    }
}
