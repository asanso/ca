use {
    crate::{Field, HammingIter, mat::Mat, weight},
    num_traits::{One, ToPrimitive},
    rand::distr::{Distribution, StandardUniform},
    std::fmt::{Debug, Display},
};

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Code<F: Field, const N: usize, const K: usize, const R: usize>
where
    StandardUniform: Distribution<F>,
{
    generator: Mat<F, K, N>,
    parity:    Mat<F, R, N>,
}

impl<F: Field, const N: usize, const K: usize, const R: usize> Code<F, N, K, R>
where
    StandardUniform: Distribution<F>,
{
    /// Creates a new linear code from the given generator matrix.
    ///
    /// The generator and parity matrices are put into standard form.
    ///
    /// Returns [None] if the generator matrix is not full rank.
    pub fn new(generator: Mat<F, K, N>) -> Option<Self> {
        let generator = generator.normal_form()?;
        let (_, p) = generator.split_horizontal::<K, R>();
        let parity = (-p).transpose().join_horizontal::<R, N>(Mat::one());
        let parity = parity.normal_form()?;
        Some(Self { generator, parity })
    }

    /// Creates a new Reed-Solomon code with the given evaluation points.
    ///
    /// Returns [None] if the `evaluation_points` are not distinct.
    pub fn new_reed_solomon(eval_points: [F; N]) -> Option<Self> {
        Self::new(Mat::vandermonde(eval_points).transpose())
    }

    /// Creates a new cyclic code with the given generator polynomial.
    pub fn new_cyclic<const R1: usize>(generator: [F; R1]) -> Option<Self> {
        assert_eq!(R1, R + 1, "generator must have length R + 1");
        let mut gen_matrix = [[F::ZERO; N]; K];
        for i in 0..K {
            gen_matrix[i][i..(i + R1)].copy_from_slice(&generator);
        }
        Self::new(Mat(gen_matrix))
    }

    pub fn dual(&self) -> Code<F, N, R, K> {
        Code {
            generator: self.parity,
            parity:    self.generator,
        }
    }

    pub fn generator(&self) -> Mat<F, K, N> {
        self.generator
    }

    pub fn parity(&self) -> Mat<F, R, N> {
        self.parity
    }

    pub fn size(&self) -> usize {
        F::MODULUS.to_usize().unwrap().pow(K as u32)
    }

    pub fn codewords(&self) -> impl Iterator<Item = [F; N]> + '_ {
        HammingIter::new(K).map(move |message| self.encode(message))
    }

    pub fn rate(&self) -> f64 {
        K as f64 / N as f64
    }

    /// Count of codewords of each weight.
    pub fn weights(&self) -> Vec<usize> {
        let mut weights = vec![0usize; N + 1];
        for codeword in self.codewords() {
            let w = weight(codeword);
            weights[w] += 1;
        }
        weights
    }

    pub fn encode(&self, message: [F; K]) -> [F; N] {
        message * self.generator
    }

    pub fn syndrome(&self, word: [F; N]) -> [F; R] {
        self.parity * word
    }

    pub fn decode(&self, word: [F; N], max_dist: usize) -> impl Iterator<Item = [F; K]> {
        let syndrome = self.syndrome(word);
        HammingIter::new(max_dist).filter_map(move |error| {
            let test_syndrome = self.syndrome(error);
            if test_syndrome == syndrome {
                // Extract message (using the fact that generator is in standard form)
                let mut message = [F::ZERO; K];
                for i in 0..K {
                    message[i] = word[i] - error[i];
                }
                Some(message)
            } else {
                None
            }
        })
    }
}

impl<F: Field, const N: usize, const K: usize, const R: usize> Display for Code<F, N, K, R>
where
    StandardUniform: Distribution<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({N}, {K}) Linear Code")
    }
}
