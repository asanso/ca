use {
    crate::{Field, HammingIter, mat::Mat, weight},
    num_traits::{One, ToPrimitive},
    rand::distr::{Distribution, StandardUniform},
    std::fmt::{Debug, Display},
};

#[derive(Clone, PartialEq, Eq)]
pub struct Code<F: Field>
where
    StandardUniform: Distribution<F>,
{
    generator: Mat<F>,
    parity:    Mat<F>,
}

impl<F: Field> Code<F>
where
    StandardUniform: Distribution<F>,
{
    /// Creates a new linear code from the given generator matrix.
    ///
    /// The generator and parity matrices are put into standard form.
    ///
    /// Returns [None] if the generator matrix is not full rank.
    pub fn new(generator: Mat<F>) -> Option<Self> {
        let n = generator.cols();
        let k = generator.rows();
        let r = n - k;
        let generator = generator.normal_form()?;
        let (_, p) = generator.split_horizontal(k);
        let parity = (-p).transpose().join_horizontal(Mat::one(r));
        let parity = parity.normal_form()?;
        Some(Self { generator, parity })
    }

    /// Creates a new Reed-Solomon code with the given evaluation points.
    ///
    /// Returns [None] if the `evaluation_points` are not distinct.
    pub fn new_reed_solomon(k: usize, eval_points: &[F]) -> Option<Self> {
        Self::new(Mat::vandermonde(eval_points, k).transpose())
    }

    /// Creates a new cyclic code with the given generator polynomial.
    pub fn new_cyclic(k: usize, generator: &[F]) -> Option<Self> {
        let r = generator.len() - 1;
        let n = k + r;
        let mut gen_matrix = Mat::zero(k, n);
        for i in 0..k {
            for j in 0..=r {
                gen_matrix[(i, i + j)] = generator[j];
            }
        }
        Self::new(gen_matrix)
    }

    pub fn dual(&self) -> Code<F> {
        Code {
            generator: self.parity.clone(),
            parity:    self.generator.clone(),
        }
    }

    pub fn generator(&self) -> &Mat<F> {
        &self.generator
    }

    pub fn parity(&self) -> &Mat<F> {
        &self.parity
    }

    /// The alphabet size.
    pub fn q(&self) -> usize {
        F::MODULUS.to_usize().unwrap()
    }

    /// The number of codeword symbols.
    pub fn n(&self) -> usize {
        self.generator.cols()
    }

    /// The number of message symbols.
    pub fn k(&self) -> usize {
        self.generator.rows()
    }

    /// The number of parity symbols.
    pub fn redundancy(&self) -> usize {
        self.parity.rows()
    }

    pub fn rate(&self) -> f64 {
        self.k() as f64 / self.n() as f64
    }

    /// The number of codewords.
    pub fn size(&self) -> usize {
        F::MODULUS.to_usize().unwrap().pow(self.k() as u32)
    }

    pub fn codewords(&self) -> impl Iterator<Item = Vec<F>> + '_ {
        HammingIter::new(self.k(), self.k()).map(move |message| self.encode(&message))
    }

    /// Count of codewords of each weight.
    pub fn weights(&self) -> Vec<usize> {
        let mut weights = vec![0_usize; self.n() + 1];
        for codeword in self.codewords() {
            let w = weight(&codeword);
            weights[w] += 1;
        }
        weights
    }

    pub fn encode(&self, message: &[F]) -> Vec<F> {
        message * &self.generator
    }

    pub fn syndrome(&self, word: &[F]) -> Vec<F> {
        &self.parity * word
    }

    pub fn decode(&self, word: &[F], max_dist: usize) -> impl Iterator<Item = Vec<F>> {
        let syndrome = self.syndrome(word);
        HammingIter::new(self.n(), max_dist).filter_map(move |error| {
            let test_syndrome = self.syndrome(&error);
            if test_syndrome == syndrome {
                // Extract message (using the fact that generator is in standard form)
                let mut message = vec![F::ZERO; self.k()];
                for i in 0..self.k() {
                    message[i] = word[i] - error[i];
                }
                Some(message)
            } else {
                None
            }
        })
    }
}

impl<F: Field> Display for Code<F>
where
    StandardUniform: Distribution<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}) Linear Code", self.n(), self.k())
    }
}
