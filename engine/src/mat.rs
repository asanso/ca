use {
    crate::Field,
    core::ops,
    num_traits::{ConstZero, One, Zero},
    rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    },
    std::{array, fmt::Display, mem::swap},
};

pub type Col<F, const N: usize> = Mat<F, N, 1>;
pub type Row<F, const M: usize> = Mat<F, 1, M>;

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Mat<F: Field, const N: usize, const M: usize>(pub [[F; M]; N])
where
    StandardUniform: Distribution<F>;

impl<F: Field, const N: usize, const M: usize> ConstZero for Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    const ZERO: Self = Self([[F::ZERO; M]; N]);
}

impl<F: Field, const N: usize, const M: usize> Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    pub fn vandermonde(eval_points: [F; N]) -> Self {
        let mut result = [[F::ZERO; M]; N];
        for i in 0..N {
            let mut v = F::ONE;
            for j in 0..M {
                result[i][j] = v;
                v *= eval_points[i];
            }
        }
        Self(result)
    }

    pub fn transpose(&self) -> Mat<F, M, N> {
        let mut result = [[F::ZERO; N]; M];
        for i in 0..N {
            for j in 0..M {
                result[j][i] = self.0[i][j];
            }
        }
        Mat(result)
    }

    pub fn split_horizontal<const M1: usize, const M2: usize>(
        &self,
    ) -> (Mat<F, N, M1>, Mat<F, N, M2>) {
        assert!(
            M == M1 + M2,
            "Expected M1 + M2 == M, got {M1} + {M2} != {M}"
        );
        let mut left = [[F::ZERO; M1]; N];
        let mut right = [[F::ZERO; M2]; N];
        for i in 0..N {
            let (l, r) = self.0[i].split_at(M1);
            left[i].copy_from_slice(l);
            right[i].copy_from_slice(r);
        }
        (Mat(left), Mat(right))
    }

    pub fn join_horizontal<const M2: usize, const MR: usize>(
        &self,
        right: Mat<F, N, M2>,
    ) -> Mat<F, N, MR> {
        assert!(
            M + M2 == MR,
            "Expected M + M2 == MR, got {M} + {M2} != {MR}"
        );
        let mut result = [[F::ZERO; MR]; N];
        for i in 0..N {
            for j in 0..M {
                result[i][j] = self.0[i][j];
            }
            for j in 0..M2 {
                result[i][M + j] = right.0[i][j];
            }
        }
        Mat(result)
    }

    /// Returns [None] if the matrix is not full rank.
    pub fn normal_form(&self) -> Option<Mat<F, N, M>> {
        let mut result = self.0;
        let min_dim = if N < M { N } else { M };
        for i in 0..min_dim {
            // Find nonzero entry in this column
            let pivot = (i..N).find(|&k| !result[k][i].is_zero())?;
            // Swap rows if necessary
            if pivot > i {
                let (_, rows) = result.split_at_mut(i);
                let (row, rows) = rows.split_first_mut().unwrap();
                let (_, rows) = rows.split_at_mut(pivot - i - 1);
                let (pivot_row, _) = rows.split_first_mut().unwrap();
                swap(row, pivot_row);
            }
            // Make the diagonal contain all ones
            let inv = result[i][i].inv();
            for j in 0..M {
                result[i][j] *= inv;
            }
            // Eliminate all other entries in this column
            for k in 0..N {
                if k != i {
                    let factor = result[k][i];
                    for j in 0..M {
                        result[k][j] -= factor * result[i][j];
                    }
                }
            }
        }
        Some(Mat(result))
    }
}

impl<F: Field, const N: usize, const M: usize> Distribution<Mat<F, N, M>> for StandardUniform
where
    StandardUniform: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mat<F, N, M> {
        Mat(array::from_fn(|_| array::from_fn(|_| rng.random())))
    }
}

impl<F: Field, const N: usize, const M: usize> Zero for Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    fn zero() -> Self {
        Self([[F::ZERO; M]; N])
    }

    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
}

impl<F: Field, const N: usize> One for Mat<F, N, N>
where
    StandardUniform: Distribution<F>,
{
    fn one() -> Self {
        let mut result = [[F::ZERO; N]; N];
        for i in 0..N {
            result[i][i] = F::ONE;
        }
        Self(result)
    }
}

impl<F: Field, const N: usize, const M: usize> ops::Add for Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut result = [[F::ZERO; M]; N];
        for i in 0..N {
            for j in 0..M {
                result[i][j] = self.0[i][j] + rhs.0[i][j];
            }
        }
        Self(result)
    }
}

impl<F: Field, const N: usize, const M: usize> ops::Neg for Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        let mut result = [[F::ZERO; M]; N];
        for i in 0..N {
            for j in 0..M {
                result[i][j] = -self.0[i][j];
            }
        }
        Self(result)
    }
}

impl<F: Field, const N: usize, const M: usize> ops::Mul<[F; M]> for Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    type Output = [F; N];

    fn mul(self, rhs: [F; M]) -> Self::Output {
        let mut result = [F::ZERO; N];
        for i in 0..N {
            for j in 0..M {
                result[i] += self.0[i][j] * rhs[j];
            }
        }
        result
    }
}

impl<F: Field, const N: usize, const M: usize> ops::Mul<Mat<F, N, M>> for [F; N]
where
    StandardUniform: Distribution<F>,
{
    type Output = [F; M];

    fn mul(self, rhs: Mat<F, N, M>) -> Self::Output {
        let mut result = [F::ZERO; M];
        for j in 0..M {
            for i in 0..N {
                result[j] += self[i] * rhs.0[i][j];
            }
        }
        result
    }
}

impl<F: Field, const N: usize, const K: usize, const M: usize> ops::Mul<Mat<F, K, M>>
    for Mat<F, N, K>
where
    StandardUniform: Distribution<F>,
{
    type Output = Mat<F, N, M>;

    fn mul(self, rhs: Mat<F, K, M>) -> Self::Output {
        let mut result = Mat::zero();
        for i in 0..N {
            for j in 0..M {
                for k in 0..K {
                    result.0[i][j] += self.0[i][k] * rhs.0[k][j];
                }
            }
        }
        result
    }
}

impl<F: Field, const N: usize, const M: usize> Display for Mat<F, N, M>
where
    StandardUniform: Distribution<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..N {
            write!(f, "[ ")?;
            for j in 0..M {
                write!(f, "{:2} ", self.0[i][j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
