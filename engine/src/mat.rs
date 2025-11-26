use {
    crate::Field,
    core::ops,
    num_traits::{ConstZero, Zero},
    rand::distr::{Distribution, StandardUniform},
};

#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Mat<F: Field, const N: usize, const M: usize>([[F; M]; N])
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
