use {
    crate::Field,
    core::ops,
    rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    },
    std::{
        fmt::Display,
        ops::{Index, IndexMut},
    },
};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Mat<F: Field>
where
    StandardUniform: Distribution<F>,
{
    rows:   usize,
    cols:   usize,
    values: Vec<F>,
}

impl<F: Field> Mat<F>
where
    StandardUniform: Distribution<F>,
{
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            values: vec![F::ZERO; rows * cols],
        }
    }

    pub fn one(size: usize) -> Self {
        let mut result = Self::zero(size, size);
        for i in 0..size {
            result[(i, i)] = F::ONE;
        }
        result
    }

    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();
        let mut values = Vec::with_capacity(rows * cols);
        for _ in 0..(rows * cols) {
            values.push(rng.random());
        }
        Self { rows, cols, values }
    }

    pub fn vandermonde(eval_points: &[F], cols: usize, extended: bool) -> Self {
        let rows = if extended {
            eval_points.len() + 1
        } else {
            eval_points.len()
        };
        let mut result = Self::zero(rows, cols);
        for i in 0..eval_points.len() {
            let mut v = F::ONE;
            for j in 0..result.cols {
                result[(i, j)] = v;
                v *= eval_points[i];
            }
        }
        if extended && cols > 0 {
            result[(rows - 1, cols - 1)] = F::ONE;
        }
        result
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, i: usize) -> &[F] {
        assert!(
            i < self.rows,
            "Row index out of bounds: {i} >= {}",
            self.rows
        );
        &self.values[i * self.cols..(i + 1) * self.cols]
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::zero(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }

    pub fn split_horizontal(&self, col: usize) -> (Self, Self) {
        assert!(col <= self.cols, "Column index out of bounds");
        let mut left = Self::zero(self.rows, col);
        let mut right = Self::zero(self.rows, self.cols - col);
        for i in 0..self.rows {
            for j in 0..col {
                left[(i, j)] = self[(i, j)];
            }
            for j in col..self.cols {
                right[(i, j - col)] = self[(i, j)];
            }
        }
        (left, right)
    }

    pub fn join_horizontal(&self, right: Self) -> Self {
        assert!(
            self.rows == right.rows,
            "Row count mismatch: {} != {}",
            self.rows,
            right.rows
        );
        let mut result = Self::zero(self.rows, self.cols + right.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(i, j)] = self[(i, j)];
            }
            for j in 0..right.cols {
                result[(i, j + self.cols)] = right[(i, j)];
            }
        }
        result
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        assert!(
            i < self.rows,
            "Row index out of bounds: {i} >= {}",
            self.rows
        );
        assert!(
            j < self.rows,
            "Row index out of bounds: {j} >= {}",
            self.rows
        );
        for k in 0..self.cols {
            self.values.swap(i * self.cols + k, j * self.cols + k);
        }
    }

    /// Returns [None] if the matrix is not full rank.
    pub fn normal_form(&self) -> Option<Self> {
        let mut result = self.clone();
        let min_dim = if self.rows < self.cols {
            self.rows
        } else {
            self.cols
        };
        for i in 0..min_dim {
            // Find nonzero entry in this column
            let pivot = (i..self.rows).find(|&k| !result[(k, i)].is_zero())?;
            // Swap rows if necessary
            if pivot > i {
                result.swap_rows(i, pivot);
            }
            // Make the diagonal contain all ones
            let inv = result[(i, i)].inv();
            for j in 0..self.cols {
                result[(i, j)] *= inv;
            }
            // Eliminate all other entries in this column
            for k in 0..self.rows {
                if k != i {
                    let factor = result[(k, i)];
                    for j in 0..self.cols {
                        let scaled = factor * result[(i, j)];
                        result[(k, j)] -= scaled;
                    }
                }
            }
        }
        Some(result)
    }

    pub fn is_normal_form(&self) -> bool {
        let min_dim = if self.rows < self.cols {
            self.rows
        } else {
            self.cols
        };
        for i in 0..min_dim {
            for j in 0..i {
                if i == j {
                    if !self[(i, j)].is_one() {
                        return false;
                    }
                } else if !self[(i, j)].is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

impl<F: Field> Index<(usize, usize)> for Mat<F>
where
    StandardUniform: Distribution<F>,
{
    type Output = F;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        assert!(
            i < self.rows,
            "Row index out of bounds: {i} >= {}",
            self.rows
        );
        assert!(
            j < self.cols,
            "Column index out of bounds: {j} >= {}",
            self.cols
        );
        &self.values[i * self.cols + j]
    }
}

impl<F: Field> IndexMut<(usize, usize)> for Mat<F>
where
    StandardUniform: Distribution<F>,
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        assert!(
            i < self.rows,
            "Row index out of bounds: {i} >= {}",
            self.rows
        );
        assert!(
            j < self.cols,
            "Column index out of bounds: {j} >= {}",
            self.cols
        );
        &mut self.values[i * self.cols + j]
    }
}

impl<F: Field> ops::Add for Mat<F>
where
    StandardUniform: Distribution<F>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert!(
            self.rows == rhs.rows && self.cols == rhs.cols,
            "Matrix size mismatch: {}x{} != {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );
        Self {
            rows:   self.rows,
            cols:   self.cols,
            values: self
                .values
                .iter()
                .zip(rhs.values.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }
}

impl<F: Field> ops::Neg for Mat<F>
where
    StandardUniform: Distribution<F>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            rows:   self.rows,
            cols:   self.cols,
            values: self.values.iter().map(|a| -*a).collect(),
        }
    }
}

impl<F: Field> ops::Mul<&Mat<F>> for &Mat<F>
where
    StandardUniform: Distribution<F>,
{
    type Output = Mat<F>;

    fn mul(self, rhs: &Mat<F>) -> Self::Output {
        assert!(
            self.cols == rhs.rows,
            "Matrix size mismatch: {}x{} * {}x{}",
            self.rows,
            self.cols,
            rhs.rows,
            rhs.cols
        );
        let mut result = Mat::zero(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                for k in 0..self.cols {
                    result[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        result
    }
}

impl<F: Field> ops::Mul<&Mat<F>> for &[F]
where
    StandardUniform: Distribution<F>,
{
    type Output = Vec<F>;

    fn mul(self, rhs: &Mat<F>) -> Self::Output {
        assert!(
            self.len() == rhs.rows,
            "Vector and matrix size mismatch: {} * {}x{}",
            self.len(),
            rhs.rows,
            rhs.cols
        );
        let mut result = vec![F::ZERO; rhs.cols];
        for j in 0..rhs.cols {
            for i in 0..rhs.rows {
                result[j] += self[i] * rhs[(i, j)];
            }
        }
        result
    }
}

impl<F: Field> ops::Mul<&[F]> for &Mat<F>
where
    StandardUniform: Distribution<F>,
{
    type Output = Vec<F>;

    fn mul(self, rhs: &[F]) -> Self::Output {
        assert!(
            self.cols == rhs.len(),
            "Matrix and vector size mismatch: {}x{} * {}",
            self.rows,
            self.cols,
            rhs.len()
        );
        let mut result = vec![F::ZERO; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self[(i, j)] * rhs[j];
            }
        }
        result
    }
}

impl<F: Field> Display for Mat<F>
where
    StandardUniform: Distribution<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            write!(f, "[ ")?;
            for j in 0..self.cols {
                write!(f, "{:2} ", self[(i, j)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}
