mod field;
mod mat;

use {
    crate::{
        field::{Field, Fu8},
        mat::Mat,
    },
    core::{
        array,
        fmt::Debug,
        sync::atomic::{AtomicUsize, Ordering},
    },
    num_traits::{ConstZero, Zero},
    rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    },
};

pub type F13 = Fu8<13>;

fn weight<T: Zero, const N: usize>(a: [T; N]) -> usize {
    let mut w = 0;
    for i in 0..N {
        if !a[i].is_zero() {
            w += 1;
        }
    }
    w
}

fn dist<T: Eq, const N: usize>(a: [T; N], b: [T; N]) -> usize {
    let mut d = 0;
    for i in 0..N {
        if a[i] != b[i] {
            d += 1;
        }
    }
    d
}

pub struct ErrorIter<F: Field, const N: usize>
where
    StandardUniform: Distribution<F>,
{
    pub max_weight: usize,
    pub index:      usize,
    pub weight:     usize,
    pub current:    [F; N],
}

impl<F: Field, const N: usize> ErrorIter<F, N>
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

impl<F: Field, const N: usize> Iterator for ErrorIter<F, N>
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

#[derive(Copy, Clone, PartialEq, Eq)]
struct Code<F: Field, const N: usize, const K: usize>
where
    StandardUniform: Distribution<F>,
{
    generator_matrix: Mat<F, N, K>,
}

impl<F: Field, const N: usize, const K: usize> Code<F, N, K>
where
    StandardUniform: Distribution<F>,
{
    const REDUNDANCY: usize = N - K;

    fn new_reed_solomon(eval_points: [F; N]) -> Self {
        Self {
            generator_matrix: Mat::vandermonde(eval_points),
        }
    }

    fn new(generator_matrix: Mat<F, N, K>) -> Self {
        Self { generator_matrix }
    }

    fn rate(&self) -> f64 {
        K as f64 / N as f64
    }

    fn encode(&self, message: [F; K]) -> [F; N] {
        self.generator_matrix * message
    }
}

impl<F: Field, const N: usize, const K: usize> Debug for Code<F, N, K>
where
    StandardUniform: Distribution<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "({N}, {K}) ρ={} Linear Code with generator matrix:",
            self.rate()
        )?;
        Ok(())
    }
}

fn random_trial<F: Field, const N: usize>(codewords: &[[F; N]])
where
    StandardUniform: Distribution<F>,
{
    let tries = AtomicUsize::new(0);
    let max_count: Vec<AtomicUsize> = (0..(N + 1)).map(|_| AtomicUsize::new(0)).collect();

    rayon::broadcast(|_ctx| {
        let mut rng = rand::rng();
        let mut count = vec![0usize; N + 1];
        loop {
            // Pick a random codeword
            let received: [F; N] = array::from_fn(|i| rng.random());

            // List decode (all codewords within distance 3)
            count.fill(0);
            for candidate in codewords {
                let d = dist(*candidate, received);
                count[d] += 1;
            }

            // Cumulative sum to get all codewords within distance <= d
            for i in 1..count.len() {
                count[i] += count[i - 1];
            }

            // Update maxima
            for (i, (max, c)) in max_count.iter().zip(count.iter()).enumerate() {
                let prev = max.fetch_max(*c, Ordering::Relaxed);
                if *c > prev {
                    println!(
                        "New maximum for distance {i}: {} codewords. Received: {:?}",
                        *c, received
                    );
                }
            }

            // Progress reporting
            let i = tries.fetch_add(1, Ordering::Relaxed);
            if i.is_multiple_of(1000) {
                print!("Distogram maxima so far: ");
                for (i, count) in max_count.iter().enumerate() {
                    print!("{}", count.load(Ordering::Relaxed));
                    if i < max_count.len() - 1 {
                        print!(", ");
                    }
                }
                println!(" after {i} tries done...");
            }
        }
    });
}

fn main() {
    println!("");
    for error in ErrorIter::<F13, 5>::new(2) {
        println!("{:?} Weight: {}", error, weight(error));
    }

    return;

    println!("Generating code:");
    let roots = [1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7].map(F13::from);
    let rs: Code<F13, 12, 6> = Code::new_reed_solomon(roots);
    println!("{rs:?}");

    println!("Computing codewords:");
    let mut codewords = Vec::with_capacity(13_usize.pow(6));
    for i in 0_u64..(13_u64.pow(6)) {
        let message = array::from_fn({
            let mut x = i;
            move |_i| {
                let element = F13::from((x % 13) as u8);
                x /= 13;
                element
            }
        });

        let codeword = rs.encode(message);
        codewords.push(codeword);
    }

    codewords.sort();
    let prev = codewords.len();
    codewords.dedup();
    assert_eq!(prev, codewords.len(), "Duplicate codewords!");
    println!("{} Codewords generated.", codewords.len());

    random_trial(&codewords);
}
