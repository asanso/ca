use core::array;
use rand::prelude::*;
use std::{
    fmt::Debug,
    sync::atomic::{AtomicUsize, Ordering},
};

trait Field: Copy + Clone + PartialEq + Eq + Debug + PartialOrd + Ord + Send + Sync + 'static {
    type Uint;
    const MODULUS: Self::Uint;
    const ZERO: Self;
    const ONE: Self;

    fn random(rng: &mut impl Rng) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn div(self, other: Self) -> Self;
    fn pow(self, exp: usize) -> Self;
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Fp8<const MODULUS: u8>(u8);

fn dist<T: Eq, const N: usize>(a: [T; N], b: [T; N]) -> usize {
    let mut d = 0;
    for i in 0..N {
        if a[i] != b[i] {
            d += 1;
        }
    }
    d
}

impl<const PRIME: u8> Field for Fp8<PRIME> {
    type Uint = u8;
    const MODULUS: Self::Uint = PRIME;
    const ZERO: Self = Fp8(0);
    const ONE: Self = Fp8(1);

    fn random(rng: &mut impl Rng) -> Self {
        Self(rng.random_range(0..Self::MODULUS) as u8)
    }

    fn add(self, other: Self) -> Self {
        Fp8((self.0 + other.0) % Self::MODULUS)
    }

    fn sub(self, other: Self) -> Self {
        Fp8((self.0 + Self::MODULUS - other.0) % Self::MODULUS)
    }

    fn mul(self, other: Self) -> Self {
        Fp8((self.0 * other.0) % Self::MODULUS)
    }

    fn div(self, other: Self) -> Self {
        self.mul(other.pow((Self::MODULUS - 2) as usize))
    }

    fn pow(self, exp: usize) -> Self {
        let mut result = Self::ONE;
        let mut base = self;
        let mut e = exp;

        while e > 0 {
            if e % 2 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            e /= 2;
        }
        result
    }
}

impl<const PRIME: u8> Debug for Fp8<PRIME> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:2}", self.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
struct Code<F: Field, const N: usize, const K: usize> {
    generator_matrix: [[F; K]; N],
}

impl<F: Field, const N: usize, const K: usize> Code<F, N, K> {
    fn new_reed_solomon(eval_points: [F; N]) -> Self {
        Self {
            generator_matrix: array::from_fn(|i| {
                let mut v = F::ONE;
                array::from_fn(|_j| {
                    let res = v;
                    v = v.mul(eval_points[i]);
                    res
                })
            }),
        }
    }

    fn new(generator_matrix: [[F; K]; N]) -> Self {
        Self { generator_matrix }
    }

    fn k(&self) -> usize {
        K
    }

    fn n(&self) -> usize {
        N
    }

    fn rate(&self) -> f64 {
        K as f64 / N as f64
    }

    fn encode(&self, message: [F; K]) -> [F; N] {
        let mut codeword = [F::ZERO; N];
        for i in 0..N {
            for j in 0..K {
                codeword[i] = codeword[i].add(self.generator_matrix[i][j].mul(message[j]));
            }
        }
        codeword
    }
}

impl<F: Field, const N: usize, const K: usize> Debug for Code<F, N, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "({}, {}) ρ={} Linear Code with generator matrix:",
            self.n(),
            self.k(),
            self.rate()
        )?;
        for row in &self.generator_matrix {
            write!(f, "[")?;
            for val in row {
                write!(f, " {val:?}",)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

fn random_trial<F: Field, const N: usize>(codewords: &[[F; N]]) {
    let tries = AtomicUsize::new(0);
    let max_count: Vec<AtomicUsize> = (0..(N + 1)).map(|_| AtomicUsize::new(0)).collect();

    rayon::broadcast(|_ctx| {
        let mut rng = rand::rng();
        let mut count = vec![0usize; N + 1];
        loop {
            // Pick a random codeword
            let received: [F; N] = array::from_fn(|i| F::random(&mut rng));

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
    println!("Generating code:");
    let roots = [1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7].map(Fp8);
    let rs: Code<Fp8<13>, 12, 6> = Code::new_reed_solomon(roots);
    println!("{rs:?}");

    println!("Computing codewords:");
    let mut codewords = Vec::with_capacity(13_usize.pow(6));
    for i in 0_u64..(13_u64.pow(6)) {
        let message = array::from_fn({
            let mut x = i;
            move |_i| {
                let element = Fp8((x % 13) as u8);
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
