mod code;
mod field;
mod hamming;
mod mat;

use {
    crate::{
        code::Code,
        field::{Field, Fu8},
        hamming::{HammingIter, dist, weight},
        mat::Mat,
    },
    core::{
        array,
        sync::atomic::{AtomicUsize, Ordering},
    },
    num_traits::{ConstZero, FromPrimitive, ToPrimitive},
    rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    },
    std::{collections::HashMap, usize},
};

#[macro_export]
macro_rules! const_for {
    ($C:ident in [$($n:expr),*] $x:block) => {
        $({
            const $C: usize = $n;
            $x
        })*
    };
    ($C:ident in SIZES $x:block) => {
        $crate::const_for!($C in [0] $x);
        $crate::const_for!($C in NON_ZERO $x);
    };
    ($C:ident in NON_ZERO $x:block) => {
        $crate::const_for!($C in [1, 2, 63, 64, 65, 127, 128, 129, 256, 384, 512, 4096] $x);
    };
    ($C:ident in BENCH $x:block) => {
        $crate::const_for!($C in [64, 128, 192, 256, 384, 512, 4096] $x);
    };
    ($C:ident in $S:tt if $($t:tt)*) => {
        $crate::const_for!($C in $S {
            if $($t)*
        });
    };
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
            // Pick a random word
            let received: [F; N] = array::from_fn(|_| rng.random());

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

/// Distrogram: number of codewords at distance d for d=0..=N
pub fn distogram<F: Field, const N: usize, const K: usize, const R: usize>(
    code: &Code<F, N, K, R>,
    word: [F; N],
) -> Vec<usize>
where
    StandardUniform: Distribution<F>,
{
    let mut distogram = vec![0usize; N + 1];
    for codeword in code.codewords() {
        let d = dist(codeword, word);
        distogram[d] += 1;
    }
    distogram
}

pub fn max_distances<F: Field, const N: usize, const K: usize, const R: usize>(
    code: &Code<F, N, K, R>,
) -> Vec<usize>
where
    StandardUniform: Distribution<F>,
{
    let mut max_count = vec![0usize; N + 1];
    for word in HammingIter::<F, N>::new(N) {
        let mut count = vec![0usize; N + 1];
        for codeword in code.codewords() {
            let d = dist(codeword, word);
            count[d] += 1;
        }
        // Cumulative sum to get all codewords within distance <= d
        for i in 1..count.len() {
            count[i] += count[i - 1];
        }
        for (i, (max, count)) in max_count.iter_mut().zip(count.iter()).enumerate() {
            if *count > *max {
                *max = *count;
                // println!(
                //     "New maximum for distance {i}: {} codewords. Word:
                // {:?}",     *count, word
                // );
            }
        }
    }
    max_count
}

pub fn mds_weights_fomula(q: usize, n: usize, k: usize) -> Vec<usize> {
    (0..=n)
        .map(|w| {
            let d = n - k + 1;
            if w == 0 {
                return 1;
            }
            if w < d {
                return 0;
            }
            let mut sum = 0usize;
            for i in 0..=(w - d) {
                let coeff = binomial_coeff(w - 1, i) * q.pow((w - d - i) as u32);
                if i.is_multiple_of(2) {
                    sum += coeff;
                } else {
                    sum -= coeff;
                }
            }
            sum * binomial_coeff(n, w) * (q - 1)
        })
        .collect()
}

pub fn print_code<F: Field, const N: usize, const K: usize, const R: usize>(code: &Code<F, N, K, R>)
where
    StandardUniform: Distribution<F>,
{
    let weights = code.weights();

    let weights_formula = mds_weights_fomula(F::MODULUS.to_usize().unwrap(), N, K);
    let weights_formula2 = mds_weights_fomula(F::MODULUS.to_usize().unwrap(), N - 1, K - 1);
    let weights_formula3 = mds_weights_fomula(F::MODULUS.to_usize().unwrap(), N - 1, K);
    println!("Weights formula:    {weights_formula:?} {weights_formula2:?} {weights_formula3:?}");

    for codeword in code.codewords() {
        let sizes = distogram(&code, codeword);
        assert_eq!(weights, sizes);
    }
    println!("Weights count:     {weights:?}");

    let mut spectra: HashMap<Vec<usize>, usize> = HashMap::new();
    for word in HammingIter::<F, N>::new(N) {
        let sizes = distogram(&code, word);
        *spectra.entry(sizes).or_default() += 1;
    }
    println!("Spectra count");

    // Sort spectra
    let mut spectra_list_max = vec![0usize; N + 1];
    let mut spectra: Vec<(Vec<usize>, usize)> = spectra.into_iter().collect();
    spectra.sort_by(|a, b| b.0.cmp(&a.0));
    for (spectrum, count) in spectra {
        println!("  {spectrum:?}: {count}");

        let mut cummulative = spectrum.clone();
        for i in 1..cummulative.len() {
            cummulative[i] += cummulative[i - 1];
        }
        for (max, count) in spectra_list_max.iter_mut().zip(cummulative.iter()) {
            if *count > *max {
                *max = *count;
            }
        }
    }
    println!("Spectral maximum: {spectra_list_max:?}");

    let max_count = max_distances(code);
    println!("Empirical maximum: {max_count:?}");
}

pub fn binomial_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result *= n - i;
        assert!(result.is_multiple_of(i + 1));
        result /= i + 1;
    }
    result
}

pub fn analyze<F: Field, const N: usize, const K: usize, const R: usize>()
where
    StandardUniform: Distribution<F>,
{
    let roots = array::from_fn(|i| F::from(F::UInt::from_usize(i).unwrap()));
    let code: Code<F, N, K, R> = Code::new_reed_solomon(roots).unwrap();

    // Compute distogram spectra
    let mut spectra: HashMap<Vec<usize>, usize> = HashMap::new();
    for word in HammingIter::<F, N>::new(N) {
        let sizes = distogram(&code, word);
        *spectra.entry(sizes).or_default() += 1;
    }
    let mut spectra: Vec<(Vec<usize>, usize)> = spectra.into_iter().collect();
    spectra.sort_by(|a, b| b.0.cmp(&a.0));
    for (spectrum, count) in spectra {
        println!("  {spectrum:?}: {count}");
    }
}

pub fn random_code<F: Field, const N: usize, const K: usize, const R: usize>() -> Code<F, N, K, R>
where
    StandardUniform: Distribution<F>,
{
    let mut rng = rand::rng();
    loop {
        // Random general code
        // let generator = rng.random();
        // if let Some(code) = Code::new(generator) {
        //     println!("Random linear code.");
        //     break code;
        // }

        // Random cyclic code
        // let generator: [Fp; R + 1] = rng.random();
        // if let Some(code) = Code::new_cyclic(generator) {
        //     println!("Random cyclic code with generator: {:?}", generator);
        //     break code;
        // }

        // Random Reed-Solomon code
        // All: [1, 1, 3, 20, 91, 209, 343]
        let eval_points = array::from_fn(|_| rng.random());
        let mut sorted = eval_points.to_vec();
        sorted.sort();
        sorted.dedup();
        if sorted.len() < N {
            continue;
        }
        println!("Eval points: {:?}", eval_points);
        break Code::new_reed_solomon(eval_points).expect("Points are unique.");
    }
}

fn main() {
    const_for!(P in [3,5,7,13,17,19,23,29,31,37,41,43,47,53,59,61] {
        type Fp = Fu8<{P as u8}>;
        const_for!(K in  [1, 2,3,4,5,6,7,8,9,10,12,13] {
            const_for!(R in [1,2,3,4,5,6,7,8,9,10,12,13] {
                const N: usize = K + R;
                if N <= P as usize {
                    println!("Analyzing [{P}, {N}, {K}] code:");
                    analyze::<Fp, N, K, R>();
                }
            });
        });
    });
}
