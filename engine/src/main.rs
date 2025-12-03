#![allow(dead_code)]

mod code;
mod coset_scale;
mod field;
mod hamming;
mod iter;
mod mat;
mod util;

use {
    crate::{
        code::Code,
        coset_scale::CosetScale,
        field::{Field, Fu8},
        hamming::{HammingIter, dist, weight},
        iter::{ProjectiveIter, SetIter, VecIter},
        mat::Mat,
        util::binomial_coefficient,
    },
    core::{
        array,
        sync::atomic::{AtomicUsize, Ordering},
    },
    num_traits::ToPrimitive,
    rand::{
        Rng,
        distr::{Distribution, StandardUniform},
    },
    std::{cmp::max, collections::HashMap, iter::repeat},
};

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
                let d = dist(candidate, &received);
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
pub fn distogram<F: Field>(code: &Code<F>, word: &[F]) -> Vec<usize>
where
    StandardUniform: Distribution<F>,
{
    let mut distogram = vec![0_usize; code.n() + 1];
    for codeword in code.codewords() {
        let d = dist(&codeword, word);
        distogram[d] += 1;
    }
    distogram
}

pub fn normalize_offset<F: Field>(offset: &mut [F], space: &Mat<F>)
where
    StandardUniform: Distribution<F>,
{
    let k = space.rows();
    let n = space.cols();
    assert!(k <= n);
    assert!(offset.len() == n);
    assert!(space.is_normal_form());

    for i in 0..k {
        if !offset[i].is_zero() {
            let coeff = offset[i];
            for j in 0..n {
                offset[j] -= coeff * space.row(i)[j];
            }
        }
    }
    assert!(offset[..k].iter().all(|x| x.is_zero()));
}

pub fn max_distances<F: Field>(code: &Code<F>) -> Vec<usize>
where
    StandardUniform: Distribution<F>,
{
    let mut max_count = vec![0usize; code.n() + 1];
    for word in HammingIter::<F>::new(code.n(), code.n()) {
        let mut count = vec![0usize; code.n() + 1];
        for codeword in code.codewords() {
            let d = dist(&codeword, &word);
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

pub fn weights_base(q: usize, n: usize, k: usize) -> Vec<usize> {
    let d = n - k + 1;
    (0..=n)
        .map(|w| {
            if w < d {
                return 0;
            }
            let mut sum = 0usize;
            for i in 0..=(w - d) {
                let coeff = binomial_coefficient(w - 1, i) * q.pow((w - d - i) as u32);
                if i.is_multiple_of(2) {
                    sum += coeff;
                } else {
                    sum -= coeff;
                }
            }
            sum * binomial_coefficient(n, w) * (q - 1)
        })
        .collect()
}

pub fn mds_weights_fomula(q: usize, n: usize, k: usize) -> Vec<usize> {
    let mut result = weights_base(q, n, k);
    result[0] = 1;
    result
}

pub fn spectrum<F: Field>(code: &Code<F>) -> HashMap<Vec<usize>, usize>
where
    StandardUniform: Distribution<F>,
{
    let q = F::MODULUS.to_usize().unwrap();
    let n = code.n();
    let k = code.k();
    let r = n - k;
    let block_size = q.pow(k as u32);
    // TODO: Missing another factor (q - 1). Must be the scaling of the offsets.

    let mut coset_scale = CosetScale::new(code.generator());

    let mut spectra: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut offset = vec![F::ZERO; n];
    if r == 0 {
        // Special case: no parity.
        let sizes = coset_scale.weigh(&offset);
        *spectra.entry(sizes).or_default() += block_size;
    } else {
        // General case: iterate over all syndromes
        for word in VecIter::<F>::new(r) {
            offset[k..].copy_from_slice(&word);
            let sizes = coset_scale.weigh(&offset);
            *spectra.entry(sizes).or_default() += 1;
        }
    }
    spectra
}

pub fn sort_spectrum(spectrum: &HashMap<Vec<usize>, usize>) -> Vec<(Vec<usize>, usize)> {
    let mut spectrum: Vec<(Vec<usize>, usize)> =
        spectrum.iter().map(|(k, v)| (k.clone(), *v)).collect();
    // Reverse lexicographic order is near-to-far
    spectrum.sort_by(|a, b| b.0.cmp(&a.0));
    spectrum
}

pub fn random_code<F: Field>(k: usize, r: usize) -> Code<F>
where
    StandardUniform: Distribution<F>,
{
    let n = k + r;
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
        let eval_points = (0..n).map(|_| rng.random()).collect::<Vec<F>>();
        let mut sorted = eval_points.to_vec();
        sorted.sort();
        sorted.dedup();
        if sorted.len() < n {
            continue;
        }
        println!("Eval points: {:?}", eval_points);
        break Code::new_reed_solomon(k, &eval_points, false).expect("Points are unique.");
    }
}

fn max_list_sizes(spectra: &HashMap<Vec<usize>, usize>) -> Vec<usize> {
    let n = spectra.keys().next().unwrap().len() - 1;
    let mut max_sizes = vec![0usize; n + 1];
    for (spectrum, count) in spectra {
        let mut cumulative = spectrum.clone();
        for i in 1..cumulative.len() {
            cumulative[i] += cumulative[i - 1];
        }
        for (list_max, list_size) in max_sizes.iter_mut().zip(cumulative.iter()) {
            *list_max = max(*list_max, *list_size);
        }
    }
    max_sizes
}

fn print_spectra<F: Field>(code: &Code<F>)
where
    StandardUniform: Distribution<F>,
{
    let q = F::MODULUS.to_usize().unwrap();
    let n = code.n();
    let k = code.k();
    let r = n - k;
    let d = r + 1;
    let block_size = q.pow(k as u32);

    let spectrum = spectrum::<F>(code);
    let sorted = sort_spectrum(&spectrum);
    let max_sizes = max_list_sizes(&spectrum);
    println!("Max list  : {:?}", max_sizes);

    let weights_base = weights_base(q, n, k);
    println!("Weights base: {weights_base:?}");
    let weights_multiples = (0..=n)
        .map(|i| binomial_coefficient(n, i))
        .collect::<Vec<_>>();
    println!("Weights multiples: {weights_multiples:?}");

    // MDS weights formula, matches the zero coset.
    let coset_zero = mds_weights_fomula(q, n, k);
    println!("Coset zero: {coset_zero:?}");

    // Lemma 7.5.1 ii assuming D is MDS.
    // If often matches one of the cosets.
    // It also seems to maximize A_{d - 1}
    let coset_mds = mds_weights_fomula(q, n, k + 1)
        .into_iter()
        .zip(weights_base.iter().copied().chain(repeat(0)))
        .map(|(d, c)| (d - c) / (q - 1))
        .collect::<Vec<_>>();
    println!("Coset mds : {coset_mds:?}");
    let mds_delta = coset_mds
        .iter()
        .zip(weights_base.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .collect::<Vec<_>>();
    println!("MDS delta : {mds_delta:?}");

    println!("Found spectrum with {} distinct entries.", sorted.len());
    for (weight_distribution, count) in &sorted {
        println!("  {weight_distribution:?}: {count} * {block_size}");
    }
    println!("Conjecture explanation remaining errors:");
    for (weight_distribution, count) in &sorted {
        let delta = weight_distribution
            .iter()
            .zip(weights_base.iter())
            .enumerate()
            .map(|(i, (a, b))| {
                if i >= d - 1 && (i - d).is_multiple_of(2) {
                    assert!(*b >= *a, "Expected base weights to be larger at index {i}");
                    b - a
                } else {
                    assert!(
                        *a >= *b,
                        "Expected spectrum weights to be larger at index {i}"
                    );
                    a - b
                }
            })
            .zip(mds_delta.iter().copied())
            .enumerate()
            .map(|(i, (diff, mds))| {
                if i >= d - 1 {
                    // assert!(
                    //     mds >= diff,
                    //     "Expected mds coset weights to be larger at index {i}"
                    // );
                    mds - diff
                } else {
                    diff
                }
            })
            .collect::<Vec<_>>();
        println!("  {delta:?}: {count} * {block_size}");
    }
}

fn enumerate_spectra() {
    const_for!(P in [2,3,5,7,13,17,19,23,29,31,37,41,43,47,53,59,61] {
        type Fp = Fu8<{P as u8}>;
        for n in 1..=P {
            let eval_points = (0..n)
                .map(|i| Fp::from_u64(i as u64))
                .collect::<Vec<Fp>>();
            for k in 1..=n {
                println!("Testing [{n}, {k}]_{P} Reed-Solomon codes...");
                let code = Code::new_reed_solomon(k, &eval_points, false).expect("eval points are distinct");
                print_spectra(&code);
            }
        }
        println!("Verified up to {P}");
    });
}

fn test_rs_equivalence() {
    const_for!(P in [2,3,5,7,13,17,19,23,29,31,37,41,43,47,53,59,61] {
        type Fp = Fu8<{P as u8}>;

        for n in 1..=(P+1) {
            for k in 1..=n {
                println!("Testing [{P}, {n}, {k}] Reed-Solomon codes...");
                let mut spectra: HashMap<Vec<(Vec<usize>, usize)>, usize> = HashMap::new();
                for eval_points in SetIter::<Fp>::new(n) {
                    let code = Code::new_reed_solomon(k, &eval_points, false).expect("eval points are distinct");
                    let spectrum = spectrum::<Fp>(&code);
                    let sorted = sort_spectrum(&spectrum);
                    *spectra.entry(sorted).or_default() += 1;
                }
                if n == 1 {
                    let code = Code::new_reed_solomon(k, &[], true).expect("eval points are distinct");
                    let spectrum = spectrum::<Fp>(&code);
                    let sorted = sort_spectrum(&spectrum);
                    *spectra.entry(sorted).or_default() += 1;
                } else {
                    for eval_points in SetIter::<Fp>::new(n - 1) {
                        let code = Code::new_reed_solomon(k, &eval_points, true).expect("eval points are distinct");
                        let spectrum = spectrum::<Fp>(&code);
                        let sorted = sort_spectrum(&spectrum);
                        *spectra.entry(sorted).or_default() += 1;
                    }
                }
                println!("Found {} distinct spectra: {:?}", spectra.len(), spectra.values().copied().collect::<Vec<_>>());
            }
        }
        println!("Verified up to {P}");
    });
}

fn main() {
    // print_spectra(&code);
    //
    enumerate_spectra();
}
