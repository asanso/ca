#[macro_export]
macro_rules! const_for {
    ($C:ident in [$($n:expr),*] $x:block) => {
        const_for!(($C: usize) in [$($n),*] $x);
    };
    (($C:ident: $T:ty) in [$($n:expr),*] $x:block) => {
        $({
            const $C: $T = $n;
            $x
        })*
    };
}

pub fn binomial_coefficient(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        result *= n - i;
        result /= i + 1;
    }
    result
}
