use {
    ::rand::{
        Rng,
        distr::{Distribution, StandardUniform, uniform::SampleUniform},
    },
    core::{
        fmt::{Debug, Display},
        ops,
    },
    num_traits::{ConstOne, ConstZero, Inv, Num, NumAssign, One, Pow, PrimInt, Zero},
};

pub trait Field:
    'static
    + Send
    + Sync
    + Copy
    + Eq
    + NumAssign
    + Inv
    + Pow<usize>
    + From<Self::UInt>
    + Into<Self::UInt>
    + ConstZero
    + ConstOne
    + Debug
    + Display
where
    StandardUniform: Distribution<Self>,
{
    type UInt: 'static + Send + Sync + Copy + PrimInt + ConstZero + ConstOne + SampleUniform;

    const MODULUS: Self::UInt;
}

macro_rules! impl_field {
    ($name:ident, $uint:ty, $wide:ty) => {
        #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name<const MODULUS: $uint>($uint);

        impl<const MODULUS: $uint> Field for $name<MODULUS> {
            type UInt = $uint;

            const MODULUS: Self::UInt = MODULUS;
        }

        impl<const MODULUS: $uint> Distribution<$name<MODULUS>> for StandardUniform {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $name<MODULUS> {
                $name(rng.random_range(..MODULUS))
            }
        }

        impl<const MODULUS: $uint> Debug for $name<MODULUS> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                <$uint as Debug>::fmt(&self.0, f)
            }
        }

        impl<const MODULUS: $uint> Display for $name<MODULUS> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                <$uint as Display>::fmt(&self.0, f)
            }
        }

        impl<const MODULUS: $uint> From<$uint> for $name<MODULUS> {
            fn from(value: $uint) -> Self {
                $name(value % MODULUS)
            }
        }

        impl<const MODULUS: $uint> From<$name<MODULUS>> for $uint {
            #[inline(always)]
            fn from(value: $name<MODULUS>) -> Self {
                value.0
            }
        }

        impl<const MODULUS: $uint> Num for $name<MODULUS> {
            type FromStrRadixErr = core::num::ParseIntError;

            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$uint>::from_str_radix(str, radix).map($name::from)
            }
        }

        impl<const MODULUS: $uint> ConstZero for $name<MODULUS> {
            const ZERO: Self = $name(0);
        }

        impl<const MODULUS: $uint> Zero for $name<MODULUS> {
            #[inline(always)]
            fn zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn is_zero(&self) -> bool {
                self == &Self::ZERO
            }
        }

        impl<const MODULUS: $uint> ConstOne for $name<MODULUS> {
            const ONE: Self = $name(1);
        }

        impl<const MODULUS: $uint> One for $name<MODULUS> {
            #[inline(always)]
            fn one() -> Self {
                Self::ONE
            }

            #[inline(always)]
            fn is_one(&self) -> bool {
                self == &Self::ONE
            }
        }

        impl<const MODULUS: $uint> ops::Add for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn add(self, other: Self) -> Self {
                $name(((self.0 as $wide + other.0 as $wide) % (MODULUS as $wide)) as $uint)
            }
        }

        impl<const MODULUS: $uint> ops::AddAssign for $name<MODULUS> {
            #[inline(always)]
            fn add_assign(&mut self, other: Self) {
                *self = *self + other;
            }
        }

        impl<const MODULUS: $uint> ops::Neg for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn neg(self) -> Self {
                $name((MODULUS - self.0) % MODULUS)
            }
        }

        impl<const MODULUS: $uint> ops::Sub for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                $name(
                    ((self.0 as $wide + MODULUS as $wide - other.0 as $wide) % MODULUS as $wide)
                        as $uint,
                )
            }
        }

        impl<const MODULUS: $uint> ops::SubAssign for $name<MODULUS> {
            #[inline(always)]
            fn sub_assign(&mut self, other: Self) {
                *self = *self - other;
            }
        }

        impl<const MODULUS: $uint> ops::Mul for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                $name(((self.0 as $wide * other.0 as $wide) % MODULUS as $wide) as $uint)
            }
        }

        impl<const MODULUS: $uint> ops::MulAssign for $name<MODULUS> {
            #[inline(always)]
            fn mul_assign(&mut self, other: Self) {
                *self = *self * other;
            }
        }

        impl<const MODULUS: $uint> ops::Div for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn div(self, other: Self) -> Self {
                self * other.inv()
            }
        }

        impl<const MODULUS: $uint> ops::DivAssign for $name<MODULUS> {
            #[inline(always)]
            fn div_assign(&mut self, other: Self) {
                *self *= other.inv();
            }
        }

        impl<const MODULUS: $uint> ops::Rem for $name<MODULUS> {
            type Output = Self;

            fn rem(self, _other: Self) -> Self {
                unimplemented!()
            }
        }

        impl<const MODULUS: $uint> ops::RemAssign for $name<MODULUS> {
            fn rem_assign(&mut self, _other: Self) {
                unimplemented!()
            }
        }

        impl<const MODULUS: $uint> Inv for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn inv(self) -> Self {
                self.pow((MODULUS - 2) as usize)
            }
        }

        impl<const MODULUS: $uint> Pow<usize> for $name<MODULUS> {
            type Output = Self;

            #[inline(always)]
            fn pow(self, mut exp: usize) -> Self {
                let mut result = $name::ONE;
                let mut base = self;

                while exp > 0 {
                    if exp % 2 == 1 {
                        result *= base;
                    }
                    base = base * base;
                    exp /= 2;
                }
                result
            }
        }
    };
}

impl_field!(Fu8, u8, u16);
impl_field!(Fu16, u16, u32);
impl_field!(Fu32, u32, u64);
impl_field!(Fu64, u64, u128);
