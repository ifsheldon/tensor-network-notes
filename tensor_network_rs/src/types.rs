use num_traits::{NumCast, PrimInt};

/// Non-negative integer type used for any natural number, like number of qubits, number of samples, length of MPS, etc.
pub type Num = u64;
/// indices that allow negative values, negative indices follow the same convention in Python
pub type Idx = i64;
/// Non-negative integer type used for qubit indices
pub type UIdx = u64;
/// The integer type used for tch-rs Tensor indices, never use this in public APIs.
pub type TInt = i64;

pub trait SafeCastTo<T> {
    fn cast(self) -> T;
}

impl<S: PrimInt, T: PrimInt> SafeCastTo<T> for S {
    #[inline]
    fn cast(self) -> T {
        NumCast::from(self).unwrap()
    }
}

pub trait SafeCastItems<T>: Iterator {
    type Output: Iterator<Item = T>;
    fn cast_items(self) -> Self::Output;
}

impl<I, S, T> SafeCastItems<T> for I
where
    I: Iterator<Item = S>,
    S: PrimInt + SafeCastTo<T>,
    T: PrimInt,
{
    type Output = std::iter::Map<I, fn(S) -> T>;

    #[inline]
    fn cast_items(self) -> Self::Output {
        self.map(<S as SafeCastTo<T>>::cast)
    }
}

pub trait SafeCastItem<T> {
    type Output;
    fn cast(self) -> Self::Output;
}

impl<S, T> SafeCastItem<T> for Option<S>
where
    S: PrimInt + SafeCastTo<T>,
    T: PrimInt,
{
    type Output = Option<T>;
    #[inline]
    fn cast(self) -> Self::Output {
        self.map(<S as SafeCastTo<T>>::cast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn test_num_cast() {
        let a = -1;
        let b: Num = a.cast();
    }
}
