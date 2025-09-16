use crate::constants::NO_OPT_PATH;
use std::collections::HashMap;
use tch::Tensor;

/// A small wrapper providing a more ergonomic, name-based einsum.
///
/// Spec format examples (whitespace between dims; comma between inputs):
/// - "i i ->"            (dot product / sum over i)
/// - "a b, b c -> a c"  (matrix multiply)
/// - "h w, h w ->"      (sum of elementwise product)
/// - "b c h w, h w -> b c" (batched contraction)
///
/// The function converts names into compact PyTorch-style labels (a-zA-Z)
/// and calls `tch::Tensor::einsum`.
pub fn named_einsum(spec: &str, inputs: &[Tensor]) -> Tensor {
    // Split LHS and optional RHS
    let parts: Vec<&str> = spec.split("->").collect();
    let lhs = parts[0].trim();
    let rhs = parts.get(1).map(|s| s.trim());

    // Tokenize inputs: split by comma, then by whitespace
    let input_specs: Vec<Vec<String>> = lhs
        .split(',')
        .map(|arg| {
            arg.split_whitespace()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        })
        .collect();

    // Collect unique labels in order of first appearance
    let mut label_map: HashMap<String, char> = HashMap::new();
    let mut next_labels: Vec<char> = (b'a'..=b'z')
        .chain(b'A'..=b'Z')
        .map(|c| c as char)
        .collect();

    let mut get_label = |name: &str| -> char {
        if let Some(&c) = label_map.get(name) {
            c
        } else {
            assert!(
                !next_labels.is_empty(),
                "named_einsum: ran out of symbols (supports up to 52 unique dims)"
            );
            let c = next_labels.remove(0);
            label_map.insert(name.to_string(), c);
            c
        }
    };

    // Build compact inputs
    let compact_inputs: Vec<String> = input_specs
        .iter()
        .map(|dims| dims.iter().map(|d| get_label(d)).collect())
        .collect();

    // Optional compact output
    let compact_rhs = rhs.map(|r| {
        if r.is_empty() {
            String::new()
        } else {
            r.split_whitespace().map(get_label).collect()
        }
    });

    // Compose compact equation string
    let mut eq = compact_inputs.join(",");
    if let Some(out) = compact_rhs {
        eq.push_str("->");
        eq.push_str(&out);
    }

    // Call tch einsum
    let owned: Vec<Tensor> = inputs.iter().map(|t| t.shallow_clone()).collect();
    Tensor::einsum(&eq, &owned, NO_OPT_PATH)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_einsum_dot() {
        let a = Tensor::f_from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let b = Tensor::f_from_slice(&[4.0, 5.0, 6.0]).unwrap();
        let s = named_einsum("i, i ->", &[a, b]);
        let val = s.double_value(&[]);
        assert!((val - (1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0)).abs() < 1e-12);
    }

    #[test]
    fn test_named_einsum_matmul() {
        let a = Tensor::arange(6, (tch::Kind::Float, tch::Device::Cpu)).view([2, 3]);
        let b = Tensor::arange(12, (tch::Kind::Float, tch::Device::Cpu)).view([3, 4]);
        let c1 = named_einsum("m n, n p -> m p", &[a.shallow_clone(), b.shallow_clone()]);
        let c2 = a.matmul(&b);
        let diff = (c1.shallow_clone() - c2)
            .abs()
            .sum(c1.kind())
            .double_value(&[]);
        assert!(diff < 1e-8);
    }
}
