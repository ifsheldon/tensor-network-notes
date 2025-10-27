//! # 指标操作与张量基本运算
//!
//! 基本操作见 [simple_tensor_operations]
//!
//! ## 向量化与矩阵化的数学公式符号约定:
//!
//! ## 向量化与矩阵化的数学公式符号约定:
//!
//! 张量的矩阵化。如果将某几个指标(例如 $i_a,i_b,i_c$) 合并作为左指标,其余合并作为右指标,该矩阵简记为 $T_{\[i_a i_b i_c\]}$ , 在不引起误解的情况下,剩余的指标可以不写出来,或也可以写在第二个方括号中,记为 $T_{\[i_a i_b i_c\][\cdots]}$ 。若不希望指定代表各个指标的名字 (字母), 可以写成 $T_{\[0,2,\cdots]}$ , 方括号中的数字代表张量的第几个指标。
//!
//! 如果将某一个指标 $i_m$ 作为矩阵左指标, 并合并其余指标作为矩阵右指标,该矩阵简记为 $T_{\[i_m\]}$ ; 如果将前 $m$ 个指标合并作为左指标,剩下的指标合并作为右指标,该矩阵简记为 $T_{\[i_0 \cdots i_{m-1}\][i_m \cdots i_{N-1}]} $ 或 $T_{\[i_0 \cdots i_{m-1}\][\cdots]} $ 或 $T_{\[0 \cdots m-1\]}$. 如果需要将一个张量的所有指标合并成一个指标,则以此获得的向量可简记为 $T_{\[:\]}$ , 其也被称为 $T$ 的向量化 (vectorization)
//!
//! ## 外积
//!
//! 外积运算: $U = X \otimes Y \leftrightarrow U_{abc...ijk...} = X_{abc...}Y_{ijk...}$
//!
//! 注意左张量的维度在前，右张量的维度在后
//!
//! 见 [outer_product]
//!
//! ### Einsum
//!
//! *   其输入为字符串公式与相关张量；
//! *   公式包含箭头 “->”，箭头左侧为各个待收缩张量的指标，右侧为收缩所得张量的指标；
//! *   左侧各个张量的指标用逗号隔开，共有指标使用同一个字母表示；
//! *   当左侧出现的指标没有出现在右侧时，说明对该指标作求和运算；
//!
//! 例子:
//! `Tensor::einsum("a,b -> ab", x, y, NO_OPT_PATH)`
//! 从 outer 对应的 einsum 可以容易看出 outer 的计算公式为 $w_{ab} = u_a v_b$。
//!
//! ### Kronecker Product
//!
//! Einsum 公式:
//! $u_av_b = w_{ab} \rightarrow w_{\[ab\]}$

use crate::utils::{NO_OPT_PATH, allclose};
use tch::Tensor;

///
/// ```
/// use tensor_network_rs::lecture_code::lecture_1_2::simple_tensor_operations;
/// simple_tensor_operations();
/// ```
pub fn simple_tensor_operations() {
    let x = Tensor::from_slice(&[1.0_f64, 2.0, 3.0, 4.0]);
    let y = x.reshape([2, 2]);
    assert_eq!(y.double_value(&[1, 0]), 3.0);
    let x = y.permute(&[1, 0]);
    assert_eq!(x.double_value(&[0, 1]), 3.0);
}

///
/// ```
/// use tensor_network_rs::lecture_code::lecture_1_2::outer_product;
/// outer_product();
/// ```
pub fn outer_product() {
    let x = Tensor::from_slice(&[1, 3, 5]);
    let y = Tensor::from_slice(&[2, 4, 6, 8]);
    let z1 = x.outer(&y);
    let z2 = Tensor::einsum("a, b -> ab", &[&x, &y], NO_OPT_PATH);
    assert!(allclose(&z1, &z2, None, None, false).unwrap_or(false));
    // equivalently
    let z3 = x.reshape([3, 1]) * y.reshape([1, 4]);
    assert!(allclose(&z1, &z3, None, None, false).unwrap_or(false));
}

///
/// ```
/// use tensor_network_rs::lecture_code::lecture_1_2::kronecker_product;
/// kronecker_product();
/// ```
pub fn kronecker_product() {
    let u = Tensor::from_slice(&[1, 3, 5]);
    let v = Tensor::from_slice(&[2, 4, 6, 8]);
    let z1 = Tensor::kron(&u, &v);
    let z2 = Tensor::einsum("a, b -> ab", &[&u, &v], NO_OPT_PATH).flatten(0, 1);
    assert!(allclose(&z1, &z2, None, None, false).unwrap_or(false));
}
