use crate::tensor_gates::functional::apply_gate;
use tch::Tensor;

pub fn time_evolving_block_decimation(
    initial_state: &Tensor,
    two_site_gate: &Tensor,
    positions: &[Vec<i64>],
    steps: i64,
) -> Tensor {
    let mut state = initial_state.shallow_clone();
    for _ in 0..steps {
        for pos in positions {
            state = apply_gate(&state, two_site_gate, pos.clone(), None);
        }
        state = &state / state.norm();
    }
    state
}
