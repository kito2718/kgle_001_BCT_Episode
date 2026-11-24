"""Reversible Reed-Solomon syndrome decoder for Decoded Quantum Interferometry (DQI).

Implements the measurement-based uncomputation (MBUC) Reed-Solomon decoder
circuit described in https://arxiv.org/pdf/2510.10967. Given quantum registers
holding the syndromes `S_0, ..., S_{n-1}` over `GF(2^b)` and classical registers
`error_mbuc` holding the X-basis measurement outcomes of each codeword position,
the circuit applies the phase

    (-1) ** sum_j popcount(error[j] & error_mbuc[j])

where `error = decode(syndromes)`, and restores all syndrome and ancilla qubits.

The circuit runs in three stages:
1. Forward synchronized Extended Euclidean Algorithm (EEA) to compute the error
   locator polynomial `sigma` and error evaluator polynomial `omega`.
2. Chien search and Forney's formula to evaluate the error at each position and
   kick back the MBUC phase.
3. Reverse synchronized EEA to uncompute `sigma` and `omega` and restore the
   syndrome registers.

Usage (writes a Benchmark II submission for `verify_submission_dqi.py`):
    python dqi/dqi_reference_circuit.py decoder.kmb [--m 4095] [--n 70] [--degree 12] [--format kmb|kmx]
"""

from __future__ import annotations

import argparse
from typing import Sequence

import kickmix as km

# Out-of-range address value used to park unary iteration cursors with match_qubit = |0>.
_UNREACHABLE = 1 << 63


def _active(cursor: km.UnaryIterationCursor, address_value: int, limit: int):
    """Moves `cursor` to `address_value` and returns `control and (x == address_value)`.

    When `address_value > limit` (outside the caller's upper bound `x <= limit`),
    parks `cursor` out of range and returns `False` so controlled operations
    become no-ops.
    """
    if address_value > limit:
        cursor.move_to(_UNREACHABLE)
        return False
    cursor.move_to(address_value)
    return cursor.match_qubit


def _xor_threshold(builder: km.CircuitBuilder, cursor: km.UnaryIterationCursor,
                   limit: int, target, control, value: int, *,
                   at_least: bool, lowest: int = 0) -> None:
    """Initializes or clears `target ^= control and (x >= value)` (or `x < value`)."""
    if value > limit:
        if not at_least:
            builder.cx(control, target)
        return
    if at_least:
        builder.cx(control, target)
    for u in range(lowest, value):
        builder.cx(_active(cursor, u, limit), target)


class ProosAndZalkaDecoder:
    """Builds the reversible Reed-Solomon MBUC decoder circuit (arXiv:2510.10967)."""

    def __init__(self, n_syndromes: int, field: km.GF2Field):
        if n_syndromes < 2:
            raise ValueError(f'n_syndromes must be at least 2, got {n_syndromes}.')
        self.n_syndromes = n_syndromes
        self.field = field

    def build_syndrome_decoding_mbuc(self, builder: km.CircuitBuilder, *,
                                     syndromes: Sequence[km.array],
                                     error_mbuc: Sequence[km.array],
                                     n_iters: int) -> None:
        """Appends the syndrome-decoding MBUC phase circuit to `builder`.

        Args:
            builder: Circuit builder receiving the instructions.
            syndromes: `n_syndromes` quantum registers of width `field.degree`.
                Restored to their initial state on completion.
            error_mbuc: Classical bit registers of width `field.degree`, one per
                codeword position, holding X-basis measurement outcomes.
            n_iters: Number of synchronized EEA iterations (`default_n_iters(n)`).
        """
        gf = self.field
        n = self.n_syndromes
        degree = gf.degree
        if len(syndromes) != n:
            raise ValueError(f'Expected {n} syndrome registers, got {len(syndromes)}.')
        if not (n <= len(error_mbuc) <= (1 << degree) - 1):
            raise ValueError(
                f'Expected between {n} and {(1 << degree) - 1} error_mbuc registers, '
                f'got {len(error_mbuc)}.'
            )
        if any(len(s) != degree for s in syndromes):
            raise ValueError(f'Every syndrome register must have length {degree}.')
        if any(len(e) != degree for e in error_mbuc):
            raise ValueError(f'Every error_mbuc register must have length {degree}.')
        m = n + 3

        reg_a = [builder.alloc_qubits(degree) if i < 3 else syndromes[i - 3] for i in range(m)]
        reg_b = [builder.alloc_qubits(degree) for _ in range(m)]

        logm = m.bit_length() + 1
        n_A = builder.alloc_qubits(logm)
        n_B = builder.alloc_qubits(logm)
        n_a = builder.alloc_qubits(logm)
        n_b = builder.alloc_qubits(logm)
        n_q = builder.alloc_qubits(logm)
        flag = builder.alloc_qubits(1)[0]
        counter = builder.alloc_qubits(2)
        halting_counter = builder.alloc_qubits(n_iters.bit_length())
        eea = (reg_a, reg_b, n_A, n_B, n_a, n_b, n_q, flag, counter, halting_counter)

        init_ones = km.array([reg_a[0][0], reg_b[m - 1][0], n_a[0], flag])
        builder.x(init_ones)
        builder.gf2_iadd(n, target=n_A)
        builder.gf2_iadd(n + 1, target=n_B)

        with builder.mark('initial_shift'):
            self._initial_shift_loop(builder, reg_a, n_A, is_adjoint=False)
        with builder.mark('forward_eea'):
            self._synchronized_eea(builder, *eea, n_iters=n_iters, is_adjoint=False)
        with builder.mark('chien_forney_mbuc'):
            self._chien_search_and_forney_mbuc(builder, reg_b, error_mbuc, n_b, n_B)
        with builder.mark('reverse_eea'):
            self._synchronized_eea(builder, *eea, n_iters=n_iters, is_adjoint=True)
        with builder.mark('uncompute_initial_shift'):
            self._initial_shift_loop(builder, reg_a, n_A, is_adjoint=True)

        builder.gf2_iadd(n + 1, target=n_B)
        builder.gf2_iadd(n, target=n_A)
        builder.x(init_ones)

        for reg in (halting_counter, counter, flag, n_q, n_b, n_a, n_B, n_A, *reg_a[:3], *reg_b):
            builder.free(reg)

    def _initial_shift_loop(self, builder: km.CircuitBuilder,
                            reg_a: Sequence[km.array], n_A: km.array, *,
                            is_adjoint: bool) -> None:
        """Left-aligns the syndrome polynomial in `reg_a` and updates `n_A`."""
        n = self.n_syndromes
        m = len(reg_a)
        shift_step = builder.mark('shift_step')
        for step in (range(n - 2, -1, -1) if is_adjoint else range(n - 1)):
            with shift_step:
                ctrl = builder.alloc_qubits(1)[0]
                if is_adjoint:
                    builder.flip_if_less_than(n_A, n - step, target=ctrl)
                    builder.iadd(1, target=n_A, control=ctrl)
                    for b in range(self.field.degree):
                        lane = km.array([r[b] for r in reg_a[3 + step:m]])
                        builder.right_rotate(lane, 1, control=ctrl)
                    builder.flip_if_equal(reg_a[m - 1], 0, target=ctrl)
                else:
                    builder.flip_if_equal(reg_a[m - 1], 0, target=ctrl)
                    for b in range(self.field.degree):
                        lane = km.array([r[b] for r in reg_a[3 + step:m]])
                        builder.left_rotate(lane, 1, control=ctrl)
                    builder.isub(1, target=n_A, control=ctrl)
                    builder.flip_if_less_than(n_A, n - step, target=ctrl)
                builder.free(ctrl)

    def _synchronized_eea(self, builder: km.CircuitBuilder,
                          reg_a: Sequence[km.array], reg_b: Sequence[km.array],
                          n_A: km.array, n_B: km.array, n_a: km.array,
                          n_b: km.array, n_q: km.array, flag,
                          counter: km.array, halting_counter: km.array, *,
                          n_iters: int, is_adjoint: bool) -> None:
        """Runs `n_iters` steps of the reversible synchronized EEA."""
        eea_step = builder.mark('eea_step')
        for it in (range(n_iters - 1, -1, -1) if is_adjoint else range(n_iters)):
            with eea_step:
                if is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=True)

                is_c3 = builder.alloc_qubits(1)[0]
                builder.init_and(counter[0], counter[1], is_c3)
                if is_adjoint:
                    self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter,
                                                 flag, is_c3, is_adjoint=True)

                is_hc_zero = builder.alloc_qubits(1)[0]
                builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                ctrl = self._counter_controls(builder, counter, is_c3, is_hc_zero)

                steps = [
                    lambda: self._update_flag(builder, reg_b, n_A, n_B, n_a, n_b, n_q,
                                              flag, ctrl),
                    lambda: self._multiply_and_add_or_sub(builder, reg_a, reg_b, n_A, n_a,
                                                          n_q, ctrl, it,
                                                          is_adjoint=is_adjoint),
                    lambda: self._shift_b_array_right(builder, reg_b, n_B, n_b, flag, ctrl,
                                                      it, is_adjoint=is_adjoint),
                    lambda: self._update_metadata(builder, n_B, n_b, n_q, flag, ctrl,
                                                  is_adjoint=is_adjoint),
                    lambda: self._swap_a_and_b(builder, reg_a, reg_b, n_A, n_B, n_a, n_b,
                                               flag, ctrl),
                ]
                for step in (reversed(steps) if is_adjoint else steps):
                    step()

                self._del_counter_controls(builder, counter, is_c3, is_hc_zero, ctrl)
                builder.flip_if_equal(halting_counter, 0, target=is_hc_zero)
                builder.free(is_hc_zero)

                if not is_adjoint:
                    self._update_halting_counter(builder, n_A, n_B, n_b, halting_counter,
                                                 flag, is_c3, is_adjoint=False)
                builder.del_and(counter[0], counter[1], is_c3)
                builder.free(is_c3)
                if not is_adjoint:
                    self._update_counter(builder, counter, flag, is_adjoint=False)

    @staticmethod
    def _counter_controls(builder: km.CircuitBuilder, counter: km.array, is_c3,
                          is_hc_zero) -> km.array:
        """One-hot decodes `counter == 0..3` gated by `is_hc_zero` using 3 Toffolis."""
        ctrl = builder.alloc_qubits(4)
        c0, c1, c2, c3 = ctrl
        builder.init_and(is_hc_zero, km.array([counter[0], is_c3]), km.array([c1, c3]))
        builder.cx(km.array([is_hc_zero, c1]), c0)
        builder.init_and(c0, counter[1], c2)
        builder.cx(km.array([c2, c3]), km.array([c0, c1]))
        return ctrl

    @staticmethod
    def _del_counter_controls(builder: km.CircuitBuilder, counter: km.array, is_c3,
                              is_hc_zero, ctrl: km.array) -> None:
        c0, c1, c2, c3 = ctrl
        builder.cx(km.array([c2, c3]), km.array([c0, c1]))
        builder.del_and(c0, counter[1], c2)
        builder.cx(km.array([is_hc_zero, c1]), c0)
        builder.del_and(is_hc_zero, km.array([counter[0], is_c3]), km.array([c1, c3]))
        builder.free(ctrl)

    @staticmethod
    def _update_counter(builder: km.CircuitBuilder, counter: km.array, flag, *,
                        is_adjoint: bool) -> None:
        """Increments the 2-bit `counter` mod 4 when `flag` is set."""
        if is_adjoint:
            builder.cx(flag, counter[0])
            builder.ccx(flag, counter[0], counter[1])
        else:
            builder.ccx(flag, counter[0], counter[1])
            builder.cx(flag, counter[0])

    def _update_flag(self, builder: km.CircuitBuilder, reg_b: Sequence[km.array],
                     n_A: km.array, n_B: km.array, n_a: km.array, n_b: km.array,
                     n_q: km.array, flag, ctrl) -> None:
        """Toggles `flag` when the active EEA sub-step reaches its termination condition."""
        with builder.mark('update_flag'):
            c0, c1, c2, c3 = ctrl
            m = len(reg_b)
            aux = builder.alloc_qubits(1)[0]

            # Sub-step 0 finishes when deg(A) < deg(B) (if n_q == 0) or deg(A) == deg(B).
            builder.flip_if_less_than(n_A, n_B, target=aux, control=c0)
            builder.flip_if_equal(n_q, 0, target=flag, control=aux)
            builder.flip_if_less_than(n_A, n_B, target=aux, control=c0)
            builder.flip_if_equal(n_A, n_B, target=flag, control=c0)

            # Sub-step 1 finishes once B's leading coefficient is aligned.
            builder.flip_if_equal(n_A, n_B, target=flag, control=c1)
            builder.cx(c1, flag)
            builder.flip_if_equal(n_B, 1, target=aux, control=c1)
            builder.cx(c1, aux)
            builder.flip_if_equal(reg_b[m - 2], 0, target=flag, control=aux)
            builder.cx(c1, aux)
            builder.flip_if_equal(n_B, 1, target=aux, control=c1)

            # Sub-step 2 finishes once the auxiliary polynomial is aligned.
            builder.cx(c2, flag)
            builder.flip_if_equal(n_b, 0, target=aux, control=c2)
            builder.cx(c2, aux)
            builder.flip_if_equal(reg_b[0], 0, target=flag, control=aux)
            builder.cx(c2, aux)
            builder.flip_if_equal(n_b, 0, target=aux, control=c2)
            builder.iadd(1, target=n_b)
            builder.flip_if_equal(n_a, n_b, target=flag, control=c2)
            builder.isub(1, target=n_b)

            # Sub-step 3 finishes once all quotient terms are applied.
            builder.flip_if_equal(n_a, n_b, target=flag, control=c3)
            builder.flip_if_equal(n_q, 1, target=flag, control=c3)

            builder.free(aux)

    def _multiply_and_add_or_sub(self, builder: km.CircuitBuilder,
                                 reg_a: Sequence[km.array], reg_b: Sequence[km.array],
                                 n_A: km.array, n_a: km.array, n_q: km.array,
                                 ctrl, it: int, *, is_adjoint: bool) -> None:
        """Computes the leading-coefficient ratio `q_i` and updates `reg_b ^= q_i * reg_a`."""
        c0_hc, _, _, c3_hc = ctrl
        gf = self.field
        m = len(reg_b)

        def accumulate_quotient(index: int, control) -> None:
            with builder.mark('quotient_div'):
                ratio, scaffold = builder.init_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], field=gf)
                builder.gf2_iadd(ratio, target=q_i, field=gf, control=control)
                builder.del_gf2_div_with_scaffold(
                    reg_b[index], reg_a[index], target=ratio, scaffold=scaffold, field=gf)
                builder.free(scaffold + ratio)

        def sweep_positions() -> None:
            with builder.mark('sweep_positions'):
                lim_q = min(m - 1, 1 + it)
                lim_q1 = min(m, 1 + it)
                lim_a = min(m - 2, it // 2 + 1)
                lim_A = m - 2
                order = range(m) if is_adjoint else range(m - 1, -1, -1)

                with (
                    builder.unary_iteration(n_q, control=c0_hc) as eq_i,
                    builder.unary_iteration(n_q, control=c3_hc) as eq_ip1,
                    builder.unary_iteration(n_a, control=c3_hc) as cur_a,
                    builder.unary_iteration(n_A, control=c0_hc) as cur_A,
                ):
                    mul = builder.alloc_qubits(1)[0]

                    def endpoint(i: int) -> None:
                        _xor_threshold(builder, cur_a, lim_a, mul, c3_hc, i + 1,
                                       at_least=True, lowest=1)
                        _xor_threshold(builder, cur_A, lim_A, mul, c0_hc, m - i,
                                       at_least=True, lowest=1)

                    prev = None
                    for i in order:
                        if prev is None:
                            endpoint(i)
                        else:
                            builder.cx(
                                km.array([
                                    _active(cur_a, min(prev, i) + 1, lim_a),
                                    _active(cur_A, m - max(prev, i), lim_A),
                                ]),
                                mul,
                            )
                        prev = i

                        swaps = [a for a in (_active(eq_i, i, lim_q),
                                             _active(eq_ip1, i + 1, lim_q1))
                                 if a is not False]
                        multiplies = i < lim_a or i >= 2

                        if is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)
                        if len(swaps) == 1:
                            builder.cswap(swaps[0], q_i, reg_a[i])
                        elif swaps:
                            both = builder.alloc_qubits(1)[0]
                            builder.cx(km.array(swaps), both)
                            builder.cswap(both, q_i, reg_a[i])
                            builder.cx(km.array(swaps), both)
                            builder.free(both)
                        if not is_adjoint and multiplies:
                            builder.ixor_gf2_mul(q_i, reg_a[i], target=reg_b[i], field=gf,
                                                 control=mul)

                    endpoint(prev)
                    builder.free(mul)

        with builder.mark('multiply_and_add_or_sub'):
            q_i = builder.alloc_qubits(gf.degree)
            leading, trailing = (0, m - 1) if is_adjoint else (m - 1, 0)
            accumulate_quotient(leading, c3_hc if is_adjoint else c0_hc)
            builder.iadd(n_a, target=n_q)
            sweep_positions()
            builder.isub(n_a, target=n_q)
            accumulate_quotient(trailing, c0_hc if is_adjoint else c3_hc)
            builder.free(q_i)

    def _shift_b_array_right(self, builder: km.CircuitBuilder,
                             reg_b: Sequence[km.array], n_B: km.array, n_b: km.array,
                             flag, ctrl, it: int, *, is_adjoint: bool) -> None:
        """Shifts the active slice of `reg_b` by one slot."""
        with builder.mark('shift_b_array_right'):
            c0_hc, c1_hc, c2_hc, c3_hc = ctrl
            m = len(reg_b)
            lim_B = m - 2
            lim_b = min(m - 2, it // 2 + 1)

            builder.x(flag)
            nf = builder.alloc_qubits(2)
            builder.init_and(km.array([c0_hc, c3_hc]), flag, nf)
            builder.x(flag)
            c0_nf, c3_nf = nf

            ctrl_nB = builder.alloc_qubits(1)[0]
            builder.cx(km.array([c0_nf, c1_hc]), ctrl_nB)

            order = range(1, m) if is_adjoint else range(m - 1, 0, -1)
            with (
                builder.unary_iteration(n_B, control=ctrl_nB) as cur_B_ge,
                builder.unary_iteration(n_b, control=c2_hc) as cur_b_ge,
                builder.unary_iteration(n_B, control=c3_nf) as cur_B_lt,
            ):
                do_shift = builder.alloc_qubits(1)[0]

                def endpoint(i: int) -> None:
                    _xor_threshold(builder, cur_B_ge, lim_B, do_shift, ctrl_nB, m - i + 1,
                                   at_least=True)
                    _xor_threshold(builder, cur_b_ge, lim_b, do_shift, c2_hc, i,
                                   at_least=True)
                    _xor_threshold(builder, cur_B_lt, lim_B, do_shift, c3_nf, m - i,
                                   at_least=False)

                prev = None
                for i in order:
                    if prev is None:
                        endpoint(i)
                    else:
                        builder.cx(
                            km.array([
                                _active(cur_B_ge, m - max(prev, i) + 1, lim_B),
                                _active(cur_b_ge, min(prev, i), lim_b),
                                _active(cur_B_lt, m - max(prev, i), lim_B),
                            ]),
                            do_shift,
                        )
                    prev = i
                    builder.cswap(do_shift, reg_b[i - 1], reg_b[i])

                endpoint(prev)
                builder.free(do_shift)

            builder.cx(km.array([c0_nf, c1_hc]), ctrl_nB)
            builder.free(ctrl_nB)
            builder.x(flag)
            builder.del_and(km.array([c0_hc, c3_hc]), flag, nf)
            builder.x(flag)
            builder.free(nf)

    @staticmethod
    def _update_metadata(builder: km.CircuitBuilder, n_B: km.array, n_b: km.array,
                         n_q: km.array, flag, ctrl, *, is_adjoint: bool) -> None:
        """Updates the polynomial degree registers `(n_B, n_b, n_q)`."""
        with builder.mark('update_metadata'):
            c0_hc, c1_hc, c2_hc, c3_hc = ctrl
            builder.x(flag)
            nf = builder.alloc_qubits(2)
            builder.init_and(km.array([c0_hc, c3_hc]), flag, nf)
            builder.x(flag)
            c0_nf, c3_nf = nf

            sign = -1 if is_adjoint else 1
            updates = [
                [(n_q, 1, c0_hc), (n_B, -1, c0_nf)],
                [(n_B, -1, c1_hc)],
                [(n_b, 1, c2_hc)],
                [(n_b, 1, c3_nf), (n_q, -1, c3_hc)],
            ]
            for step in (reversed(updates) if is_adjoint else updates):
                for target, offset, control in (reversed(step) if is_adjoint else step):
                    builder.iadd(sign * offset, target=target, control=control)

            builder.x(flag)
            builder.del_and(km.array([c0_hc, c3_hc]), flag, nf)
            builder.x(flag)
            builder.free(nf)

    @staticmethod
    def _swap_a_and_b(builder: km.CircuitBuilder, reg_a: Sequence[km.array],
                      reg_b: Sequence[km.array], n_A: km.array, n_B: km.array,
                      n_a: km.array, n_b: km.array, flag, ctrl) -> None:
        """Swaps `(reg_a, n_A, n_a)` and `(reg_b, n_B, n_b)` at the end of a division round."""
        with builder.mark('swap_a_and_b'):
            c3_hc = ctrl[3]
            swap_ctrl = builder.alloc_qubits(1)[0]
            builder.init_and(c3_hc, flag, swap_ctrl)
            lhs = km.array([q for r in (n_A, n_a, *reg_a) for q in r])
            rhs = km.array([q for r in (n_B, n_b, *reg_b) for q in r])
            builder.cswap(swap_ctrl, lhs, rhs)
            builder.del_and(c3_hc, flag, swap_ctrl)
            builder.free(swap_ctrl)

    @staticmethod
    def _update_halting_counter(builder: km.CircuitBuilder, n_A: km.array,
                                n_B: km.array, n_b: km.array,
                                halting_counter: km.array, flag, is_c3, *,
                                is_adjoint: bool) -> None:
        """Increments `halting_counter` once the EEA stopping criterion holds."""
        with builder.mark('update_halting_counter'):
            work = builder.alloc_qubits(3)
            ctrl, busy, running = work
            builder.init_and(is_c3, flag, ctrl)
            builder.flip_if_less_than(n_b, n_B, target=busy, control=ctrl)
            builder.cx(busy, running)
            builder.flip_if_equal(n_A, 0, target=running, control=busy)

            builder.cx(running, ctrl)
            if is_adjoint:
                builder.isub(1, target=halting_counter, control=ctrl)
            else:
                builder.iadd(1, target=halting_counter, control=ctrl)
            builder.cx(running, ctrl)

            builder.flip_if_equal(n_A, 0, target=running, control=busy)
            builder.cx(busy, running)
            builder.flip_if_less_than(n_b, n_B, target=busy, control=ctrl)
            builder.del_and(is_c3, flag, ctrl)
            builder.free(work)

    def _pre_align_reg_b(self, builder: km.CircuitBuilder, reg_b: Sequence[km.array],
                         n_b: km.array, n_B: km.array, *, is_adjoint: bool) -> None:
        """Rotates the evaluator slice of `reg_b` by `n_B - n_b` into fixed indices."""
        omega = reg_b[len(reg_b) // 2 + 1:]
        for b in range(self.field.degree):
            lane = km.array([r[b] for r in omega])
            if is_adjoint:
                builder.right_rotate(lane, n_B)
                builder.left_rotate(lane, n_b)
            else:
                builder.right_rotate(lane, n_b)
                builder.left_rotate(lane, n_B)

    def _chien_search_and_forney_mbuc(self, builder: km.CircuitBuilder,
                                      reg_b: Sequence[km.array],
                                      error_mbuc: Sequence[km.array],
                                      n_b: km.array, n_B: km.array) -> None:
        """Evaluates `sigma` and `omega` via Chien search and kicks back the Forney MBUC phase."""
        gf = self.field
        degree = gf.degree
        m = len(reg_b)
        t = self.n_syndromes // 2
        block_length = (1 << degree) - 1
        primitive = gf.primitive_element

        with builder.mark('pre_align_reg_b'):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=False)

        sigma = reg_b[0]
        sigma_prime = reg_b[1]
        omega = reg_b[m // 2 + t]
        sigma_prime_inv = builder.alloc_qubits(degree)
        located = builder.alloc_qubits(degree)
        is_sigma_zero = builder.alloc_qubits(1)[0]
        is_error = builder.alloc_bits(1)

        # Fold the Forney prefactor `c_j ** -(m - 2)` (`c_j = primitive ** -j`) into
        # `omega`'s per-step multipliers so each position's polynomial evaluation is
        # a plain XOR sum followed by in-place constant multiplications.
        omega_indices = range(m // 2 + 1, m // 2 + t + 1)
        step_multipliers = [
            (k, gf.pow(primitive, k))
            for k in range(1, t + 1)
        ] + [
            (k, gf.pow(primitive, (m - 2 - k) % block_length))
            for k in omega_indices
            if (m - 2 - k) % block_length != 0
        ]

        sum_pairs = (
            [(reg_b[k], omega) for k in range(m // 2 + 1, m // 2 + t)]
            + [(reg_b[k], sigma_prime) for k in range(3, t + 1, 2)]
            + [(reg_b[k], sigma) for k in range(2, t + 1, 2)]
            + [(sigma_prime, sigma)]
        )
        fwd_src = km.array([q for src, _ in sum_pairs for q in src])
        fwd_dst = km.array([q for _, dst in sum_pairs for q in dst])
        rev_src = km.array([q for src, _ in reversed(sum_pairs) for q in src])
        rev_dst = km.array([q for _, dst in reversed(sum_pairs) for q in dst])

        num_positions = min(block_length, len(error_mbuc))
        forney_step = builder.mark('forney_step')
        for j in range(num_positions):
            with forney_step:
                builder.bit_store(is_error[0], True, control=error_mbuc[j])
                builder.push_condition(is_error[0])

                builder.cx(fwd_src, fwd_dst)
                builder.flip_if_equal(sigma, 0, target=is_sigma_zero)

                _, inverse_chain = builder.init_gf2_inverse_with_scaffold(
                    sigma_prime, target=sigma_prime_inv, field=gf)

                builder.init_and(is_sigma_zero, omega, located)
                builder.gf2_phase_by_product(located, sigma_prime_inv,
                                             mask=error_mbuc[j], field=gf)
                builder.del_and(is_sigma_zero, omega, located)

                builder.del_gf2_inverse_with_scaffold(
                    sigma_prime, target=sigma_prime_inv, scaffold=inverse_chain, field=gf)
                builder.free(inverse_chain)

                builder.flip_if_equal(sigma, 0, target=is_sigma_zero)
                builder.cx(rev_src, rev_dst)

                builder.pop_condition()
                builder.bit_store(is_error, False)

                if j + 1 < num_positions:
                    for k, step in step_multipliers:
                        builder.gf2_imul(step, target=reg_b[k], field=gf)

        if num_positions > 1:
            for k, step in step_multipliers:
                builder.gf2_idiv(
                    gf.pow(step, num_positions - 1), target=reg_b[k], field=gf)

        builder.free(is_error)
        builder.free(located + sigma_prime_inv + km.array([is_sigma_zero]))

        with builder.mark('unalign_reg_b'):
            self._pre_align_reg_b(builder, reg_b, n_b, n_B, is_adjoint=True)


def default_n_iters(n_syndromes: int) -> int:
    """Returns the EEA iteration count `3 * (n_syndromes + 1) + 9`."""
    return 3 * (n_syndromes + 1) + 9


def build_reference_circuit(m: int, n: int, degree: int) -> km.Circuit:
    """Builds the standalone reference Reed-Solomon MBUC decoder circuit."""
    builder = km.CircuitBuilder()
    syndromes = [builder.create_quantum_register(degree, name=f'S{i}') for i in range(n)]
    error_mbuc = [builder.create_classical_register(degree, name=f'E{j}') for j in range(m)]
    ProosAndZalkaDecoder(n, km.GF2Field(degree)).build_syndrome_decoding_mbuc(
        builder, syndromes=syndromes, error_mbuc=error_mbuc, n_iters=default_n_iters(n))
    return builder.finish_circuit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', help='Path of the .kmb (or .kmx) file to write.')
    parser.add_argument('--m', type=int, default=4095, help='Codeword length (default: 4095).')
    parser.add_argument('--n', type=int, default=70, help='Number of syndromes (default: 70).')
    parser.add_argument('--degree', type=int, default=12,
                        help='Field degree b for GF(2^b) (default: 12).')
    parser.add_argument('--format', choices=('kmb', 'kmx'), default='kmb',
                        help="Output circuit format: 'kmb' (binary, default) or 'kmx' (text).")
    args = parser.parse_args()

    circuit = build_reference_circuit(args.m, args.n, args.degree)
    circuit.to_file(args.output, format=args.format)
    toffolis = circuit.operation_statistics().num_toffoli_like
    print(f'Wrote {args.output} ({args.format}): {toffolis:,} Toffolis, {circuit.num_qubits} qubits.')


if __name__ == '__main__':
    main()
