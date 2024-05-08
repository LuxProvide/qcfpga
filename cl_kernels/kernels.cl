#include <pyopencl-complex.h>

/*
 * Returns the nth number where a given digit
 * is cleared in the binary representation of the number
 */
static int nth_cleared(int n, int target)
{
    int mask = (1 << target) - 1;
    int not_mask = ~mask;

    return (n & mask) | ((n & not_mask) << 1);
}

///////////////////////////////////////////////
// KERNELS
///////////////////////////////////////////////

/*
 * Applies a single qubit gate to the register.
 * The gate matrix must be given in the form:
 *
 *  A B
 *  C D
 */
__kernel void apply_gate(
    __global cfloat_t *amplitudes,
    int target,
    cfloat_t A,
    cfloat_t B,
    cfloat_t C,
    cfloat_t D)
{
    int const global_id = get_global_id(0);

    int const zero_state = nth_cleared(global_id, target);

    // int const zero_state = state & (~(1 << target)); // Could just be state
    int const one_state = zero_state | (1 << target);

    cfloat_t const zero_amp = amplitudes[zero_state];
    cfloat_t const one_amp = amplitudes[one_state];

    amplitudes[zero_state] = cfloat_add(cfloat_mul(A, zero_amp), cfloat_mul(B, one_amp));
    amplitudes[one_state] = cfloat_add(cfloat_mul(D, one_amp), cfloat_mul(C, zero_amp));
}

/*
 * Applies a controlled single qubit gate to the register.
 */
__kernel void apply_controlled_gate(
    __global cfloat_t *amplitudes,
    int control,
    int target,
    cfloat_t A,
    cfloat_t B,
    cfloat_t C,
    cfloat_t D)
{
    int const global_id = get_global_id(0);
    int const zero_state = nth_cleared(global_id, target);
    int const one_state = zero_state | (1 << target); // Set the target bit

    int const control_val_zero = (((1 << control) & zero_state) > 0) ? 1 : 0;
    int const control_val_one = (((1 << control) & one_state) > 0) ? 1 : 0;

    cfloat_t const zero_amp = amplitudes[zero_state];
    cfloat_t const one_amp = amplitudes[one_state];

    if (control_val_zero == 1)
    {
        amplitudes[zero_state] = cfloat_add(cfloat_mul(A, zero_amp), cfloat_mul(B, one_amp));
    }

    if (control_val_one == 1)
    {
        amplitudes[one_state] = cfloat_add(cfloat_mul(D, one_amp), cfloat_mul(C, zero_amp));
    }
}

/*
 * Applies a controlled-controlled single qubit gate to the register.
 */
__kernel void apply_controlled_controlled_gate(
    __global cfloat_t *amplitudes,
    int control,
    int control_2,
    int target,
    cfloat_t A,
    cfloat_t B,
    cfloat_t C,
    cfloat_t D)
{
    int const global_id = get_global_id(0);
    int const zero_state = nth_cleared(global_id, target);
    int const one_state = zero_state | (1 << target); // Set the target bit

    int const control_val_zero = (((1 << control) & zero_state) > 0) ? 1 : 0;
    int const control_val_one = (((1 << control) & one_state) > 0) ? 1 : 0;
    int const control_val_two_zero = (((1 << control_2) & zero_state) > 0) ? 1 : 0;
    int const control_val_two_one = (((1 << control_2) & one_state) > 0) ? 1 : 0;

    cfloat_t const zero_amp = amplitudes[zero_state];
    cfloat_t const one_amp = amplitudes[one_state];

    if (control_val_zero == 1 && control_val_two_zero == 1)
    {
        amplitudes[zero_state] = cfloat_add(cfloat_mul(A, zero_amp), cfloat_mul(B, one_amp));
    }

    if (control_val_one == 1 && control_val_two_one == 1)
    {
        amplitudes[one_state] = cfloat_add(cfloat_mul(D, one_amp), cfloat_mul(C, zero_amp));
    }
}

/*
 * Swaps the states of two qubits in the register
 * NOT MIGRATED
 */
// __kernel void swap(
//     __global cfloat_t *const amplitudes,
//     __global cfloat_t *amps,
//     int first_qubit,
//     int second_qubit)
// {
//     int const state = get_global_id(0);

//     int const first_bit_mask = 1 << first_qubit;
//     int const second_bit_mask = 1 << second_qubit;

//     int const new_second_bit = ((state & first_bit_mask) >> first_qubit) << second_qubit;
//     int const new_first_bit = ((state & second_bit_mask) >> second_qubit) << first_qubit;

//     int const new_state = (state & !first_bit_mask & !second_bit_mask) | new_first_bit | new_second_bit;

//     amps[new_state] = amplitudes[state];
// }


/**
 * Get a single amplitude
 */
__kernel void get_single_amplitude(
    __global cfloat_t *const amplitudes,
    __global cfloat_t *out,
    int i)
{
    out[0] = amplitudes[i];
}

/**
 * Calculates The Probabilities Of A State Vector
 */
__kernel void calculate_probabilities(
    __global cfloat_t *const amplitudes,
    __global float *probabilities)
{
    int const state = get_global_id(0);
    cfloat_t amp = amplitudes[state];

    probabilities[state] = cfloat_abs(cfloat_mul(amp, amp));
}

/**
 * Initializes a register to the value 1|0..100...0>
 *                                          ^ target
 */
__kernel void initialize_register(
    __global cfloat_t *amplitudes,
    int const target)
{
    int const state = get_global_id(0);
    if (state == target)
    {
        amplitudes[state] = cfloat_new(1, 0);
    }
    else
    {
        amplitudes[state] = cfloat_new(0, 0);
    }
}

/**
 * Collapses a qubit in the register
 */
__kernel void collapse(
    __global cfloat_t *amplitudes, 
    int const target,
    int const outcome, 
    float const norm)
{
    int const state = get_global_id(0);

    if (((state >> target) & 1) == outcome) {
        amplitudes[state] = cfloat_mul(amplitudes[state], cfloat_new(norm, 0.0));
    }
    else
    {
        amplitudes[state] = cfloat_new(0.0, 0.0);
    }
}
