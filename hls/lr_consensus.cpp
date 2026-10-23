#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include "lr_weights.h"

typedef ap_axis<32, 0, 0, 0> axis_t;

// One snapshot in: 4 relays x 9 features, 36 int16 values sign-extended into
// 32-bit AXI-Stream words. One bit out: the OR consensus across the 4 relays.
void lr_consensus(hls::stream<axis_t> &in, hls::stream<axis_t> &out) {
#pragma HLS INTERFACE axis port=in
#pragma HLS INTERFACE axis port=out
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL

    ap_int<16> x[N_RELAYS][N_FEATURES];
#pragma HLS ARRAY_PARTITION variable=x complete dim=0

READ:
    for (int r = 0; r < N_RELAYS; r++) {
        for (int i = 0; i < N_FEATURES; i++) {
#pragma HLS PIPELINE II=1
            axis_t v = in.read();
            x[r][i] = (ap_int<16>)(int)v.data;   // truncate to the 16-bit value
        }
    }

    ap_uint<1> attack = 0;

RELAYS:
    for (int r = 0; r < N_RELAYS; r++) {
#pragma HLS UNROLL
        ap_int<40> acc = BIAS;
    MAC:
        for (int i = 0; i < N_FEATURES; i++) {
#pragma HLS UNROLL
            acc += (ap_int<40>)x[r][i] * (ap_int<40>)W[i];
        }
        attack |= (acc >= 0) ? 1 : 0;          // the OR gate, in one line
    }

    axis_t o;
    o.data = attack;
    o.keep = -1;
    o.strb = -1;
    o.user = 0;
    o.id   = 0;
    o.dest = 0;
    o.last = 1;                                 // DMA needs TLAST to end the transfer
    out.write(o);
}