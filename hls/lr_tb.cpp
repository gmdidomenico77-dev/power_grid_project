// Vectors generated from training files only (see notebooks/model_exports.ipynb).
// Expected outputs come from the fixed-point Python reference, not the float
// model, so a correct kernel matches exactly. The held-out files are reserved
// for the final hardware replay.


#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <cstdio>
#include "lr_weights.h"

typedef ap_axis<32, 0, 0, 0> axis_t;

void lr_consensus(hls::stream<axis_t> &in, hls::stream<axis_t> &out);

int main() {
    FILE *f = fopen("test_vectors.txt", "r");
    if (!f) { printf("ERROR: cannot open test_vectors.txt\n"); return 1; }

    int n;
    if (fscanf(f, "%d", &n) != 1) { printf("ERROR: bad header\n"); return 1; }
    printf("loaded %d snapshots\n", n);

    const int WORDS = N_RELAYS * N_FEATURES;
    int fails = 0;

    for (int s = 0; s < n; s++) {
        hls::stream<axis_t> in("in"), out("out");

        for (int k = 0; k < WORDS; k++) {
            int val;
            if (fscanf(f, "%d", &val) != 1) { printf("ERROR: truncated file\n"); return 1; }
            axis_t w;
            w.data = val;
            w.keep = -1; w.strb = -1;
            w.last = (k == WORDS - 1);
            in.write(w);
        }

        int expected;
        fscanf(f, "%d", &expected);

        lr_consensus(in, out);

        int got = (int)out.read().data;
        if (got != expected) {
            printf("MISMATCH snapshot %d: got %d, expected %d\n", s, got, expected);
            fails++;
        }
    }
    fclose(f);

    printf("\n%d / %d matched\n", n - fails, n);
    if (fails) { printf("C SIMULATION FAILED\n"); return 1; }
    printf("C SIMULATION PASSED\n");
    return 0;
}