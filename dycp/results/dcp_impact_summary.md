# DCP Impact Summary

Source: `/vllm/dycp/results/*/*.json`

All runs completed without failed requests. Deltas are computed as:

`(DCP enabled - DCP disabled) / DCP disabled * 100%`

Positive throughput deltas are better. Negative latency deltas are better.

## Key takeaways

- `dp16 @ 16 QPS`: DCP has little throughput impact (+1.4%) because the load is request-rate limited, but it substantially improves generation latency: mean TPOT -29.6%, p50 TPOT -31.8%, p90 TPOT -26.3%.
- `dp16 @ 32 QPS`: DCP is a large win for sustained throughput: request/output/total-token throughput all +161.8%. Mean TPOT improves -78.5%. TTFT is mixed: p50/mean TTFT are worse with DCP, but p90/p99 TTFT are much better because the non-DCP run is severely overloaded.
- `dp4tp4 @ 16 QPS`: DCP has effectively no performance benefit. Throughput changes by only +0.1%, and latency is mostly flat.
- `dp4tp4 @ 32 QPS`: DCP reduces throughput by -7.4%. It slightly improves mean/p50/p90 TPOT and TTFT, but worsens p99 tail latency.

## DCP impact vs disabled

| Topology | Target QPS | Req throughput | Output tok/s | Total tok/s | Mean TTFT | P50 TTFT | P90 TTFT | P99 TTFT | Mean TPOT | P50 TPOT | P90 TPOT | P99 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp16 | 16 | +1.4% | +1.4% | +1.4% | -27.2% | -34.6% | +32.3% | -56.2% | -29.6% | -31.8% | -26.3% | +0.7% |
| dp16 | 32 | +161.8% | +161.8% | +161.8% | +57.4% | +1464.0% | -55.5% | -79.7% | -78.5% | -81.5% | -80.9% | -80.8% |
| dp4tp4 | 16 | +0.1% | +0.1% | +0.1% | +1.8% | -0.7% | +2.0% | +18.1% | +0.2% | +0.6% | +0.5% | -0.3% |
| dp4tp4 | 32 | -7.4% | -7.4% | -7.4% | -2.8% | -5.5% | -3.7% | +12.5% | -1.3% | -1.1% | -3.1% | +12.2% |

## Raw metrics

| Topology | DCP | Target QPS | Duration s | Completed | Failed | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | P50 TTFT ms | P90 TTFT ms | P99 TTFT ms | Mean TPOT ms | P50 TPOT ms | P90 TPOT ms | P99 TPOT ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dp16 | off | 16 | 312.40 | 4800 | 0 | 15.36 | 3933.36 | 143444.61 | 193.46 | 129.68 | 152.62 | 2237.95 | 46.31 | 46.63 | 47.25 | 49.85 |
| dp16 | on | 16 | 308.09 | 4800 | 0 | 15.58 | 3988.44 | 145453.35 | 140.80 | 84.86 | 201.90 | 980.17 | 32.61 | 31.80 | 34.81 | 50.18 |
| dp16 | off | 32 | 1039.33 | 9600 | 0 | 9.24 | 2364.59 | 86233.77 | 5919.05 | 643.84 | 24099.76 | 55023.79 | 184.94 | 214.41 | 218.29 | 225.69 |
| dp16 | on | 32 | 397.03 | 9600 | 0 | 24.18 | 6190.02 | 225742.15 | 9316.52 | 10069.61 | 10736.03 | 11193.23 | 39.79 | 39.73 | 41.79 | 43.44 |
| dp4tp4 | off | 16 | 307.53 | 4800 | 0 | 15.61 | 3995.74 | 145719.71 | 82.91 | 78.43 | 93.49 | 258.91 | 28.09 | 27.44 | 30.89 | 32.18 |
| dp4tp4 | on | 16 | 307.14 | 4800 | 0 | 15.63 | 4000.73 | 145901.52 | 84.37 | 77.86 | 95.31 | 305.73 | 28.13 | 27.59 | 31.04 | 32.07 |
| dp4tp4 | off | 32 | 338.09 | 9600 | 0 | 28.40 | 7269.12 | 265095.84 | 6433.35 | 8437.07 | 9235.74 | 9813.73 | 34.28 | 34.14 | 36.42 | 38.36 |
| dp4tp4 | on | 32 | 365.18 | 9600 | 0 | 26.29 | 6729.85 | 245429.05 | 6254.38 | 7971.65 | 8889.63 | 11043.12 | 33.83 | 33.75 | 35.30 | 43.05 |

## Interpretation notes

- The `dp16 @ 32 QPS` non-DCP run only reaches 9.24 req/s, so it is overloaded. DCP does not reach the full 32 req/s either, but it improves sustained throughput to 24.18 req/s and reduces per-token generation latency sharply.
- The `dp4tp4` results do not show a clear DCP benefit in this sample. At 16 QPS it is flat; at 32 QPS it loses throughput.
- These are single-run results. Small deltas, especially around 1-2%, should be treated as noise until repeated.
