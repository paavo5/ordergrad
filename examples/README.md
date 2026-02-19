# Examples playground

These scripts are intended as a hands-on playground for understanding and profiling `ordergrad` behavior.

## 1) Plot order weights

`plot_order_weights.py` plots unconditional weight curves `W[m,j]` over sorted rank `m`.

```bash
python examples/plot_order_weights.py --N 120 --k 20 --ranks 1,5,10,15,20
```

## 2) Benchmark methods

`benchmark_methods.py` compares efficient vs matmul paths and dense vs non-dense precompute costs.

```bash
python examples/benchmark_methods.py --N 500 --k 40 --repeats 100
```

## 3) Monte Carlo accuracy curve

`monte_carlo_accuracy.py` compares MC estimates to exact known-`(r,p)` values and plots estimation error vs sample count.

```bash
python examples/monte_carlo_accuracy.py --k 6 --trials 30 --t-grid 100,300,1000,3000,10000
```

---

## Notes

- The plotting scripts require `matplotlib`.
- Outputs are saved under `examples/artifacts/` by default.
