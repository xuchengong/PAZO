# PAZO: Private Zeroth-Order Optimization with Public Data

This is the implementation for the paper [Private Zeroth-Order Optimization with Public Data](https://openreview.net/pdf?id=zytITzY4IW) at NeurIPS 2025. 

This paper proposes leveraging public batch gradients to guide private zeroth-order optimization and introduces three <u>p</u>ublic-data-<u>a</u>ssisted <u>z</u>eroth-<u>o</u>rder optimizers (**PAZO**). PAZO achieves strong privacy/utility trade-offs while remaining computationally and memory efficient. We show that, under tight privacy budgets, private zeroth-order methods with public guidance can outperform first-order counterparts across both vision and language tasks, in pre-training and fine-tuning settings.

<p>
  <img src="assets/fig1.svg?raw=true" alt="Fig" width="100%"/>
  <em>
  Without public data, vanilla zeroth-order (ZO) underperforms first-order (FO); With public data, PAZO outperforms the best first-order with public data (FO+PUB), especially in highly private regimes.
  </em>
</p>

<p>
  <img src="assets/fig2.svg?raw=true" alt="Fig" width="100%"/>
  <em>
  Detailed comparison between PAZO-* and all the baselines.
  </em>
</p>


## Data and experiments
To generate (public, private) data pairs, please refer to the dataloader and the [data](data) folder.

To reproduce CIFAR-10/TinyImageNet/IMDB experiments, please refer to the [experiments](experiments) for implementations and [jobs](jobs) folder for scripts. For MNLI experiments, please refer to the [mnli](mnli) folder. Any file name with a `_vec` suffix indicates that it has a vectorized implementation.


## Citation

```bibtex
@inproceedings{gongprivate,
  title={Private Zeroth-Order Optimization with Public Data},
  author={Gong, Xuchen and Li, Tian},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```