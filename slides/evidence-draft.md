# C’s evidence slides (2 min, speaker: Surkhay)

Draft content for D to assemble. See the [paper evidence notes](../docs/c_paper_evidence.md)
for sources and the [study guide](../docs/c_presentation_and_study.md) for Q&A preparation.

## Two-slide contribution

### Slide 1: Does the evidence support the method?

Visual: the paper's Figure 5 (both panels), attributed to Zhou et al., NIPS 2003,
Figure 5. Use the original figure from the paper when assembling slides; no
redrawn numerical curve or invented exact values.

Only three short bullets:

- USPS digits + newsgroup text; 100 trials, at least one label per class.
- Lower error than 1-NN/SVM in the reported curves; mixed versus harmonic SSL.
- Optimal-parameter comparisons leave label-scarce model selection unresolved.

Explain axes aloud, point to the relevant curves, and avoid reading every method
name. Use roughly 60-70 seconds.

### Slide 2: Our small reproduction - scope and status

Visual now: [Baseline prediction figure](../results/baseline/example.png).

- Generated two moons; 400 points; one revealed label per class; 10 seeds.
- 1-NN error: 22.36% mean, 4.22 percentage points sample SD.
- Core LGC comparison pending. This is baseline preparation, not a completed reproduction.

Once A's core is ready: replace the pending line with actual paired results and
use the comparison plot. Keep the exact configuration and scope qualification.
Don't present baseline-only work as the bonus result.

## Speaking draft usable at the current baseline-only stage

Approximately 240 words; rehearse aloud and adjust to your pace.

"To assess whether the method works, the authors test it on two moons,
handwritten digits, and newsgroup text. Here, the horizontal axis is the number
of labeled examples and the vertical axis is classification error, so lower is
better. The digit and text curves average one hundred trials, with at least one
labeled example from every class.

On digits, the consistency method performs substantially better than nearest
neighbor and SVM. The text results are more nuanced: the related harmonic method
does better at some label budgets. So the evidence supports exploiting unlabeled
structure, rather than claiming this algorithm always wins.

An important limitation is parameter selection. The authors compare algorithms
using their respective optimal parameters while acknowledging that tuning is
difficult with very few labels. These results therefore leave a practical
question: how would we choose good parameters with almost no validation labels?

For our small experiment, we have generated four hundred two-moons points and
revealed just one label per class. We evaluate the remaining three hundred and
ninety-eight points, repeating the experiment over ten fixed seeds.

Our nearest-neighbor baseline currently averages twenty-two point three six
percent error, with a sample standard deviation of four point two two percentage
points. The evaluation pipeline is ready, but the comparison with our core
algorithm is still pending. We therefore cannot yet claim an improvement.

This is a small qualitative reproduction of the idea, not a replication of the
paper's real-world benchmarks."

## Replace the last part after integration

Use an honest result paragraph, filled only from the actual summary:

"Using the same data and revealed labels for both methods, 1-NN had [X] percent
mean error and our consistency implementation had [Y] percent at sigma [S] and
alpha [A]. Across seeds, [describe spread and paired differences]. This
[supports/does not support] the expected advantage under this configuration.
Our generated toy data does not reproduce the paper's full real-data study."

If bonus work is not presentation-ready, the fallback is to use two minutes for
the paper's Figure 5 and its limitations. Don't let the optional experiment
displace the base assessment's paper understanding and critique.
