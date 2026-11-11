# C: evidence from the paper

Owner: Surkhay. Read this alongside Section 4 and Figure 5, not as a substitute
for reading the full paper. Paper facts below were checked against the original
eight-page PDF, including visual inspection of Figure 5 on printed page 327.

Source: Zhou, Bousquet, Lal, Weston, and Scholkopf, *Learning with Local and Global
Consistency*, NIPS 2003, vol. 16, pp. 321-328.
[Original paper](https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).
Page locators below refer to PDF pages 1-8; Figure 5 is on PDF page 7.

## The claim being tested

With very few labeled examples, the geometry of the unlabeled examples can help
classify them. The method assumes labels vary smoothly along the data's graph
structure. Unlabeled data supplies coordinates/relationships, not free ground-truth
labels. The paper's question is whether exploiting those relationships helps.

This is transductive evaluation: the algorithm can use the feature vectors of
the points it will predict. It cannot use their withheld labels. Section 1
defines performance as error on unlabeled points only. This differs from testing
on entirely new feature vectors that were absent during graph construction.

## What the authors actually tested

| Study | Data and setup | Main purpose |
|---|---|---|
| Toy problem, Section 4.1 | Two interlocking moons with two labeled points; Figures 2-3 illustrate iteration and smoothness | Show how labels diffuse along a curved structure |
| Digit recognition, Section 4.2 | USPS digits 1-4, 16x16 images; 3,874 examples; label budgets shown from 4 to 50 | Quantitative evidence on image data |
| Text classification, Section 4.3 | Four `rec` topics from 20-news-18828: autos, motorcycles, baseball, hockey; 3,970 documents, 8,014 features | Test a different domain and similarity measure |

The toy study also shows smoothing an SVM's output (Figure 4). That is a separate
demonstration; our small experiment does not implement this extension.

Digit class counts are 1,269 / 929 / 824 / 852. Text preprocessing uses Porter
stemming, SMART stop-word removal, header removal, discarding words appearing
in at most five documents, removal of empty documents, and TF-IDF representation.
Text distance is cosine dissimilarity, `1 - cosine_similarity`, rather than raw
Euclidean distance between document vectors.

## Comparators and settings

Section 4 compares the symmetric-normalized consistency method against:

- 1-nearest neighbor (the paper sets k=1).
- One-versus-rest SVMs with an RBF kernel.
- A related semi-supervised harmonic Gaussian field method with class mass
  normalization (CMN).
- Two consistency variants using different graph operators.

The real-data error curves average 100 trials. Labeled samples contain at least
one example from each class. The algorithm and variants use alpha=0.99.

| Task | SVM RBF width | Harmonic / consistency RBF width |
|---|---|---|
| Digits | 5 | 1.25 |
| Text | 1.5 | 0.15 |

Widths belong to different feature spaces. The text width 0.15 is not a verified
toy-data setting. The zero diagonal of W is part of the consistency construction.

Important qualification from the start of Section 4 (PDF page 4): the authors
acknowledge the difficulty of model selection with very few labeled points and
use each algorithm's respective optimal parameters, apart from fixing alpha.
The paper does not give a deployable label-scarce tuning protocol there. Do not
claim it proves the method is easy to tune, or assert a specific tuning-label
leakage procedure that the paper does not document.

## How to explain Figure 5

- Left panel: USPS digits. Right panel: text classification.
- Horizontal axis: total number of revealed labels, from 4 to 50.
- Vertical axis: error rate on unrevealed examples. Lower is better.
- Each curve is a method, averaged over 100 trials.
- A point at four labels means one label for each of four classes, not four
  labeled examples per class.

Digits: the main consistency method has a much lower curve than 1-NN and SVM
and is lower than the harmonic comparator in the reported experiment. One of
the variants also performs well; another variant performs substantially worse.
This supports the usefulness of unlabeled structure and shows the normalization
choice matters. Do not say all consistency variants perform equally well.

Text: consistency has lower error than 1-NN and SVM in the displayed results,
but the harmonic method is better at four labels and again at larger label
budgets. The authors attribute its unusual behavior at small budgets to the
class-proportion estimates used by CMN. Present this mechanism as the authors'
explanation, not something our toy experiment has established.

Do not invent exact percentages by eyeballing the curves. The paper explicitly
reports about 1% error with 30 labels after adding a one-pixel translation
jittered kernel to the digit task. That is an additional invariance-aware
configuration, not the ordinary consistency curve or our toy result.

## Our interpretation: strengths and limitations

These points are our analysis of the design, not additional author experiments.

1. **More convincing than a toy alone:** real image and text tasks accompany the
   visual example; the harmonic comparator tests against a related SSL method,
   rather than only supervised baselines.
2. **No universal winner:** the text comparison prevents a blanket claim of
   dominance. Distinguish "helps here" from "always beats alternatives."
3. **Tuning remains unresolved:** optimal-parameter comparisons show capability
   under favorable settings; they do not establish how to find those settings
   when almost no validation labels are available.
4. **Sampling is favorable:** at least one revealed label per class is ensured.
   Performance with a missing class or naturally biased label acquisition is
   not settled by these experiments.
5. **Uncertainty is incompletely displayed:** averages over 100 trials reduce
   sensitivity to a single split, but Figure 5 has no displayed error bars.
   The graph alone cannot establish the variability or significance of small gaps.
6. **Graph quality is an assumption:** similarity must reflect class structure.
   Shortcuts between classes, a poor metric, or inappropriate bandwidth can
   spread mistakes. This motivates D's sensitivity analysis.
7. **Benchmark breadth is limited:** two real datasets do not establish
   performance across arbitrary data distributions or modern large-scale tasks.
8. **Compute matters:** a dense n-by-n graph has quadratic storage cost; a
   straightforward dense linear solve has cubic time cost. These are complexity
   observations about a naive implementation, not runtime measurements in the paper.

Our strongest presentation takeaway: the evidence supports graph-based learning
when similarity and labels align, but practical performance depends on graph
construction, label coverage, and parameter choice.

## Keep paper evidence and our evidence separate

| Paper | Our current work |
|---|---|
| Original two-moons illustration and two real-data benchmarks | Newly generated 400-point toy datasets only |
| 100 trials for the digit and text curves | 10 fixed seeds for our paired protocol |
| Multiple supervised and semi-supervised comparators | Paired comparison of Euclidean 1-NN and our NumPy LGC |
| Author-reported outcomes | Measured two-moons results in `results/comparison/` |

On the 398 unrevealed points per trial, LGC averaged 0.45% error (sample SD
0.54 percentage points) at σ = 0.15 and α = 0.99. The 1-NN baseline averaged
22.36% (sample SD 4.22 percentage points). The bandwidth sweep includes a
setting where LGC does worse than 1-NN. Both results describe the generated
moons and the stated label budget. [Full comparison](../results/comparison/README.md)
