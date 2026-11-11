# C’s evidence slides (2 min, speaker: Surkhay)

## Slide 1 — What the paper’s experiments show (about 60 seconds)

**Visual:** Both panels of [Zhou et al., Figure 5](https://proceedings.neurips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf#page=7). Label the panels USPS digits and newsgroup text and credit Zhou et al., NIPS 2003.

- The digit and text curves average 100 trials with at least one label per class.
- LGC beats 1-NN and RBF SVM on digits. The comparison with harmonic label propagation varies across text label budgets.
- The authors compare methods using their respective optimal parameters, leaving tuning with very few labels unresolved.

**Say:** “The authors illustrate the method on two moons, then test it on handwritten digits and newsgroup text. Figure 5 shows the number of revealed labels along the bottom and the error on unrevealed points on the vertical axis. Lower is better. Each point averages a hundred trials, with at least one label per class.

“On digits, the consistency method has substantially lower error than nearest neighbor and SVM. On text, the related harmonic method does better at some label budgets. The paper also compares methods at their respective optimal parameters, so choosing those parameters with very few labels remains an open practical question.”

## Slide 2 — Our paired two-moons experiment (about 60 seconds)

**Visual:** [Seed 0 comparison](../results/comparison/example.png). Stars mark the two revealed labels. The numbers below summarize all ten seeds.

- Ten trials, 400 points per trial, one revealed label per class.
- Mean error on the 398 unrevealed points per trial was **22.36% for 1-NN** and **0.45% for LGC** at σ = 0.15 and α = 0.99.
- LGC error rose to **32.59%** at σ = 0.6 in the bandwidth sweep.

**Say:** “We generated ten two-moons datasets of four hundred points each and revealed one label per moon. Both methods used the same two labels in each trial. On this example, nearest neighbor cuts across the moons, while LGC can spread the label signal along the arcs using the unrevealed points.

“Across all ten trials, nearest neighbor averaged 22.36 percent error. At our fixed bandwidth, LGC averaged 0.45 percent. At a much wider bandwidth, its error climbed to 32.59 percent. Wider affinities can connect points across the moons and blur the class boundary. That sensitivity matters when only a few labels are available to guide the choice.”

For the full numbers and limitations, see the [comparison results](../results/comparison/README.md) and the [study guide](../docs/c_presentation_and_study.md).
