# Surkhay's presentation and understanding guide

Your role: tell the audience what the evidence supports. Your allotted time in
the team plan is two minutes. These are contribution notes, not a finished slide
deck; D assembles the presentation. Numerical statements about the paper are
supported in `docs/c_paper_evidence.md` with original section/figure locators.

Use the [evidence slide draft](../slides/evidence-draft.md) for the two-slide outline,
speaking script, and comparison update after integration.

## Understand these concepts in your own words

**Semi-supervised:** we have a few labels and many unlabeled feature vectors.
**Transductive:** we use the feature vectors of the points we want to classify
when building the graph. Their unknown labels remain hidden.

**Local consistency:** close points tend to share a label.
**Global consistency:** a connected curved structure may belong to the same class
even when its endpoints are far apart in straight-line distance. This is an
assumption, not a fact guaranteed for every dataset.

**1-NN:** find the nearest revealed point and copy its label. With only two
revealed points it can cut across a moon because it does not trace the chain of
unlabeled points. It has no sigma or alpha.

**Graph:** points are vertices; similarity determines edge weights. Intermediate
unlabeled points can link distant parts of the same moon.

**Bandwidth sigma:** determines how fast similarity decreases with distance.
Too small can produce numerically isolated/weakly connected regions; too large
can link different classes strongly. The useful scale depends on the features.

**Alpha:** balances spreading from neighbors against retaining the original
label signal. Higher alpha gives relatively more weight to propagation. The
paper uses 0.99 in the experiments, but this is not a universal optimal value.

**Error versus accuracy:** 22.36% error means 77.64% accuracy, on the same scored
examples. We score only the unrevealed points to avoid counting labels supplied
to the model as prediction successes.

**Mean versus sample SD:** the mean is the average across ten runs. The SD
describes spread across those runs; 4.22 percentage points is not a 4.22% relative
change, not the standard error of the mean, and not a 95% confidence interval.

**Seed:** fixes pseudorandom choices so a run can be reproduced. It is not a
hyperparameter to tune. Our seeds change data noise and label placement.

**Fair comparison:** same points, same two revealed labels, same metric, no
tuning on hidden labels. LGC may see hidden points' coordinates because this
is the declared transductive protocol; seeing their labels would leak answers.

## Minimum method knowledge for whole-team Q&A

You don't need to lead the proof, but understand each symbol:

| Symbol | Meaning |
|---|---|
| W | Pairwise similarity weights; diagonal zero |
| D | Diagonal matrix of weighted degrees (row sums of W) |
| S = D^(-1/2) W D^(-1/2) | Symmetrically normalized similarity operator |
| Y | One-hot known labels; zero rows for unknown labels |
| F | One score per point per class; largest score gives the prediction |
| alpha | Neighbor propagation versus original-label signal |

Update: `F_next = alpha * S @ F + (1-alpha) * Y`.
Intuition: receive neighbors' current scores and reintroduce the original signal.
F scores are not automatically calibrated class probabilities. Labeled rows are
not hard-clamped at each step in this algorithm.

At convergence: `F_star = (1-alpha) * solve(I-alpha*S, Y)`.
For classification, dropping the common positive `(1-alpha)` factor leaves the
argmax unchanged. It does change score magnitudes, so B must account for it
when checking numerical equality. Prefer a linear solve over explicitly forming
the inverse. With normalized S and 0<alpha<1, the propagation is stable and has
a unique fixed point; B handles the proof and assumptions.

## Likely questions and short answers

**Why can unlabeled examples help?** They reveal geometric connections that a
classifier using only two labeled examples cannot observe from those two alone.

**Does unlabeled data always help?** No. If graph structure does not match class
membership, propagation can spread wrong labels across classes.

**Did your experiment reproduce the paper?** Only the two-moons idea using new
data. We have not replicated its exact samples or its USPS/text benchmarks.
At the current stage, only the baseline has actually run.

**Why one label per class?** It creates the sparse-label setting while ensuring
every class has a starting signal. It is also a favorable coverage assumption.

**Why repeat ten seeds?** Results can depend on where labels fall and on noise.
Ten runs provide a small descriptive robustness check, not a broad statistical
guarantee. The paper uses 100 trials for its real-data curves.

**How did you choose sigma?** We proposed a fixed initial scale before running
LGC, not a best value selected on evaluation labels. D will show a descriptive
sweep. If we later tune sigma, we need a separate validation protocol.

**How strong are the paper's comparisons?** They include both supervised and a
related semi-supervised comparator and average trials. But the text outcome is
mixed, the figure lacks displayed error bars, and practical tuning is unresolved.

**Why does harmonic SSL sometimes win on text?** The authors link this to CMN's
class-proportion estimates. At four labels, balanced sampling happens to match
the approximately balanced classes. Present that as the authors' interpretation.

**Can this predict a brand-new point?** This study classifies the points already
in the graph. Handling unseen points requires an extension or graph update;
that is outside our current experiment.

**What did you personally contribute?** "I own the experimental protocol,
baseline, repeated evaluation, and interpretation of the paper's evidence.
I used Codex to help implement and draft this work, and reviewed the evaluation
and explanations." Only claim review/understanding once you have actually done it.

## Reading and rehearsal sequence

1. Read the paper's introduction, Section 4, and Figure 5 alongside the evidence notes.
2. Explain why two labeled points can be insufficient for 1-NN and how graph
   structure could help. Draw the two moons yourself.
3. Read `experiments/README.md`, then follow `protocol.py` from data to masked
   labels to prediction to scoring. Run the baseline command yourself.
4. Explain the current result without saying anything about unmeasured LGC performance.
5. Learn the symbols and update rule at an intuitive level; ask B to check your
   explanation of convergence and A to walk you through the eventual core.
6. Rehearse your two-minute draft with the actual figures. Answer the questions
   above without reading. Meet D to check the transition into their critique.

Ready means you can explain each plot's axes, each reported number's source,
what labels the model saw, why the comparison is fair, and what remains untested.
