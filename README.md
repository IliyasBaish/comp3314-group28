# COMP3314 · Group 28

**Paper:** Learning with Local and Global Consistency — Zhou, Bousquet, Lal, Weston & Schölkopf. NIPS 2003, vol. 16, pp. 321–328. [Read the paper](https://papers.nips.cc/paper_files/paper/2003/file/87682805257e619d49b8e0dfdc14affa-Paper.pdf).

**Goal:** Explain the method, evaluate its evidence, and provide our own critical insights. Optional bonus: implement a small partial reproduction from scratch.

## Deadlines

| Deliverable | Course deadline | Our target |
|---|---|---|
| Slides | **15 November 2026, 23:59** | Ready 14 November; submit 15 November |
| Report (~3–4 pages, LaTeX) | **29 November 2026, 23:59** | Final PDF 28 November; submit 29 November |

Times assumed to be Hong Kong time; verify against Moodle. Presentation: **9 minutes + 3 minutes Q&A**; all members attend. Presentation date: confirm separately.

## Who does what?

Replace B–D with names at our first meeting. Everyone reads the full paper, writes their section, prepares their slides, and learns the whole presentation for Q&A.

| Owner | Main work | Bonus work | Final coordination |
|---|---|---|---|
| **A — Iliyas** | Algorithm (Section 2): explain W, D, S, Y, F and the update | Core NumPy implementation | Repository and code integration |
| **B — TBD** | Theory (Section 3): regularization, convergence, course connections | Check iteration against a linear-system solution | LaTeX integration and equation checks |
| **C — TBD** | Evidence (Section 4): datasets, baselines, results, what claims they support | Toy data, 1-NN baseline, repeated evaluation | References and results consistency |
| **D — TBD** | Motivation, assumptions, strengths, limitations, extensions | Bandwidth sensitivity and plots | Slide assembly and rehearsal timing |

**Review:** A reviews B; B reviews C; C reviews D; D reviews A. Each author fixes their own section. Rebalance by estimated hours; integration work counts toward the workload.

## First five days: 10–14 November

| Date | Each member's task | Team checkpoint |
|---|---|---|
| **10 Nov** | Read paper; claim role; outline assigned section | Agree main argument, course connections, experiment scope and code interfaces |
| **11 Nov** | Draft assigned slides; build assigned bonus component | Explain method to one another; first toy run if feasible |
| **12 Nov** | Review another member's work; validate evidence and experiments | Complete slide draft; freeze experiment scope |
| **13 Nov** | Refine visuals, citations and critique | First timed rehearsal; fix unclear explanations |
| **14 Nov** | Final checks and Q&A practice | Second rehearsal; freeze submission-ready slides |

**15 Nov:** A submits slides; B checks the uploaded file and submission receipt. Daily check-in: 15 minutes (done / next / blocked). If bonus work is not ready on 12 Nov, prioritize the presentation; continue it afterward for the report.

## Report phase: 16–29 November

- **16–20 Nov:** Each person expands their section into LaTeX; finish the agreed small experiment.
- **21–24 Nov:** B integrates a complete draft; everyone reviews content and page budget.
- **25–27 Nov:** Fix gaps in analysis; verify citations, figures and reproducibility.
- **28 Nov:** Freeze final PDF and code; all four approve the deliverables.
- **29 Nov:** A submits; B verifies upload and receipt before 23:59.

Suggested report space: introduction 0.4 page; method/theory 1.1; original evidence 0.6; critique 0.9; reproduction 0.6; conclusion/references 0.4. Adjust to the instructor's page-count rules.

## Presentation plan (9 minutes)

| Speaker | Content | Time |
|---|---|---|
| D | Problem and motivation | 1:30 |
| A | Algorithm and visual example | 2:15 |
| B | Theory and course connections | 2:00 |
| C | Paper evidence and our results, if ready | 2:00 |
| D | Critical insight and conclusion | 1:15 |

Target ~8 slides. Clearly label paper figures versus our results. Everyone prepares for questions about assumptions, graph construction, parameters, limitations and experimental fairness.

## Optional reproduction

**Scope:** small two-moons dataset; one revealed label per class; our NumPy implementation versus 1-NN; 10 fixed seeds; one bandwidth-sensitivity plot. Evaluate only unrevealed labels. Do not use evaluation labels to tune the model. Newly generated data means a **partial/qualitative reproduction**, not exact replication.

Prioritize understanding (40%) and critique (30%); report is 10%, presentation/Q&A 20%. Bonus is up to 10 additional percentage points.

**Current status:** planning scaffold only; no implementation or results yet. Before bonus submission, add exact setup/run commands, dependency versions, seed/configuration details, results and discrepancies here. Credit any reused code or data. The submitted bonus repository must be **public**.

## GitHub workflow

1. Create a task issue: owner, reviewer, due date, and completion checklist.
2. Use a task branch, e.g. `method/core` or `slides/theory`.
3. Open a pull request; link the issue with `Closes #NUMBER`.
4. One teammate reviews before merging into `main`. Merge small changes regularly.
5. Board columns: **Todo → In progress → In review → Done**.

Done = checked, cited/documented, reviewed, and merged. Avoid editing the same slide/notebook simultaneously; D assembles slide contributions and B integrates report sections.

| Folder | Contents |
|---|---|
| `src/` | Core algorithm |
| `experiments/` | Data, baselines and experiment scripts |
| `results/` | Small result tables and generated plots |
| `report/` | LaTeX source and bibliography |
| `slides/` | Presentation source and submitted version |

Do not commit passwords, virtual environments, downloaded datasets or uncredited third-party materials.
