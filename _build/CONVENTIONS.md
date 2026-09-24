# Authoring conventions (for build scripts)

Course: "Machine Learning — Hands-on Course", university level (upper-undergrad / first-year grad).
Each lecture = ONE notebook in notebooks/, produced by a build script _build/build_NN.py
that imports `from nbtools import md, code, build` (run from the _build dir: `cd _build && python3 build_NN.py`).
`build()` executes the notebook and keeps outputs, so every code cell MUST run cleanly, offline, on CPU,
and the whole notebook should run in < ~3 minutes.

## Notebook structure (every lecture)
1. Title cell: `# Lecture NN — Title`, then a short "**Learning objectives**" bullet list,
   "**Prerequisites**" (links to earlier notebooks by filename), and a "Contents" list.
2. Setup code cell: imports, `np.random.seed(42)` / `rng = np.random.default_rng(42)`, `%matplotlib inline`, 
   `plt.rcParams["figure.figsize"] = (7, 4.5)`.
3. Alternating sections: THEORY (markdown with LaTeX math: `$...$` inline, `$$...$$` display; derivations,
   intuition, assumptions, complexity) followed by HANDS-ON code. Pattern for algorithms:
   (a) derive it, (b) implement FROM SCRATCH in NumPy with clear comments, (c) verify against scikit-learn
   (or PyTorch) on the same data, (d) visualize (matplotlib; decision boundaries, loss curves, etc.).
4. "Pitfalls & practical tips" markdown section.
5. "Exercises" section: 4–6 graded exercises (mix of pen-and-paper math and coding), each with a
   `# TODO` starter code cell where appropriate (starter cells must still execute without error — e.g. 
   define a stub that returns None and don't call it in a way that crashes, or leave the body as `pass`).
6. "Summary" (key takeaways bullets) and "Further reading" (textbook chapters: ISLR/ISLP, ESL, Bishop PRML,
   Murphy PML, Goodfellow DL, Géron; plus landmark papers where relevant).

## Rules
- Depth: university-level. Real math (derivations of gradients, closed forms, probabilistic views),
  not just API calls. But explain intuition too. Aim ~35–60 cells per notebook.
- Data: ONLY offline data — sklearn bundled datasets (load_iris, load_breast_cancer, load_digits,
  load_wine, load_diabetes) or generated data (make_classification, make_moons, make_blobs, make_regression,
  numpy synthetic). NEVER fetch_* or downloads (no network).
- Libraries available: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, torch (CPU). Nothing else.
- Keep outputs small: no printing huge arrays; limit figures to what's instructive (~6–12 per notebook).
- Suppress noisy warnings only if needed (`warnings.filterwarnings("ignore", category=ConvergenceWarning)`).
- Markdown: use headings `##` for sections, `###` for subsections, numbered like `## 3. Gradient Descent`.
- English, clear academic tone. No emojis.
