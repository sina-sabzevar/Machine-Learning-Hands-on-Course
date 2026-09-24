# Syllabus — Machine Learning (Hands-on)

**Level:** upper-undergraduate / first-year graduate. **Credits:** 3–4 (2 × 75 min lectures + 1 lab per week).

**Prerequisites:** programming in Python; linear algebra (matrices, eigenvalues); multivariable calculus
(gradients, chain rule); probability and statistics (random variables, expectation, Gaussian distribution).
Lecture 01 reviews all of these.

## Course learning outcomes

By the end of the course, students will be able to:

1. Formulate real problems as supervised or unsupervised learning tasks, and choose suitable losses and metrics.
2. Derive the core algorithms (linear/logistic regression, SVMs, trees, EM, PCA, backpropagation) and implement them from scratch.
3. Evaluate models rigorously with cross-validation, avoid data leakage, and diagnose bias versus variance.
4. Use scikit-learn and PyTorch effectively to build complete, reproducible ML pipelines.
5. Interpret models and assess their fairness, and communicate results responsibly.

## Weekly schedule

| Week | Lecture | Lab / Deliverable |
|------|---------|-------------------|
| 1 | 00 — Introduction & Python toolkit | Lab 0: environment setup, NumPy drills |
| 2 | 01 — Mathematical foundations | Problem Set 1 (math) |
| 3 | 02 — Linear regression & regularization | Lab: GD vs closed form |
| 4 | 03 — Logistic regression & classification | Problem Set 2 |
| 5 | 04 — Model evaluation & selection | Lab: pipelines & CV |
| 6 | 05 — k-NN & Naive Bayes | Problem Set 3 |
| 7 | 06 — Support vector machines | **Midterm exam** (Lectures 00–05) |
| 8 | 07 — Decision trees & ensembles | Problem Set 4 |
| 9 | 08 — Clustering | Lab: EM from scratch |
| 10 | 09 — Dimensionality reduction | Problem Set 5 · capstone proposal due |
| 11 | 10 — Neural networks from scratch | Lab: backprop & gradient checking |
| 12 | 11 — Deep learning with PyTorch | Problem Set 6 |
| 13 | 12 — Capstone: end-to-end ML & responsible ML | Capstone work sessions |
| 14 | Project presentations | **Capstone report due** |

## Assessment

| Component | Weight |
|-----------|--------|
| Problem sets (the exercises at the end of each notebook; best 6 count) | 30% |
| Labs / participation | 10% |
| Midterm exam | 25% |
| Capstone project (proposal 5%, report + code 20%, presentation 10%) | 35% |

## Policies

- **Collaboration:** you may discuss ideas with classmates, but code and write-ups must be your own.
- **AI tools:** allowed for explaining concepts and debugging. Say so in your submission whenever you used them.
  You must be able to explain every line you submit.
- **Late work:** 10% deducted per day, up to 3 days.

## Reading list

| Lecture | Reading |
|---------|---------|
| 00–01 | ISLP ch. 1–2; Murphy PML ch. 2–3, 7–8; Goodfellow DL ch. 2–4 |
| 02 | ISLP ch. 3, 6; ESL ch. 3; Bishop ch. 3 |
| 03 | ISLP ch. 4; Bishop ch. 4; Murphy ch. 10 |
| 04 | ISLP ch. 5; ESL ch. 7 |
| 05 | ISLP ch. 2.2, 4.4; ESL ch. 13; Murphy ch. 9 |
| 06 | ISLP ch. 9; Bishop ch. 7; ESL ch. 12 |
| 07 | ISLP ch. 8; ESL ch. 9–10, 15; Friedman (2001) "Greedy Function Approximation" |
| 08 | ISLP ch. 12; Bishop ch. 9; Murphy ch. 21 |
| 09 | ISLP ch. 12; Bishop ch. 12; van der Maaten & Hinton (2008) |
| 10 | Goodfellow DL ch. 6, 8; Bishop ch. 5 |
| 11 | Goodfellow DL ch. 7, 9, 10; Vaswani et al. (2017) "Attention Is All You Need" |
| 12 | Géron ch. 2; Mitchell et al. (2019) "Model Cards"; Barocas, Hardt & Narayanan — *Fairness and ML* |
