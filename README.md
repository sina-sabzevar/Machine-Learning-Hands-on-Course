# Machine Learning — Hands-on Course

A complete, **university-level Machine Learning course** delivered as Jupyter notebooks.
Every lecture combines **theory** (derivations, intuition, assumptions) with **hands-on code**:
algorithms are first implemented **from scratch in NumPy**, then checked against
**scikit-learn** / **PyTorch**, and explored visually.

Level: upper-undergraduate / first-year graduate. Suggested length: one semester (14 weeks).

## Syllabus

| # | Notebook | Topics |
|---|----------|--------|
| 00 | [Course Introduction & Python Toolkit](notebooks/00_course_introduction_and_python_toolkit.ipynb) | The learning problem, risk minimization, NumPy, pandas, matplotlib, scikit-learn API |
| 01 | [Mathematical Foundations](notebooks/01_mathematical_foundations.ipynb) | Linear algebra, SVD, gradients, probability, MLE/MAP, gradient descent |
| 02 | [Linear Regression & Regularization](notebooks/02_linear_regression.ipynb) | Least squares, normal equations, GD/SGD, bias–variance, Ridge, Lasso |
| 03 | [Logistic Regression & Classification](notebooks/03_logistic_regression_and_classification.ipynb) | Cross-entropy, softmax regression, ROC/AUC, PR curves, calibration |
| 04 | [Model Evaluation & Selection](notebooks/04_model_evaluation_and_selection.ipynb) | Cross-validation, learning curves, hyperparameter search, leakage, pipelines |
| 05 | [k-NN & Naive Bayes](notebooks/05_knn_and_naive_bayes.ipynb) | Instance-based learning, curse of dimensionality, generative classifiers, LDA/QDA |
| 06 | [Support Vector Machines](notebooks/06_support_vector_machines.ipynb) | Max margin, duality & KKT, hinge loss, kernel trick |
| 07 | [Decision Trees & Ensembles](notebooks/07_decision_trees_and_ensembles.ipynb) | CART, Random Forests, AdaBoost, gradient boosting |
| 08 | [Clustering](notebooks/08_clustering.ipynb) | k-means, GMM & EM, hierarchical clustering, DBSCAN |
| 09 | [Dimensionality Reduction](notebooks/09_dimensionality_reduction.ipynb) | PCA (eigen & SVD), kernel PCA, t-SNE |
| 10 | [Neural Networks from Scratch](notebooks/10_neural_networks_from_scratch.ipynb) | Perceptron, MLP, backpropagation, initialization, Adam |
| 11 | [Deep Learning with PyTorch](notebooks/11_deep_learning_with_pytorch.ipynb) | Autograd, training loops, regularization, CNNs, attention |
| 12 | [Capstone: End-to-End Project & Responsible ML](notebooks/12_capstone_end_to_end_project.ipynb) | Full workflow, interpretation, fairness, deployment |

See [SYLLABUS.md](SYLLABUS.md) for the weekly schedule, assessment scheme and reading list.

## Structure of each notebook

1. Learning objectives and prerequisites
2. Theory sections with full mathematical derivations
3. From-scratch implementations, checked against library implementations
4. Visualizations and experiments
5. Pitfalls and practical tips
6. Exercises (pen-and-paper and coding)
7. Summary and further reading

## Getting started

```bash
git clone https://github.com/sina-sabzevar/Machine-Learning-Hands-on-Course.git
cd Machine-Learning-Hands-on-Course
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

You can also open any notebook in **Google Colab** (File → Open notebook → GitHub). All required libraries are pre-installed there.

All datasets are either bundled with scikit-learn or generated synthetically, so **no downloads are needed**
and every notebook runs on a laptop CPU in a few minutes. The notebooks are committed **with outputs**,
so you can read them directly on GitHub.

## Regenerating the notebooks

The notebooks are generated from the Python scripts in [`_build/`](_build) (`cd _build && python build_NN.py`),
which also execute them. Build one notebook at a time; on machines with few cores, `export OMP_NUM_THREADS=1` speeds things up. Edit the scripts, not the `.ipynb` files, if you want reproducible changes.

## Recommended textbooks

- G. James, D. Witten, T. Hastie, R. Tibshirani, J. Taylor — *An Introduction to Statistical Learning (ISLP)*
- T. Hastie, R. Tibshirani, J. Friedman — *The Elements of Statistical Learning (ESL)*
- C. Bishop — *Pattern Recognition and Machine Learning (PRML)*
- K. Murphy — *Probabilistic Machine Learning: An Introduction*
- I. Goodfellow, Y. Bengio, A. Courville — *Deep Learning*
- A. Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*

## License

Course material is released under the [MIT License](LICENSE).
