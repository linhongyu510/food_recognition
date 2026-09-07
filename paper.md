---
title: 'food_recognition: a reproducible pipeline for food image classification with seed-aware ablation reporting'
tags:
  - Python
  - PyTorch
  - deep learning
  - image classification
  - reproducibility
  - attention mechanisms
authors:
  - name: Hongyu Lin
    affiliation: 1
affiliations:
  - index: 1
    name: Independent researcher
date: 7 September 2026
bibliography: paper.bib
---

# Summary

`food_recognition` is a Python package for training, evaluating and explaining
image classifiers that sort photographs of food into categories such as bread,
dairy, soup or seafood. It wraps PyTorch [@paszke2019pytorch] in a
configuration-driven pipeline: a single YAML file records the architecture, input
resolution, optimiser schedule and random seed, and command-line tools then train
a model, score it, explain individual predictions with saliency heatmaps, and
publish the resulting weights. Anyone with the configuration file can rerun the
experiment and obtain the same numbers.

The package is aimed at a specific and often overlooked failure mode. Small
image-classification studies are usually reported as a single training run per
configuration, and the resulting table of accuracies is read as a ranking. When
the validation set is small, the differences between neighbouring rows in such a
table can be smaller than the variation caused by changing nothing but the random
seed. A ranking built that way is not reproducible, even by its own author. This
package therefore treats seed variation as a first-class measurement rather than
an afterthought: it ships tooling that reruns a configuration under several seeds,
reports the mean, standard deviation and spread, and renders comparison figures
with error bars. In the authors' own benchmark this machinery overturned a
previously published claim in the project's own documentation, which is the
clearest available demonstration that the tooling does something useful.

# Statement of need

Researchers and students who apply convolutional networks to a domain dataset
face a practical question: given a fixed compute budget, is it better to use a
larger backbone or to feed a smaller backbone higher-resolution images? Answering
it requires a grid of runs that differ in exactly two variables, plus a way to
tell which of the resulting differences are real.

The second half of that requirement is where existing workflows tend to fall
short. Frameworks make it easy to train one model per configuration and tabulate
the results; they do not make it easy to ask whether a 0.15-percentage-point gap
between two cells means anything. The consequences are well documented: random
seeds alone shift computer-vision accuracy enough to reorder architectures
[@picard2021], benchmark comparisons draw on several independent sources of
variation that a single run cannot separate [@bouthillier2021accounting], and
reporting conventions that hide this variation make published results difficult
to interpret [@dodge2019show]. Despite this, single-seed comparison tables remain
standard in applied and course-project work, partly because measuring variance
means paying for repeated runs and partly because no lightweight tooling makes the
resulting bookkeeping easy.

`food_recognition` targets that gap for the applied practitioner. It provides the
grid machinery and the variance machinery in one place, so that the cost of doing
the honest thing is one extra command rather than a bespoke analysis script. The
intended users are researchers running modest ablations on modest datasets,
instructors who want students to confront measurement noise directly, and
practitioners choosing between an accurate model and a cheap one.

# State of the field

General training frameworks such as PyTorch Lightning and fastai remove
boilerplate from the training loop, and experiment trackers such as MLflow and
Weights & Biases record runs and hyperparameters. Neither category answers the
question this package addresses. Trackers store what happened across many runs but
leave the practitioner to decide whether a difference between two of them is
meaningful; training frameworks standardise the loop but take no position on how
many seeds a claim requires. Statistical treatments of benchmark variance
[@bouthillier2021accounting; @dodge2019show] supply the methodology but not an
implementation tied to a runnable pipeline.

The contribution here is deliberately narrow, and building rather than
contributing upstream was chosen because the gap is one of integration rather than
of missing primitives. Reruns with fixed seeds, aggregation into mean and spread,
figures carrying error bars, and prose that states which comparisons survive the
noise are individually trivial; what is missing is a pipeline in which they are
the default path and in which the published claims are mechanically checked
against the recorded measurements. Rather than add a variance module to a large
framework, this package keeps a small, auditable surface where the configuration,
the metrics, the figure and the documented claim are all derived from the same
files.

# Software design

The central design decision is that a run's configuration is data, not code. A
`TrainingConfig` is loaded and validated from YAML, individual fields can be
overridden from the command line, and the resolved configuration is embedded in
every checkpoint. Reconstructing a model therefore never depends on the caller
remembering how it was built: the architecture, class list and input resolution
travel with the weights, which is what allows published checkpoints to be
reloaded and rescored by a third party. The corresponding trade-off is that the
package is opinionated about file layout and less suited to research that needs to
modify the training loop itself.

Attention is implemented as a wrapper rather than a fork. CBAM
[@woo2018cbam] applies channel attention followed by spatial attention, and the
implementation reads the attention width from the wrapped backbone at build time,
so the same code composes with any supported feature extractor. Twenty-nine
architectures are exposed through one factory function, and the compound-scaling
family [@tan2019efficientnet] supplies the backbones used in the reported
experiments. Saliency maps use Grad-CAM [@selvaraju2017gradcam], which requires no
architectural change and therefore does not constrain the model factory.

The variance tooling follows the same principle of deriving conclusions from
recorded artefacts. Each run writes its own metrics file; an aggregation script
reads those files and emits per-cell mean, standard deviation, spread and
per-seed values; the plotting script consumes that report and draws error bars.
Because every number in the documentation is derived from committed JSON rather
than transcribed by hand, a claim and its evidence cannot silently drift apart.

# Research impact statement

The package's own benchmark demonstrates its purpose. A grid of nine runs on
Food-11 varied backbone size and input resolution under an identical schedule,
and the single-seed table appeared to show that the smallest backbone at 300 px
outperformed the largest at its native 380 px by 0.15 percentage points while
using a quarter of the compute. Rerunning four cells with two additional seeds
reversed that finding: across three seeds the larger backbone leads by 0.45
points, because the original seed happened to be its worst of three. The observed
seed spread reached 0.76 points on a 660-image validation split, which is wider
than most of the gaps the original table had invited readers to rank. The claim
was retracted in the project's documentation and replaced with an explicit
statement of which comparisons survive the noise and which do not.

That episode is the substantive result: on a dataset of this size, a single-seed
ablation grid produced a confidently stated and incorrect ordering, and the
tooling described here detected it. The materials needed to check the finding are
public. Eleven trained checkpoints are published with model cards recording the
configuration and measured accuracy of each, the aggregated variance report and
the ablation figure are committed to the repository, and the test suite runs on
three Python versions in continuous integration. Food-101 [@bossard2014food101]
is supported as a second, larger benchmark, providing a route to test whether the
resolution effect reported here persists when training data is abundant.

# AI usage disclosure

Generative AI tools were used during development of the software, its
documentation and the preparation of this paper. All quantitative results reported
here were produced by executing the software on real data, not generated by a
language model. Every figure was re-derived from committed measurement files, and
the numerical claims in the documentation are checked against those files by a
script that parses the rendered text, so that a mismatch between a stated number
and its recorded evidence fails rather than passing silently. Reference metadata
was verified against Crossref and OpenAlex records rather than recalled.

# Acknowledgements

This work received no financial support.

# References
