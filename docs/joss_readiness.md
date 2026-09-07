# JOSS submission readiness

Status as of 2026-09-07: **not ready to submit.** Two of JOSS's pre-review
screening gates are currently failed. Both are about the repository's history,
not about the paper or the code, and neither can be fixed by editing text.

This file records what was checked, against which source, so the situation can
be re-assessed later without repeating the research.

## Gate 1 — public development history (FAILED)

JOSS requires that the repository "must have been public for more than six
months prior to submission, with active development spanning that period," and
states plainly: "We run automated checks on commit distribution — a repo dump is
not a history." A repository "made public immediately before submission" will not
be accepted.

Source: <https://github.com/openjournals/joss/blob/main/docs/submitting.md>

Measured on this repository:

| Metric | Value |
|---|---|
| Repository created | 2025-03-23 |
| Made public | 2026-09 (this month) |
| Total commits | 33 |
| Commits dated 2026-09 | 22 (67%) |
| Distinct commit days | 6 |

The commit history spans 2025-03 to 2026-09, but two thirds of it lands in a
single month, across three days. This is the concentration pattern the gate
describes. The six-month public-history clock also started only this month.

**What would change this:** continued open development over at least six months
from the date the repository became public, spread across time rather than
concentrated. This is a matter of elapsed calendar time and cannot be
accelerated.

## Gate 2 — demonstrated research impact (FAILED as stated)

JOSS requires "evidence that the software is being used for research — at
minimum by the developers themselves, and ideally by others," and rejects
"aspirational statements about future use." Acceptable signals are references in
published papers or preprints, documented adoption by other research groups, or
clear integration into research workflows.

Measured: 1 star, 0 forks, 0 issues, 12 pull requests all authored by the
repository owner, no external contributors, no citing publication.

The seed-variance result in `docs/benchmarks/food11_seed_variance.json` is a
genuine finding produced with this software, which is the "at minimum by the
developers themselves" case. Whether an editor accepts that as sufficient
research use is a judgement call, and it is weaker than the usual evidence.

## Also relevant: scope

JOSS states that "pre-trained machine learning models and notebooks are not
in-scope," and that "'Minor utility' packages, including 'thin' API clients, and
single-function packages are not acceptable." This package is a training and
evaluation pipeline rather than a pre-trained model, so it is not excluded on
that basis, but the published weights are not themselves the contribution and
the paper should not be framed as being about them. The current `paper.md`
frames the contribution as the pipeline plus its seed-variance tooling, which is
the correct framing.

## Indexing and catalogue status

These were checked because they determine what a JOSS publication is worth for
formal evaluation purposes:

- **Not indexed in Scopus or Web of Science.** JOSS's own editorial team reports
  having applied multiple times, and that "neither Scopus nor Web of Science
  have chosen to index JOSS."
  Source: <https://mail.danielskatz.org/papers/JOSS_JLSC_2025.pdf>
- **No JCR impact factor**, which follows from not being in SCIE/SSCI.
- **Not in the CCF recommended catalogue.** Verified by full-text search of the
  official PDFs of both the 2022 edition and the current seventh edition
  (published 2026-03-31), using TPAMI, TOSEM, CVPR and ICSE as positive
  controls to confirm the search itself was working.
  Source: <https://www.ccf.org.cn/Academic_Evaluation/By_category/>
- **Is** indexed by Crossref and DOAJ, and assigns a citable DOI.

Note that the CCF catalogue itself states that its purpose "is not to serve as a
basis for academic evaluation" and that it explicitly regards open-source
software as a legitimate form of scholarly output.

## Alternatives if a catalogued venue is required

Not investigated in depth; recorded only so the options are known:

- **SoftwareX** (Elsevier) — also a software-paper journal, indexed in Scopus
  and SCIE, but also absent from the CCF catalogue, and charges an APC.
- A **domain conference or journal paper** about the measurement finding rather
  than about the software. The seed-variance result would need to be extended
  well beyond four cells of one small dataset to carry a paper on its own.

## Re-assessment checklist

Before reconsidering submission, re-check:

1. Has the repository been public for more than six months, with commits spread
   across that period rather than clustered?
2. Is there any external signal — an issue, a fork, a citation, a user?
3. Has anything been published or preprinted that uses this software?
