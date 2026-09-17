#!/usr/bin/env python3
"""
Bibliometric coding of food-image-recognition papers that use attention modules.

Question: among published attention-gain claims in food image recognition, how many
report enough information (multi-run variance + isolated no-attention ablation) for
the claimed gain to be judged against seed noise at all?

Reference noise scale: this project's own measured 3-seed max range on Food-11
(EfficientNet-B0@300, seeds 1/2) = 0.76 accuracy points.
See work/seeds/*/metrics.json.

All fields coded from FULL TEXT (body tables), not abstracts. Field `src_*` gives
the provenance for each judgement. Papers whose full text could not be obtained are
coded status="unverifiable" and EXCLUDED from denominators (reported separately).

Run: python3 code_corpus.py
Outputs: corpus.csv, corpus.json, stats.json
"""

import csv
import json
import math

# Project's own measured seed noise (accuracy points), Food-11, 3 seeds
SEED_NOISE_PTS = 0.76

# ---------------------------------------------------------------------------
# INCLUSION / EXCLUSION PROTOCOL (fixed BEFORE coding; do not adjust post hoc)
# ---------------------------------------------------------------------------
PROTOCOL = {
    "research_question": (
        "In food image recognition papers that add an attention module, what fraction "
        "report (a) multi-run/multi-seed variance and (b) an isolated no-attention "
        "ablation, such that the claimed attention gain can be compared against seed noise?"
    ),
    "databases_searched": [
        "arXiv (full-text HTML + PDF)",
        "Google Scholar (via general web search)",
        "Semantic Scholar",
        "PapersWithCode / HuggingFace papers",
        "PubMed Central (PMC)",
        "MDPI / Frontiers / PLOS ONE / IEEE Xplore / SciOpen (CMC)",
        "SCIRP (J. Computer and Communications)",
        "CNKI-indexed Chinese journals (粮油食品科技 / 软件学报 / 计算机工程)",
        "GitHub REST API (code availability)",
        "HuggingFace Hub API (weight availability)",
    ],
    "queries_used": [
        "food image classification CBAM attention module accuracy Food-101 ResNet EfficientNet "
            "2023 2024 2025",
        "fine-grained food recognition attention mechanism ablation study baseline improvement "
            "Food-101 ChineseFoodNet UECFOOD",
        "食物图像识别 注意力机制 CBAM 消融实验 准确率 提升",
        "food recognition transformer attention Food-101 accuracy 2024 2025 ablation w/o "
            "attention module arXiv",
        "EfficientNet CBAM food classification attention accuracy improvement percentage ablation "
            "table",
        "food image recognition squeeze excitation SE attention module Food-101 Vireo UEC "
            "accuracy ablation improvement",
        "fruit vegetable freshness classification CBAM attention ResNet accuracy improvement "
            "ablation baseline 2025",
        "food image classification attention mechanism standard deviation multiple seeds reported "
            "statistical significance",
    ],
    "inclusion_criteria": [
        "I1: task is food / food-adjacent image RECOGNITION or CLASSIFICATION",
        "I2: the method incorporates an explicit attention module (CBAM/SE/HA/self-"
            "attn/spatial/channel)",
        "I3: reports a classification accuracy figure on at least one dataset",
        "I4: full text obtainable, so ablation presence can be verified from body tables",
    ],
    "exclusion_criteria": [
        "E1: no attention module (e.g. pure transfer-learning backbone comparison)",
        "E2: not classification (detection / segmentation / counting only)",
        "E3: not food domain (medical, plant disease, industrial, generic ImageNet-only)",
        "E4: full text unobtainable -> coded unverifiable, excluded from denominators",
    ],
    "time_window": "2021-01-01 .. 2026-09-13 (search date)",
    "domain_scope_note": (
        "Primary scope is food dish/category recognition. Food-adjacent produce-quality "
        "classification (ripeness/freshness) is coded separately via `subdomain` so it can "
        "be included or excluded in sensitivity analysis."
    ),
}

# ---------------------------------------------------------------------------
# CODED CORPUS
# reports_multirun    : bool  - any multi-run / multi-seed protocol reported
# multirun_form       : mean_only | mean_std | per_run_values | none
# n_runs              : int or None
# has_attn_ablation   : bool  - ISOLATED no-attention vs attention comparison in body
# attn_gain_pts       : float or None - gain attributable to the ATTENTION module alone
# gain_is_headline    : bool  - True if only a whole-system gain exists (not attn-isolated)
# code_public / weights_public
# dataset_obtainable  : yes | partial | no
# stat_test           : none | t-test | ci | other
# ---------------------------------------------------------------------------
PAPERS = [
    {
        "id": "P01", "cite": "Rokhva & Teimourpour 2024, arXiv:2410.02304 / Food & Humanity 2025",
        "method": "EfficientNetB7 + CBAM + TL + aug", "dataset": "Food-11 (Kaggle, 16643)",
        "subdomain": "food_dish", "year": 2024, "status": "included",
        "reports_multirun": True, "multirun_form": "per_run_values", "n_runs": 5,
        "per_run_values": [96.24, 96.44, 96.51, 96.42, 96.38],
        "reported_range_pts": 0.27,
        "has_attn_ablation": False, "attn_gain_pts": None, "gain_is_headline": True,
        "headline_acc": 96.40,
        "code_public": True, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "Sec.3 Results: 'executed on the test data 5 times, each time all "
            "parameters were initialized from scratch'; five values 96.24/96.44/96.51/96.42/96.38 "
            "listed verbatim (PDF p.11)",
        "src_ablation": "Full 20p text grep: no 'without CBAM'/'w/o'/'ablation'. Table 1 = "
            "EfficientNet-vs-others on ImageNet (cited, not own expt); Table 2 = cross-paper "
            "Food-11 "
            "comparison. No B7-vs-B7+CBAM row.",
        "src_code": "github.com/Shayan1999rokh/Accurate-Food-Image-Recognition-...: 2 stars, 1 "
            "commit (2025-10-16), 0 issues, no license, 2 files (.ipynb/.py), no README; grep "
            "seed/manual_seed -> 0 hits",
        "src_weights": "repo tree has no .pt; code line 1307 saves to author's private Google "
            "Drive",
        "notes": "Only paper in corpus reporting per-run values. Reported range 0.27pt < this "
            "project's 0.76pt. But NO attention ablation -> no isolable attention gain exists to "
            "test.",
    },
    {
        "id": "P02", "cite": "Xu, He, Qu 2021, J. Comput. Commun. 9, 10-28",
        "method": "CBAM + MobileNetV2/VGG16/ResNet50 + Mixup", "dataset": "UECFOOD100 (14361)",
        "subdomain": "food_dish", "year": 2021, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 1.83, "gain_is_headline": False,
        "headline_acc": 87.33,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "no", "stat_test": "none",
        "src_multirun": "Sec.4.2: all results single figures; no repetition/seed/std language "
            "anywhere in full text",
        "src_ablation": "Sec.4.2.2 + Table 2: 'CBAM models have better classification performance "
            "... than benchmark models'; Sec.4.2.1 Table 1 compares CBAM orderings, max spread "
            "1.83pt "
            "(VGG16 channel-before-space vs parallel)",
        "src_code": "No code statement in full text (read via scirp.org paperid=107550)",
        "src_weights": "none stated",
        "notes": "Has proper attention ablation but no variance. Dataset blocked: "
            "mm.cs.uec.ac.jp/uecfood/ -> HTTP 404; custom bbox-crop preprocessing script "
            "unpublished.",
    },
    {
        "id": "P03", "cite": "Deng, Wu, Chen 2024, Comput. Mater. Continua 80(2) 1985-2003",
        "method": "ConvNeXt-B + Hybrid Attention (HA) + MSLF",
        "dataset": "Food-101 / ChineseFoodNet / Roushi60",
        "subdomain": "food_dish", "year": 2024, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 0.70, "gain_is_headline": False,
        "attn_gain_all": [0.70, 1.24, 0.64],
        "headline_acc": 91.12,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "Full text (SciOpen PDF): no seed/std/repeat language; all cells single "
            "values",
        "src_ablation": "Sec.5.2.1 Table 5: ConvNeXt-B 90.08/79.44/91.14 -> +HA "
            "90.78/80.68/91.78, i.e. ISOLATED HA gain = +0.70/+1.24/+0.64 pts. (Headline "
            "+1.04/+3.42/+1.36 is the FULL MAMS-net, not attention alone.)",
        "src_code": "No code link in CMC open-access full text; GitHub search 'Roushi60' -> total "
            "0, 'MSLF+food+ConvNeXt' -> total 0",
        "src_weights": "none",
        "notes": "KEY: isolated HA gain (0.70/1.24/0.64) is SMALLER than headline "
            "(1.04/3.42/1.36). Two of three isolated gains < 0.76pt noise. Roushi60 self-built, "
            "unreleased.",
    },
    {
        "id": "P04", "cite": "Gao, Ye, Xiao 2025, arXiv:2509.18692",
        "method": "AlsmViT + WMHAM + SAM (window/spatial attention)",
        "dataset": "Food-101 / Vireo Food-172",
        "subdomain": "food_dish", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 0.16, "gain_is_headline": False,
        "attn_gain_all": [0.16, -0.11],
        "headline_acc": 95.24,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "Full HTML text: 0 hits for seed/std/standard deviation/runs/repeated",
        "src_ablation": "Sec.3.2 Table 1. Food-101: AlsmViT 95.17 -> +SAM 95.33 (+0.16) -> Ours "
            "95.24. Vireo: AlsmViT 94.29 -> +WMHAM 94.18 (-0.11), +SAM 94.19 (-0.10), Ours 94.33 "
            "(+0.04)",
        "src_code": "No code/github link in paper",
        "src_weights": "none",
        "notes": "STRONGEST evidence: +SAM alone (95.33) BEATS the full proposed model (95.24) on "
            "Food-101; on Vireo BOTH attention variants UNDERPERFORM the baseline. All deltas "
            "|.|<=0.16pt, far below 0.76pt noise, yet reported as improvement.",
    },
    {
        "id": "P05", "cite": "Zou et al. 2025, arXiv:2503.11995 (Fraesormer)",
        "method": "Adaptive Top-k Sparse Partial Attention (ATK-SPA) + HSSFGN",
        "dataset": "UEC-256 / Food-101 / Vireo",
        "subdomain": "food_dish", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 1.833, "gain_is_headline": False,
        "headline_acc": None,
        "code_public": True, "code_official": True, "weights_public": False,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "Full HTML: 0 hits for seed/std/runs/repeated",
        "src_ablation": "Sec.III-C Table II: 'When we removed ATK-SPA, the model's accuracy "
            "decreased by 1.833%'; Table VI gating/multi-scale 63.406->64.548",
        "src_code": "'code is available at https://zs1314.github.io/Fraesormer' (abstract)",
        "src_weights": "not stated",
        "notes": "Attention-removal gain 1.833pt > 0.76pt noise. One of few with code link.",
    },
    {
        "id": "P06", "cite": "Zhuang, Hu et al. 2024, arXiv:2403.12109 (GCAM)",
        "method": "Gaussian feature fusion + causal counterfactual attention (CRA)",
        "dataset": "Food-101/UEC256/Vireo172/CUB",
        "subdomain": "food_dish", "year": 2024, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 1.90, "gain_is_headline": False,
        "attn_gain_all": [1.03, 1.90, 1.68, 2.16],
        "headline_acc": 91.11,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "Full HTML: 0 hits for seed/std/repeated runs",
        "src_ablation": "Sec.4.3 Table 4 rows (a) vs (c): baseline->+CRA. FOOD101 86.54->87.57 "
            "(+1.03); FOOD256 83.69->85.59 (+1.90); Vireo172 86.06->87.74 (+1.68); CUB "
            "84.06->86.22 "
            "(+2.16). Text: 'CRA module contributed 1.9% increase on FOOD256'",
        "src_code": "No code link found in paper text",
        "src_weights": "none",
        "notes": "Isolated attention (CRA) gains 1.03-2.16pt, above noise. Has clean incremental "
            "ablation table.",
    },
    {
        "id": "P07", "cite": "Liu & Xiao 2025, arXiv:2507.12828 (FE-TResNet)",
        "method": "TResNet-XL + StyleRM + Deep Channel-wise Attention (DCA)",
        "dataset": "ChineseFoodNet / CNFOOD-241",
        "subdomain": "food_dish", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 0.06, "gain_is_headline": False,
        "attn_gain_all": [0.06, 0.07],
        "headline_acc": 81.37,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "Full HTML: 'standard deviation' appears only as global-std POOLING "
            "inside StyleRM, not as run variance. 0 hits for seed/runs",
        "src_ablation": "Sec.4.4 Table 3. ChineseFoodNet: base 80.85 -> +DCA only 80.91 (+0.06); "
            "CNFOOD-241: 79.85 -> 79.92 (+0.07). (StyleRM alone +0.37/+0.30; both 81.37/80.29)",
        "src_code": "No code link in paper",
        "src_weights": "none",
        "notes": "Isolated DCA attention gain +0.06/+0.07 pts -- an order of magnitude BELOW the "
            "0.76pt seed noise, yet reported as a contribution.",
    },
    {
        "id": "P08", "cite": "Du, Cui, Wang et al. 2024, 粮油食品科技 32(1) 91-98",
        "method": "InceptionV3 + 11x CBAM + transfer learning", "dataset": "Food-101",
        "subdomain": "food_dish", "year": 2024, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 1.40, "gain_is_headline": False,
        "attn_gain_all": [1.40, 3.05],
        "headline_acc": 82.01,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "Full text: reports single best-accuracy figures "
        "('最高准确率'); explicitly "
            "disables augmentation to avoid randomness bias but never repeats runs",
        "src_ablation": "Sec.3.2 + Table 2: with ImageNet weights InceptionV3 80.61 -> CBAM-"
            "InceptionV3 82.01 (+1.40pt); without weights 63.52 -> 66.57 (+3.05pt)",
        "src_code": "none stated (hardware: RTX3080, CUDA 12.1, TF 2.12)",
        "src_weights": "none",
        "notes": "Clean 2x2 design (CBAM x pretrained). Gain +1.40pt above noise; notes CBAM "
            "helps more without pretraining.",
    },
    {
        "id": "P09", "cite": "Chen et al. 2024/2025, Res-VMamba, arXiv:2402.15761 / PLOS ONE 2025",
        "method": "VMamba + deep residual (selective state space)", "dataset": "CNFOOD-241",
        "subdomain": "food_dish", "year": 2024, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": None, "gain_is_headline": False,
        "headline_acc": None,
        "code_public": True, "code_official": True, "weights_public": True,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "No seed/variance protocol stated; single reported figures",
        "src_ablation": "Paper compares VMamba backbone vs Res-VMamba (residual variant); "
            "ablation present but attention-vs-none not isolable (SSM is the whole backbone)",
        "src_code": "github.com/ChiShengChen/ResVMamba: 84 stars, 67 files, pushed 2025-08-07, "
            "configs + data splits included",
        "src_weights": "HF ms57rd/Res-VMamba/ckpt_epoch_166.pth verified real: x-linked-"
            "size=711402283 (711MB), HTTP 200",
        "notes": "Best artifact hygiene in corpus (official code + real weights). But mamba_ssm "
            "wheel is cu12+linux_x86_64 only and README requires torch.distributed.launch -> "
            "cannot "
            "run on MPS.",
    },
    {
        "id": "P10", "cite": "Sayudha & Sthevanie 2025, ICoDSA",
        "method": "ResNet50 + CBAM", "dataset": "Cavendish banana ripeness (custom)",
        "subdomain": "produce_quality", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": None, "gain_is_headline": False,
        "headline_acc": 94.42,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "no", "stat_test": "none",
        "src_multirun": "Abstract/record only; no variance language",
        "src_ablation": "Abstract: 'attention-fitted CNN model was able to perform better than "
            "the baseline model' -> baseline comparison present but gain magnitude not numerically "
            "stated in accessible record",
        "src_code": "none",
        "src_weights": "none",
        "notes": "Qualitative gain statement only; dataset unpublished.",
    },
    {
        "id": "P11", "cite": "Zhang, Song et al. 2024, 中国食品学报 (fruit freshness)",
        "method": "ResNet34 + CBAM + CAM visualisation",
        "dataset": "public fruit freshness dataset",
        "subdomain": "produce_quality", "year": 2024, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 2.91, "gain_is_headline": False,
        "headline_acc": 99.71,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "Abstract/record: single figures",
        "src_ablation": "Abstract: ResNet34 before/after CBAM = 96.80% -> 99.71% (+2.91pt)",
        "src_code": "none",
        "src_weights": "none",
        "notes": "Large gain but ceiling-effect dataset (99.71%); before/after pair explicitly "
            "given.",
    },
    {
        "id": "P12", "cite": "Frontiers Sustain. Food Syst. 2024, 10.3389/fsufs.2024.1310042",
        "method": "CA-EfficientNet-CBAM", "dataset": "vegetable quality grading (custom)",
        "subdomain": "produce_quality", "year": 2024, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 2.32, "gain_is_headline": False,
        "attn_gain_all": [2.32, 1.47],
        "headline_acc": 94.23,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "no", "stat_test": "none",
        "src_multirun": "Full PDF table: single values per strategy",
        "src_ablation": "Improvement-strategy table: EfficientNet 90.92 -> +CBAM 93.24 (+2.32pt); "
            "CA+EffNet 92.76 -> CA+EffNet+CBAM 94.23 (+1.47pt)",
        "src_code": "none",
        "src_weights": "none",
        "notes": "Clean incremental strategy table; both isolated CBAM gains above noise.",
    },
    {
        "id": "P13", "cite": "MDPI Foods 2025, 14(3) 383 (CBDTN)",
        "method": "Coarse-to-fine aggregation + Boundary-Aware Module (BAM)",
        "dataset": "Food-5k / Food-101 / Food-2k",
        "subdomain": "food_dish", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 0.90, "gain_is_headline": False,
        "attn_gain_all": [0.60, 0.90, 0.75],
        "headline_acc": 99.17,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "Ablation table single values",
        "src_ablation": "Ablation table: CBDTN 0.9980/0.9917/0.8587 vs w/o BAM "
            "0.9920/0.9827/0.8512 -> BAM gain +0.60/+0.90/+0.75 pts",
        "src_code": "none stated",
        "src_weights": "none",
        "notes": "Two of three isolated BAM gains straddle the 0.76pt noise line (0.60 below, "
            "0.90/0.75 near).",
    },
    {
        "id": "P14", "cite": "Liu, Min et al. 2022, 软件学报 33(11) 4379-4395 (MJR-Net)",
        "method": "Multi-scale jigsaw + channel attention module",
        "dataset": "Food-101 / ChineseFoodNet / ISIA Food-500",
        "subdomain": "food_dish", "year": 2022, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 0.82, "gain_is_headline": False,
        "attn_gain_all": [0.82, 0.70],
        "headline_acc": None,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "Ablation prose reports single deltas",
        "src_ablation": "Ablation: channel attention improves 0.82% at 224x224 and 0.70% at large "
            "resolution ('明显提升分类性能(0.82%)' / '仍有较大的性能提升(0.70%)')",
        "src_code": "none stated",
        "src_weights": "none",
        "notes": "Isolated channel-attention gains 0.82/0.70pt -- straddle the 0.76pt noise line "
            "exactly. Described as '明显'(marked) and '较大'(large).",
    },
    {
        "id": "P15", "cite": "Frontiers Comput. Sci. 2026, 10.3389/fcomp.2026.1753764",
        "method": "DenseNet121 + SE attention after transition layers",
        "dataset": "Indian food platters",
        "subdomain": "food_dish", "year": 2026, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": 0.52, "gain_is_headline": False,
        "headline_acc": None,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "No variance language in accessible text",
        "src_ablation": "Ablation: 'SE attention modules following DenseNet transition layers "
            "produces the best accuracy-efficiency trade-off ... +0.52% accuracy increase over the "
            "baseline DenseNet121'",
        "src_code": "none stated",
        "src_weights": "none",
        "notes": "Isolated SE gain +0.52pt, BELOW the 0.76pt seed noise, reported as the best "
            "configuration.",
    },
    {
        "id": "P16", "cite": "Machine Intelligence Research 2025, 10.1007/s11633-025-1574-0",
        "method": "Adaptive Bidirectional Hybrid Net (ACVCBlock + inflated attention)",
        "dataset": "food benchmarks",
        "subdomain": "food_dish", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": None, "gain_is_headline": False,
        "headline_acc": 92.831,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "partial", "stat_test": "none",
        "src_multirun": "No variance language in accessible text",
        "src_ablation": "Table 7 ablation: replacing ACVCBlock with unidirectional simple "
            "attention, and inflated attention with standard ViT self-attention, both reduce "
            "accuracy "
            "(deltas not extractable from accessible snippet)",
        "src_code": "none stated",
        "src_weights": "none",
        "notes": "Ablation exists (attention-variant swaps) but gain magnitude not extractable -> "
            "attn_gain_pts=None.",
    },
    {
        "id": "P17", "cite": "PMC 2025, PMC11816403",
        "method": "Multi-level fusion + self-attention + KL divergence",
        "dataset": "Food-101 / Vireo-172 / UEC-100",
        "subdomain": "food_dish", "year": 2025, "status": "included",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "per_run_values": None, "reported_range_pts": None,
        "has_attn_ablation": True, "attn_gain_pts": None, "gain_is_headline": False,
        "headline_acc": 90.22,
        "code_public": False, "code_official": False, "weights_public": False,
        "dataset_obtainable": "yes", "stat_test": "none",
        "src_multirun": "No variance language in accessible text",
        "src_ablation": "Paper includes component ablation for the attention/fusion modules; per-"
            "module deltas not extractable from accessible text",
        "src_code": "none stated",
        "src_weights": "none",
        "notes": "Ablation present per abstract/structure; magnitudes unextractable.",
    },
    {
        "id": "P18", "cite": "Min et al. 2019, ACM MM (IG-CMAN)",
        "method": "Ingredient-guided cascaded multi-attention network",
        "dataset": "Food-101 / Vireo-172 / WikiFood-200",
        "subdomain": "food_dish", "year": 2019, "status": "excluded",
        "exclude_reason": "E-time: predates 2021 time window",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "has_attn_ablation": True, "attn_gain_pts": None, "gain_is_headline": False,
        "code_public": False,
        "weights_public": False,
        "dataset_obtainable": "yes",
        "stat_test": "none",
        "src_multirun": "n/a", "src_ablation": "n/a", "src_code": "n/a", "src_weights": "n/a",
        "notes": "Frequently cited attention baseline; excluded on time window only.",
    },
    {
        "id": "P19", "cite": "Singh & Susan 2023, ICCCNT (IEEE 10307479)",
        "method": "Xception transfer learning (NO attention module)", "dataset": "Food-101",
        "subdomain": "food_dish", "year": 2023, "status": "excluded",
        "exclude_reason": "E1: no attention module -- pure pretrained-backbone comparison",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "has_attn_ablation": False, "attn_gain_pts": None, "gain_is_headline": False,
        "code_public": False,
        "weights_public": False,
        "dataset_obtainable": "yes",
        "stat_test": "none",
        "src_multirun": "n/a", "src_ablation": "n/a", "src_code": "n/a", "src_weights": "n/a",
        "notes": "Was on the original candidate list; multi-source description confirms it is a "
            "transfer-learning backbone comparison (Xception 84.54%, EffNet-B0 ~79.8%), no "
            "attention. "
            "EXCLUDED.",
    },
    {
        "id": "P20", "cite": "Woo et al. 2018, ECCV (CBAM original)",
        "method": "CBAM", "dataset": "ImageNet-1k / MS-COCO / VOC",
        "subdomain": "generic_cv", "year": 2018, "status": "excluded",
        "exclude_reason": "E3+E-time: not food domain; predates window",
        "reports_multirun": False, "multirun_form": "none", "n_runs": None,
        "has_attn_ablation": True, "attn_gain_pts": None, "gain_is_headline": False,
        "code_public": True,
        "weights_public": False,
        "dataset_obtainable": "no",
        "stat_test": "none",
        "src_multirun": "n/a",
        "src_ablation": "extensive ablations (canonical)",
        "src_code": "github.com/Jongchan/attention-module: 2233 stars, pushed 2023-03-09, 42 open "
            "issues",
        "src_weights": "training scripts only",
        "notes": "Method-origin reference. Excluded from food-domain denominators.",
    },
    {
        "id": "P21", "cite": "BSAM 2025, J. Syst. Eng. Electron. (JSEE) 10.23919/JSEE.2025.000051",
        "method": "Brief self-attention module (BSA + advanced channel attention)",
        "dataset": "Food-101 / Caltech-256 / Mini-ImageNet",
        "subdomain": "food_dish", "year": 2025, "status": "unverifiable",
        "exclude_reason": "E4: full text not obtainable; ablation deltas unverified",
        "reports_multirun": None, "multirun_form": "unknown", "n_runs": None,
        "has_attn_ablation": None, "attn_gain_pts": None, "gain_is_headline": None,
        "code_public": False,
        "weights_public": False,
        "dataset_obtainable": "yes",
        "stat_test": "unknown",
        "src_multirun": "UNVERIFIED - abstract only (jseepub.com/EN/abstract/abstract10738)",
        "src_ablation": "UNVERIFIED - module papers conventionally ablate, but not confirmed from "
            "body tables",
        "src_code": "GitHub search 'BSAM attention module' -> total 0",
        "src_weights": "none found",
        "notes": "Coded unverifiable per E4; EXCLUDED from denominators, reported separately.",
    },
    {
        "id": "P22", "cite": "Xiao, Liu et al. 2026, 计算机工程 (MTFNet)",
        "method": "Detail attention + self-attention multi-feature fusion",
        "dataset": "Food-101 / ChineseFoodNet / ISIA Food-500",
        "subdomain": "food_dish", "year": 2026, "status": "unverifiable",
        "exclude_reason": "E4: only abstract accessible; per-module attention delta not confirmed",
        "reports_multirun": None, "multirun_form": "unknown", "n_runs": None,
        "has_attn_ablation": None, "attn_gain_pts": None, "gain_is_headline": None,
        "code_public": False,
        "weights_public": False,
        "dataset_obtainable": "yes",
        "stat_test": "unknown",
        "src_multirun": "UNVERIFIED - abstract reports +0.44/+1.01/+0.66 pts vs MJR-Net (whole-"
            "system, not attention-isolated)",
        "src_ablation": "UNVERIFIED",
        "src_code": "none found",
        "weights_public_note": "none",
        "notes": "Whole-system gains 0.44-1.01pt straddle noise, but attention not isolated in "
            "accessible text.",
    },
]

# ---------------------------------------------------------------------------
def wilson(k, n, z=1.959963985):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (round(p, 4), round((c-m)/d, 4), round((c+m)/d, 4))


def _prop(k: int, n: int) -> dict:
    """Proportion with its Wilson interval, as a flat record."""
    p, lo, hi = wilson(k, n)
    return {"p": p, "lo": lo, "hi": hi, "k": k, "n": n}


def main():
    inc = [p for p in PAPERS if p["status"] == "included"]
    exc = [p for p in PAPERS if p["status"] == "excluded"]
    unv = [p for p in PAPERS if p["status"] == "unverifiable"]
    K = len(inc)

    # ---- core fields
    var = [p for p in inc if p["reports_multirun"]]
    abl = [p for p in inc if p["has_attn_ablation"]]
    both = [p for p in inc if p["reports_multirun"] and p["has_attn_ablation"]]
    neither = [p for p in inc if not p["reports_multirun"] and not p["has_attn_ablation"]]
    var_only = [p for p in inc if p["reports_multirun"] and not p["has_attn_ablation"]]
    abl_only = [p for p in inc if not p["reports_multirun"] and p["has_attn_ablation"]]
    stat = [p for p in inc if p["stat_test"] not in ("none", "unknown")]
    code = [p for p in inc if p.get("code_public")]
    offi = [p for p in inc if p.get("code_official")]
    wts = [p for p in inc if p.get("weights_public")]

    # ---- gains vs noise (only papers with a numeric ISOLATED attention gain)
    quant = [p for p in inc if p.get("attn_gain_pts") is not None]
    below = [p for p in quant if p["attn_gain_pts"] < SEED_NOISE_PTS]
    # per-dataset granularity: use attn_gain_all where present
    all_deltas = []
    for p in quant:
        ds = p.get("attn_gain_all") or [p["attn_gain_pts"]]
        for d in ds:
            all_deltas.append((p["id"], d))
    below_d = [x for x in all_deltas if x[1] < SEED_NOISE_PTS]

    stats = {
        "seed_noise_reference_pts": SEED_NOISE_PTS,
        "seed_noise_provenance": (
            "This project's measured max range across seeds on Food-11, EfficientNet-B0@300 "
            "(b0_300_s1=0.953030 vs b0_300_s2=0.945455 -> 0.757pt ~ 0.76pt). "
            "Source: work/seeds/b0_300_s{1,2}/metrics.json"
        ),
        "funnel": {
            "total_screened": len(PAPERS),
            "excluded": len(exc),
            "excluded_detail": [{"id": p["id"], "reason": p["exclude_reason"]} for p in exc],
            "unverifiable_full_text": len(unv),
            "unverifiable_detail": [{"id": p["id"], "reason": p["exclude_reason"]} for p in unv],
            "included_K": K,
        },
        "proportions_of_K": {
            "reports_multirun_variance": _prop(len(var), K),
            "has_isolated_attn_ablation": _prop(len(abl), K),
            "BOTH_variance_and_ablation": _prop(len(both), K),
            "any_statistical_test": _prop(len(stat), K),
            "code_public": _prop(len(code), K),
            "official_code": _prop(len(offi), K),
            "weights_public": _prop(len(wts), K),
        },
        "crosstab_2x2_variance_x_ablation": {
            "both": {"n": len(both), "ids": [p["id"] for p in both]},
            "variance_only": {"n": len(var_only), "ids": [p["id"] for p in var_only]},
            "ablation_only": {"n": len(abl_only), "ids": [p["id"] for p in abl_only]},
            "neither": {"n": len(neither), "ids": [p["id"] for p in neither]},
            "n_total": K,
        },
        "gain_vs_noise": {
            "papers_with_numeric_isolated_gain": {"n": len(quant), "ids": [p["id"] for p in quant]},
            "papers_gain_below_noise": _prop(len(below), len(quant))
            | {"ids": [p["id"] for p in below]},
            "dataset_level_deltas_n": len(all_deltas),
            "dataset_level_below_noise": _prop(len(below_d), len(all_deltas)),
            "all_deltas_pts": all_deltas,
        },
    }

    # ---- write CSV
    cols = ["id", "cite", "year", "subdomain", "status", "exclude_reason", "method", "dataset",
            "reports_multirun", "multirun_form", "n_runs", "reported_range_pts",
            "has_attn_ablation", "attn_gain_pts", "attn_gain_all", "gain_is_headline",
            "gain_below_seed_noise", "headline_acc",
            "code_public", "code_official", "weights_public", "dataset_obtainable", "stat_test",
            "src_multirun", "src_ablation", "src_code", "src_weights", "notes"]
    with open("corpus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in PAPERS:
            r = dict(p)
            g = r.get("attn_gain_pts")
            r["gain_below_seed_noise"] = (g < SEED_NOISE_PTS) if isinstance(g, (int, float)) else ""
            if isinstance(r.get("attn_gain_all"), list):
                r["attn_gain_all"] = ";".join(str(x) for x in r["attn_gain_all"])
            if isinstance(r.get("per_run_values"), list):
                r.setdefault("notes", "")
            w.writerow(r)

    with open("corpus.json", "w", encoding="utf-8") as f:
        json.dump({"protocol": PROTOCOL, "papers": PAPERS}, f, ensure_ascii=False, indent=2)
    with open("stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ---- console report
    P = stats["proportions_of_K"]
    print(f"screened={len(PAPERS)}  excluded={len(exc)}  unverifiable={len(unv)}  INCLUDED K={K}\n")
    for k, v in P.items():
        ci = f"[{v['lo'] * 100:4.1f}, {v['hi'] * 100:5.1f}]"
        print(f"  {k:34s} {v['k']:2d}/{v['n']:2d} = {v['p'] * 100:5.1f}%  95%CI {ci}")
    ct = stats["crosstab_2x2_variance_x_ablation"]
    print(f"\n  2x2  both={ct['both']['n']} var_only={ct['variance_only']['n']} "
          f"abl_only={ct['ablation_only']['n']} neither={ct['neither']['n']}")
    gv = stats["gain_vs_noise"]
    b = gv["papers_gain_below_noise"]
    d = gv["dataset_level_below_noise"]
    ci_b = f"[{b['lo'] * 100:.1f}, {b['hi'] * 100:.1f}]"
    print(f"  gain<{SEED_NOISE_PTS}pt (paper-level  ) {b['k']}/{b['n']}"
          f" = {b['p'] * 100:.1f}%  95%CI {ci_b}")
    ci_d = f"[{d['lo'] * 100:.1f}, {d['hi'] * 100:.1f}]"
    print(f"  gain<{SEED_NOISE_PTS}pt (dataset-level) {d['k']}/{d['n']}"
          f" = {d['p'] * 100:.1f}%  95%CI {ci_d}")
    print("\nwrote corpus.csv corpus.json stats.json")


if __name__ == "__main__":
    main()
