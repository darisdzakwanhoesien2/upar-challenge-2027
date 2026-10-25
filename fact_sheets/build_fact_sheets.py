#!/usr/bin/env python3
"""Fill the organizers' UPAR 2027 fact-sheet template (main.tex) once per track.

    python fact_sheets/build_fact_sheets.py --template DIR_WITH_main.tex --out OUT_DIR [--repo-url URL] [--results-dir DIR --leaderboard-prefix PREFIX --username U --affiliation A --email E --team T --paper P]

The template (a zip from the organizers: main.tex, ieeeconf.cls, IEEEtran.cls/.bst, references.bib) is not stored in this repository.
Items that only the team can know (Codabench user name, affiliation, email, leaderboard numbers, workshop paper) are left as
clearly marked TODO fields. Each track gets its own folder with .tex, .bib and the class files, ready to compile with
pdflatex + bibtex + pdflatex x2.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

TODO = r"\textbf{\textcolor{red}{[TODO: %s]}}"
X, O = "(X)", "(~)"


def todo(s: str) -> str:
    return TODO % s


TRACKS = {
    1: dict(
        key="Track 1", folder="track1_attribute_recognition",
        title="Frozen ImageNet ResNet-18 features with a class-weighted linear head for pedestrian attribute recognition",
        summary=r"""A frozen, ImageNet-pretrained ResNet-18 turns each 256$\times$128 pedestrian crop into a 512-dimensional feature vector. A linear layer with
one sigmoid output per attribute, trained with a per-attribute positive-class weight on the organizers' training images, predicts the 40
attribute probabilities. Nothing is fine-tuned, no augmentation is used, and the decision threshold stays at the required 0.5. The contribution
is a deliberately simple and fully reproducible baseline, not a state-of-the-art system.""",
        method=r"""\paragraph{Pre-processing} Every crop is converted to RGB, resized to $256\times128$ (height $\times$ width) with bilinear interpolation, scaled to
$[0,1]$ and normalised with the ImageNet mean and standard deviation.

\paragraph{Backbone} \texttt{torchvision} ResNet-18 with the \texttt{IMAGENET1K\_V1} weights, final classifier removed, global average pooling, giving a
512-d feature. The backbone is frozen: it is never updated, and features are extracted once and cached.

\paragraph{Head} Features are standardised with the mean and standard deviation of the training set (stored with the weights), then a linear layer
$512\rightarrow40$ gives logits; the attribute probability is the sigmoid of the logit.

\paragraph{Loss and training} Binary cross-entropy with logits and a per-attribute positive weight $w_a=\mathrm{clip}\big((n^-_a/n^+_a)^{0.5},\,1,\,20\big)$, where
$n^+_a$ and $n^-_a$ count the positive and negative training examples of attribute $a$. The square-root weighting compensates part of the class imbalance,
because the label-based metric mA weighs rare attributes as much as frequent ones. Optimiser AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, batch size 1024,
40 epochs, seed 0, all 97,669 training images (Market-1501, PA-100K and PETA in the organizers' release).

\paragraph{Model selection} The exponent of the class weight (0.25, 0.5, 0.75, 1.0) was chosen on a random subset of 10,000 images of the organizers' validation
split (seed 0), which was never used for training: challenge average 0.679, 0.703, 0.698 and 0.682 respectively. Per-attribute threshold calibration was tried
and rejected because it lowered the score when fitted on one dataset and tested on another.

\paragraph{Inference} Probabilities are returned for the 40 attributes in the required order; a batch of images is processed per call. No post-processing.
Training takes about 35 minutes of CPU feature extraction on one Apple-silicon laptop plus seconds for the head; the code is deterministic and retraining from cached
features reproduces the submitted weights byte for byte.""",
        results_cols=["Avg", "mA", "Label\\_F1", "Inst\\_Acc", "Inst\\_Prec", "Inst\\_Rec", "Inst\\_F1"],
        local=r"""Local validation check (own implementation of the formulas on the Codabench Evaluation page, 10,000 random validation images; the validation
split comes from the same three datasets as training, so it is an in-domain estimate): challenge average 0.703 (mA 0.686, instance F1 0.720), against 0.548
for the attribute-prior baseline; averaged over the three datasets 0.690 against 0.546.""",
        remarks=r"""\textbf{Pros:} very small and fast (25~ms per image on CPU), fully deterministic, no external data, trivial to reproduce. \textbf{Cons:} the backbone is
not adapted to pedestrians or to the target domain, so accuracy on the hidden domain is limited; rare attributes (for example glasses, hats, unusual colours)
are weak; no augmentation, ensembling or domain-generalisation technique was used. A fine-tuned or larger backbone would be the obvious next step but needs a GPU
that was not available to us.""",
        trained="Yes", pretrained=r"""Yes. ResNet-18, \texttt{torchvision} \texttt{IMAGENET1K\_V1} weights (supervised ImageNet-1k classification). Used as a frozen feature extractor.""",
        external=r"No. Only ImageNet pre-training of the backbone; no other data.",
        aug=r"No.", valid=r"No. 10,000 random validation images were used only to choose the class-weight exponent; the validation split was not used for training.",
        reg=r"Yes. Weight decay $10^{-4}$ (AdamW) and a per-attribute positive-class weight to counter class imbalance; the backbone is frozen.",
        pre=r"Yes. Resize to $256\times128$ and ImageNet normalisation.", post=r"No. The threshold is the required 0.5.", vlm="No", vlm_text="",
    ),
    2: dict(
        key="Track 2", folder="track2_attribute_retrieval",
        title="Attribute-probability retrieval with a likelihood distance on frozen ResNet-18 features",
        summary=r"""The gallery images are scored by the same frozen ImageNet ResNet-18 with a class-weighted linear head used for Track 1. Each gallery image gets 40
attribute probabilities. A query is a binary 40-vector, and the gallery is ranked by the negative log-likelihood of the query under the image's independent
attribute probabilities. This replaces the organizers' L1 distance and improved our local retrieval score on every dataset.""",
        method=r"""\paragraph{Attribute model} Identical to our Track 1 entry: $256\times128$ crops, ImageNet normalisation, frozen \texttt{torchvision} ResNet-18
(\texttt{IMAGENET1K\_V1}), feature standardisation, a linear layer $512\rightarrow40$ trained with binary cross-entropy and a per-attribute positive weight
$\mathrm{clip}((n^-/n^+)^{0.5},1,20)$, AdamW (learning rate $10^{-3}$, weight decay $10^{-4}$), 40 epochs, batch size 1024, seed 0, on all 97,669 training images.
No training data of the retrieval task (query/identity files) is used; the model only sees attribute labels.

\paragraph{Ranking} For a query $q\in\{0,1\}^{40}$ and a gallery image with probabilities $p\in(0,1)^{40}$, the distance is
$d(q,p)=-\sum_{a}\big[q_a\log p_a+(1-q_a)\log(1-p_a)\big]$, with $p$ clipped to $[10^{-4},1-10^{-4}]$. The whole gallery is returned as a $Q\times G$ distance
matrix, smaller meaning a better match. The organizers' example uses $\sum_a|q_a-p_a|$ instead.

\paragraph{Choice of the distance} On 10,000 validation images with up to 1,500 queries per set, our own implementation of the page's formulas gave a mean average degree of
match (mADM) of 0.331 for the likelihood distance against 0.292 for L1 on all three datasets together, and the likelihood distance was better on each dataset
(Market-1501 0.331 vs 0.297, PA-100K 0.346 vs 0.296, PETA 0.288 vs 0.256). A prior-only ranking (all gallery images identical) reaches about 0.125.
The distance can be switched with the environment variable \texttt{UPAR\_DISTANCE} (\texttt{nll} or \texttt{l1}).""",
        results_cols=["mADM", "mAP", "Rank-1", "Rank-5", "Rank-10", "mINP"],
        local=r"""Local validation check (own implementation, in-domain, gallery of 10,000 validation images, up to 1,500 queries): mADM 0.331 for the likelihood distance, 0.292 for L1.
A correct match needs all 40 attributes to agree with the query, which is why Rank-1 is low (about 4\%).""",
        remarks=r"""\textbf{Pros:} reuses one small attribute model, no retrieval-specific training, a principled distance that uses the full probability instead of a hard threshold.
\textbf{Cons:} attributes are treated as independent, the backbone is frozen and not adapted to the hidden domain, and ties between near-identical gallery images are broken only by
gallery order. No re-ranking, ensembling or learned query embedding was used.""",
        trained="Yes", pretrained=r"""Yes. ResNet-18, \texttt{torchvision} \texttt{IMAGENET1K\_V1} weights (supervised ImageNet-1k), frozen.""",
        external=r"No. Only ImageNet pre-training of the backbone; no other data.",
        aug=r"No.", valid=r"No. 10,000 random validation images were used only to choose the class-weight exponent and to compare the two distances; the validation split was not used for training.",
        reg=r"Yes. Weight decay $10^{-4}$ and a per-attribute positive-class weight; the backbone is frozen.",
        pre=r"Yes. Resize to $256\times128$ and ImageNet normalisation.", post=r"Yes. Probabilities are clipped to $[10^{-4},1-10^{-4}]$ before the likelihood distance; no re-ranking.", vlm="No", vlm_text="",
    ),
    3: dict(
        key="Track 3", folder="track3_pose_estimation",
        title="Zero-shot COCO Keypoint R-CNN per given box with a learned head-slot mapping",
        summary=r"""For each given person box we crop the image with padding, run \texttt{torchvision}'s Keypoint R-CNN (ResNet-50 FPN) with its public COCO weights, keep the detection that
best overlaps the box, and map the 17 COCO keypoints to the 17 PoseTrack18 slots. The two head slots that COCO does not have (head\_bottom and head\_top) come from a
six-parameter linear map of the predicted nose and shoulders, fitted on the organizers' training annotations. No pose network was trained or fine-tuned on the challenge data.""",
        method=r"""\paragraph{Detector and keypoints} \texttt{keypointrcnn\_resnet50\_fpn} with the \texttt{COCO\_V1} weights, box score threshold 0.05. For every provided box
$(x,y,w,h)$ the crop is $[x-0.15w,\,x+1.15w]\times[y-0.15h,\,y+1.15h]$ clipped to the image. The model runs on the crop; among its detections we keep the one with the
highest IoU with the provided box (in crop coordinates) if the IoU exceeds 0.3, and add the crop offset to its keypoints to get image coordinates.

\paragraph{Slot mapping} COCO slots 0 (nose), 3--4 (ears) and 5--16 (shoulders to ankles) are copied to the PoseTrack slots with the same meaning. head\_bottom
and head\_top are $\text{neck}+w_s\,[\,d_x,\,d_y,\,1\,]\,B$, where neck is the midpoint of the shoulders, $w_s$ the shoulder width, $(d_x,d_y)=(\text{nose}-\text{neck})/w_s$,
and $B$ a $3\times2$ matrix (one for each slot). The matrices were fitted by least squares on 80\% of 33,624 labelled people in the organizers' \texttt{train\_upar\_all.json}
(seed 0); on the held-out 20\% with ground-truth inputs the mean error is 0.098 shoulder widths for head\_bottom and 0.231 for head\_top (0.136 and 0.533 for our first geometric guess).
If the shoulder width is below 2 pixels, a fixed geometric rule is used instead.

\paragraph{Score and fallback} The per-person score is the detection score times the IoU. If no detection overlaps the box, a fixed standing-pose template scaled to the box is
returned with score 0.05, so the output is always valid. Keypoint visibility is set to 1.

\paragraph{No training} The only fitted quantities are the twelve numbers of the two matrices. The COCO weights are used as published.""",
        results_cols=["AP (mean)", "AP50", "AP\\_medium", "AP\\_large"],
        local=r"""Local check with \texttt{pycocotools} on 200 labelled PETA training images with their ground-truth boxes, assuming COCO-style OKS sigmas (0.035 for the two head slots; not the organizers' values):
AP 0.661 and AP50 0.942, against AP 0.625 with our first head-slot heuristic and 0.739 if the head slots were perfect. These are small, low-resolution pedestrian crops, not the
full-scene images of the hidden test sets, so the number only shows that the pipeline works.""",
        remarks=r"""\textbf{Pros:} no training, compliant with COCO-only pre-training, always returns a valid output, tiny code. \textbf{Cons:} not adapted to the target domains or to PoseTrack-style
annotations; the head slots are an approximation; inference is slow on CPU (about 3~s per box) and relies on the evaluation GPU; one box is handled at a time, so crowded scenes
are slow; there is no test-time augmentation or ensembling.""",
        trained="No", pretrained=r"""Yes. \texttt{torchvision} Keypoint R-CNN (ResNet-50 FPN) \texttt{COCO\_V1} weights, trained by the \texttt{torchvision} authors on COCO keypoints. Used as published, without fine-tuning.""",
        external=r"No. Only the COCO pre-training of the published model; the training annotations of the challenge were used only to fit the twelve-parameter head-slot map.",
        aug=r"No.", valid=r"No. The validation data of the challenge was not used.",
        reg=r"No. Nothing is trained.",
        pre=r"Yes. Crop each given box with 15\% padding on every side.", post=r"Yes. COCO-to-PoseTrack slot mapping, the head-slot linear map, and a template-pose fallback when no detection overlaps the box.", vlm="No", vlm_text="",
    ),
}

QUESTIONS = [
    (r"\textbf{For the competition track associated with this fact sheet, you confirm that you have trained your model on the predefined datasets}", "trained", "yesno"),
    (r"\textbf{Did you use any pre-trained model:}", "pretrained", "pre"),
    (r"\textbf{Did you use external data?}", "external", "yn_text"),
    (r"\textbf{Did you perform any data augmentation?}", "aug", "yn_text"),
    (r"\textbf{Did you use the provided validation set as part of your training set?}", "valid", "yn_text"),
    (r"\textbf{Did you use any regularization strategies/terms?}", "reg", "yn_text"),
    (r"\textbf{Did you use any pre-processing steps?}", "pre", "yn_text"),
    (r"\textbf{Did you use any post-processing steps?}", "post", "yn_text"),
    (r"\textbf{Did you use Vision Language Models (VLMs) of any kind?}", "vlm", "yn_text"),
]


def yn(text: str) -> tuple[str, str]:
    return (X, O) if text.strip().lower().startswith("yes") else (O, X)


def additional_section(t: dict) -> str:
    out = [r"\section{Additional method details}",
           r"Please, reply if your challenge entry considered (or not) the following strategies and provide a brief explanation. For each question, mark with X one single option. \\", "",
           r"\begin{itemize}"]
    for q, key, kind in QUESTIONS:
        val = t[key]
        if kind == "yesno":
            y, n = yn(val)
            out.append(rf"\item {q} {y}~Yes, {n}~No\\")
        elif kind == "pre":
            y, n = yn(val)
            out.append(rf"\item {q} {y}~Yes, {n}~No.\\ If yes, please detail (e.g., model architecture, training strategy, train data, etc.).\\ {val}\\")
        else:
            y, n = yn(val)
            detail = val.split(".", 1)[1].strip() if val.lower().startswith(("yes.", "no.")) and len(val.split(".", 1)[1].strip()) else ""
            out.append(rf"\item {q} {y}~Yes, {n}~No\\")
            if detail:
                out.append(rf"{'If yes, please detail' if y == X else 'Comment'}:\\ {detail}\\")
        out.append("")
    out.append(r"\end{itemize}")
    return "\n".join(out)


def read_results(results_dir: Path, username_prefix: str) -> dict[int, dict]:
    """Our row from each track's exported leaderboard CSV, with its rank and the best score. Files: 'Results - Evaluation Phase Track{1,2}.csv', '... Track 3.csv'."""
    import csv
    out = {}
    names = {1: ["Results - Evaluation Phase Track1.csv"], 2: ["Results - Evaluation Phase Track2.csv"], 3: ["Results - Evaluation Phase Track 3.csv", "Results - Evaluation Phase Track3.csv"]}
    for n, files in names.items():
        f = next((results_dir / x for x in files if (results_dir / x).exists()), None)
        if f is None:
            continue
        rows = list(csv.DictReader(open(f, newline="")))
        metric_cols = [c for c in rows[0] if c not in ("Username", "fact_sheet_answers")]
        short = {c: c.split(")-", 1)[-1] if ")-" in c else c for c in metric_cols}
        primary = metric_cols[0]
        ranked = sorted(rows, key=lambda r: -float(r[primary]))
        mine = [i for i, r in enumerate(ranked) if r["Username"].startswith(username_prefix)]
        if not mine:
            continue
        r = ranked[mine[0]]
        out[n] = dict(values={short[c]: float(r[c]) for c in metric_cols}, rank=mine[0] + 1, teams=len(rows), best=float(ranked[0][primary]),
                      primary=short[primary], entry=r["Username"])
    return out


def results_table(t: dict, res: dict | None = None) -> str:
    """One row per leaderboard column (a 7-column table does not fit the narrow two-column layout)."""
    intro = r"Add a table with your obtained results, shown in the leaderboard of the challenge." "\n\n"
    if res:
        rows = "\n".join(rf"{c.replace('_', chr(92) + '_')} & {res['values'][c]:.4f} \\" for c in res["values"])
        lead = (rf"\textbf{{Leaderboard results of our last submission, {t['key']}}} (entry \texttt{{{res['entry'].replace('_', chr(92) + '_')}}}; the ranking metric is {res['primary'].replace('_', chr(92) + '_')}). "
                rf"On the exported results our entry is ranked {res['rank']} of {res['teams']} teams; the best score is {res['best']:.4f}." "\n\n")
    else:
        rows = "\n".join(rf"{c} & \textcolor{{red}}{{TODO}} \\" for c in t["results_cols"])
        lead = rf"\textbf{{Leaderboard results of our last submission, {t['key']}}} (copy the values shown on Codabench; every red TODO must be replaced):" "\n\n"
    return (intro + lead + r"\begin{center}\small" "\n" r"\begin{tabular}{lc}" "\n" r"\hline" "\n" r"Metric & Ours \\" "\n" r"\hline" "\n"
            + rows + "\n" r"\hline" "\n" r"\end{tabular}" "\n" r"\end{center}" "\n\n" + t["local"] + "\n")


def gap_text(n: int, res: dict | None) -> str:
    if not res:
        return ""
    name, local = LOCAL[n]
    got = res["values"][res["primary"]]
    extra = {1: r" The validation images come from the same three datasets as the training images, whereas the hidden test is a new domain.",
             2: r" The validation gallery comes from the training datasets, whereas the hidden gallery is a new domain.",
             3: r" The local check used small pedestrian crops with their ground-truth boxes, not full-scene images."}[n]
    return (rf"\textbf{{Gap to the hidden test.}} Our local {name} estimate was {local:.3f}; the hidden test gives {got:.3f} and a rank of {res['rank']} of {res['teams']}." + extra +
            r" We did not investigate the cause further, and our entry is a simple baseline rather than a competitive system." "\n\n")


LOCAL = {1: ("challenge average", 0.703), 2: ("mADM", 0.331), 3: ("AP", 0.661)}


def build(template: Path, out: Path, repo_url: str, info: dict, results: dict) -> None:
    base = (template / "main.tex").read_text()
    val = lambda key, label: info.get(key) or todo(label)
    for n, t in TRACKS.items():
        d = out / t["folder"]
        d.mkdir(parents=True, exist_ok=True)
        s = base
        s = s.replace(r"\item Team leader name: ", r"\item Team leader name: Daris Dzakwan Hoesien")
        s = s.replace(r"\item Username(s) on Codabench:", r"\item Username(s) on Codabench: " + val("username", "Codabench user name"))
        s = s.replace(r"\item Team leader affiliation: ", r"\item Team leader affiliation: " + val("affiliation", "affiliation"))
        s = s.replace(r"\item Team leader email: ", r"\item Team leader email: " + val("email", "email"))
        s = s.replace(r"\item Name of other team members (and affiliation): ", r"\item Name of other team members (and affiliation): " + val("team", "other team members, or 'none'"))
        s = s.replace(r"\item Team website URL (if any): ", r"\item Team website URL (if any): none")
        for i in (1, 2, 3):
            mark = "X" if i == n else " "
            s = re.sub(rf"\\item \( \) Track {i}:", rf"\\item ({mark}) Track {i}:", s)
        s = s.replace("Provide a brief summary of the method and contributions.\n", t["title"] + r"\\" + "\n\n" + t["summary"] + "\n")
        s = s.replace("An image (or several images) to support method description to better understand the approach and model pipeline. You can refer to these images in the method description part.\n",
                      "\\begin{center}\\small\n\\fbox{\\parbox{0.92\\linewidth}{\\centering " + {
                          1: r"crop $\rightarrow$ resize $256{\times}128$ + ImageNet normalisation $\rightarrow$ frozen ResNet-18 $\rightarrow$ 512-d feature $\rightarrow$ standardise $\rightarrow$ linear $512{\to}40$ $\rightarrow$ sigmoid $\rightarrow$ 40 probabilities ($>0.5$ = present)",
                          2: r"gallery crop $\rightarrow$ frozen ResNet-18 + linear head $\rightarrow$ 40 probabilities $p$; query $q$ (40 bits) $\rightarrow$ distance $-\sum[q\log p+(1{-}q)\log(1{-}p)]$ $\rightarrow$ gallery ranked per query",
                          3: r"image + given box $\rightarrow$ padded crop $\rightarrow$ Keypoint R-CNN (COCO weights) $\rightarrow$ best-overlap detection $\rightarrow$ COCO$\to$PoseTrack slots + learned head\_bottom/head\_top map $\rightarrow$ 17 keypoints + score",
                      }[n] + "}}\n\\end{center}\n")
        s = s.replace("Provide a technical and detailed description of the method and contributions. The explanations must be self-contained and one must be able to reproduce the approach by reading this section. You can explain and justify the approach by any means, e.g., citations~\\cite{Cormier_2023_WACV,Specker_2023_WACV,cormier2024upar}, equations, tables, algorithms, platforms and code libraries utilised, etc. We expect a detailed explanation of the architecture, preprocessing/postprocessing steps, loss function, training/inference details, and hyper-parameters.\n",
                      "The method is described below; the code and exact commands are in the repository linked at the end. Related work: UPAR~\\cite{Cormier_2023_WACV,Specker_2023_WACV,cormier2024upar}.\n\n" + t["method"] + "\n")
        s = s.replace("Add a table with your obtained results, shown in the leaderboard of the challenge.\n", results_table(t, results.get(n)))
        s = s.replace("Please identify the pros and cons (if any) of the proposed approach.\n", t["remarks"] + "\n\n" + gap_text(n, results.get(n)) + r"\textbf{Workshop paper:} " + val("paper", "state whether a paper was submitted to the RWS workshop, with its title or OpenReview number; write 'none' otherwise") + "\n")
        a = s.index(r"\section{Additional method details}"); b = s.index(r"\section{Code repository}")
        s = s[:a] + additional_section(t) + "\n\n\n\n" + s[b:]
        s = s.replace(r"\textbf{Code repository:} \url{http://your_link_here}", r"\textbf{Code repository:} \url{" + repo_url + "} (inference and training code, Dockerfile, instructions in the README; trained weights are attached to the repository's release, with SHA-256 sums).")
        (d / "main.tex").write_text(s)
        for f in ("references.bib", "ieeeconf.cls", "IEEEtran.cls", "IEEEtran.bst"):
            if (template / f).exists():
                shutil.copy(template / f, d / f)
        print("wrote", d / "main.tex")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-url", default="https://github.com/darisdzakwanhoesien2/upar-challenge-2027")
    ap.add_argument("--results-dir", type=Path, help="folder with the exported leaderboard CSVs (kept out of the repository: they list other teams)")
    ap.add_argument("--leaderboard-prefix", default="darisdzakwan-", help="start of our entry's Username in the exported CSVs")
    ap.add_argument("--username", help="Codabench user name for the Team details")
    for k in ("affiliation", "email", "team", "paper"):
        ap.add_argument(f"--{k}")
    a = ap.parse_args()
    results = read_results(a.results_dir, a.leaderboard_prefix) if a.results_dir else {}
    build(a.template, a.out, a.repo_url, {k: getattr(a, k) for k in ("username", "affiliation", "email", "team", "paper") if getattr(a, k)}, results)
