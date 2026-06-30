"""Command-line entry point. One subcommand per pipeline stage.

Every stage reads explicit input paths and writes to an explicit output path;
there are no hidden defaults. Stage implementations are imported lazily inside
each handler so that ``--help`` works without importing model or plotting
dependencies.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="symptommonster",
        description="Pharmacovigilance signal detection from paired clinical notes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract symptom mentions from masked notes.")
    extract.add_argument("--notes", required=True, help="Note source: a directory tree or a .jsonl export.")
    extract.add_argument("--alias-map", required=True, dest="alias_map", help="JSON map of real drug name -> token.")
    extract.add_argument("--prompt", default=None, help="Override the packaged extraction template.")
    extract.add_argument("--backend", default="ollama", choices=["ollama", "mlx"])
    extract.add_argument("--model", required=True, help="Opaque model id passed to the backend.")
    extract.add_argument("--out", required=True, help="Output extractions JSONL.")
    extract.add_argument("--limit", type=int, default=None)
    extract.add_argument("--seed", type=int, default=0)
    extract.add_argument("--temperature", type=float, default=0.05)
    extract.add_argument(
        "--num-ctx",
        type=int,
        default=131072,
        dest="num_ctx",
        help="Ollama context window in tokens; ignored by the MLX backend.",
    )
    extract.add_argument("--resume", action="store_true", help="Skip patients already in --out.")
    extract.set_defaults(func=_cmd_extract)

    normalize = sub.add_parser("normalize", help="Map raw terms to a canonical vocabulary.")
    nsub = normalize.add_subparsers(dest="subcommand", required=True)
    build = nsub.add_parser("build", help="Build a raw->canonical mapping over extractions.")
    build.add_argument("--in", nargs="+", required=True, dest="inputs", help="One or more extractions JSONL files.")
    build.add_argument("--ctcae", default=None, help="JSON list of preferred terms (tier 1).")
    build.add_argument("--rules", default=None, help="Override the packaged tier-2 synonym rules (TOML).")
    build.add_argument("--backend", default="ollama", choices=["ollama", "mlx"])
    build.add_argument("--model", default=None, help="Model for tier-3 residual grouping; tier 3 is skipped if unset.")
    build.add_argument("--out-mapping", required=True, dest="out_mapping", help="Reviewable mapping CSV.")
    build.set_defaults(func=_cmd_normalize_build)
    apply = nsub.add_parser("apply", help="Apply a reviewed mapping to extractions.")
    apply.add_argument("--in", required=True, dest="input", help="Extractions JSONL.")
    apply.add_argument("--mapping", required=True, help="Mapping CSV from `normalize build`.")
    apply.add_argument("--out", required=True, help="Normalized records JSONL.")
    apply.set_defaults(func=_cmd_normalize_apply)

    noise = sub.add_parser("noise", help="Build scrambled-surrogate pre/post pairs (the empirical null).")
    noise.add_argument("--notes", required=True, help="Note source to draw surrogate pre-treatment notes from.")
    noise.add_argument("--seed", type=int, default=0)
    noise.add_argument("--out", required=True, help="Surrogate note pairs JSONL (feed back through `extract`).")
    noise.set_defaults(func=_cmd_noise)

    stats = sub.add_parser("stats", help="Per-pair paired t-test with BH-FDR.")
    stats.add_argument("--signal", required=True, help="Normalized signal records JSONL.")
    stats.add_argument("--noise", required=True, help="Normalized noise records JSONL.")
    stats.add_argument("--out-dir", required=True, dest="out_dir")
    stats.add_argument("--fdr-q", type=float, default=0.10, dest="fdr_q")
    stats.add_argument("--alpha", type=float, default=0.05)
    stats.add_argument(
        "--scope",
        default="within-drug",
        choices=["within-drug", "global"],
        help="FDR family: per-group (primary) or all pairs pooled (sensitivity).",
    )
    stats.add_argument(
        "--covariates",
        default=None,
        help="Optional CSV keyed by patient_id; adds the covariate-adjusted residual test.",
    )
    stats.add_argument("--seed", type=int, default=0)
    stats.set_defaults(func=_cmd_stats)

    bayes = sub.add_parser("bayes", help="Beta-Binomial posterior P(signal > noise) per pair.")
    bayes.add_argument("--signal", required=True)
    bayes.add_argument("--noise", required=True)
    bayes.add_argument("--out-dir", required=True, dest="out_dir")
    bayes.add_argument("--draws", type=int, default=10000)
    bayes.add_argument("--threshold", type=float, default=0.95)
    bayes.add_argument("--seed", type=int, default=0)
    bayes.set_defaults(func=_cmd_bayes)

    subgroup = sub.add_parser("subgroup", help="Stratify top signals by covariate; test heterogeneity.")
    subgroup.add_argument("--signal", required=True)
    subgroup.add_argument("--noise", required=True)
    subgroup.add_argument("--covariates", required=True, help="CSV keyed by patient_id with covariate columns.")
    subgroup.add_argument("--strata", default="institution,sex,age", help="Comma-separated covariate columns.")
    subgroup.add_argument("--top-from", default=None, dest="top_from", help="Summary CSV selecting which pairs to test.")
    subgroup.add_argument("--out", required=True)
    subgroup.set_defaults(func=_cmd_subgroup)

    benchmark = sub.add_parser("benchmark", help="Evaluate model extractions against annotator ground truth.")
    benchmark.add_argument("--extractions", required=True, help="Directory with one normalized JSONL per model.")
    benchmark.add_argument("--annotations", required=True, help="Directory with one CSV per annotator.")
    benchmark.add_argument("--gt-mode", default="union", choices=["union", "majority", "intersection"], dest="gt_mode")
    benchmark.add_argument("--out", required=True)
    benchmark.set_defaults(func=_cmd_benchmark)

    reference = sub.add_parser("reference", help="Parse reference rate dumps; join to observed signals.")
    reference.add_argument("--dumps", required=True, help="Directory of per-group reference rate JSON files.")
    reference.add_argument("--signal", required=True, help="Signal summary CSV.")
    reference.add_argument("--out", required=True)
    reference.set_defaults(func=_cmd_reference)

    figures = sub.add_parser("figures", help="Render figures from documented-format tables.")
    figures.add_argument("--which", default="all", help="Figure name or 'all'.")
    figures.add_argument("--signal", default=None)
    figures.add_argument("--benchmark", default=None)
    figures.add_argument("--reference", default=None)
    figures.add_argument("--matrix", default=None, help="Drug-by-symptom matrix CSV for the dendrogram.")
    figures.add_argument("--out-dir", required=True, dest="out_dir")
    figures.set_defaults(func=_cmd_figures)

    tables = sub.add_parser("tables", help="Generate manuscript tables from documented-format inputs.")
    tables.add_argument("--signal", required=True)
    tables.add_argument("--demographics", default=None)
    tables.add_argument("--out-dir", required=True, dest="out_dir")
    tables.set_defaults(func=_cmd_tables)

    return parser


def _cmd_extract(a: argparse.Namespace) -> None:
    from symptommonster.extract.runner import run_extract

    run_extract(
        notes=a.notes,
        alias_map=a.alias_map,
        prompt=a.prompt,
        backend=a.backend,
        model=a.model,
        out=a.out,
        limit=a.limit,
        seed=a.seed,
        temperature=a.temperature,
        num_ctx=a.num_ctx,
        resume=a.resume,
    )


def _cmd_normalize_build(a: argparse.Namespace) -> None:
    from symptommonster.normalize.pipeline import run_build

    run_build(
        inputs=a.inputs,
        ctcae=a.ctcae,
        rules=a.rules,
        backend=a.backend,
        model=a.model,
        out_mapping=a.out_mapping,
    )


def _cmd_normalize_apply(a: argparse.Namespace) -> None:
    from symptommonster.normalize.pipeline import run_apply

    run_apply(input=a.input, mapping=a.mapping, out=a.out)


def _cmd_noise(a: argparse.Namespace) -> None:
    from symptommonster.noise.surrogate import run_noise

    run_noise(notes=a.notes, seed=a.seed, out=a.out)


def _cmd_stats(a: argparse.Namespace) -> None:
    from symptommonster.stats.run import run_stats

    run_stats(
        signal=a.signal,
        noise=a.noise,
        out_dir=a.out_dir,
        fdr_q=a.fdr_q,
        alpha=a.alpha,
        scope=a.scope,
        covariates=a.covariates,
        seed=a.seed,
    )


def _cmd_bayes(a: argparse.Namespace) -> None:
    from symptommonster.stats.run import run_bayes

    run_bayes(
        signal=a.signal,
        noise=a.noise,
        out_dir=a.out_dir,
        draws=a.draws,
        threshold=a.threshold,
        seed=a.seed,
    )


def _cmd_subgroup(a: argparse.Namespace) -> None:
    from symptommonster.subgroup.run import run_subgroup

    run_subgroup(
        signal=a.signal,
        noise=a.noise,
        covariates=a.covariates,
        strata=a.strata,
        top_from=a.top_from,
        out=a.out,
    )


def _cmd_benchmark(a: argparse.Namespace) -> None:
    from symptommonster.benchmark.run import run_benchmark

    run_benchmark(extractions=a.extractions, annotations=a.annotations, gt_mode=a.gt_mode, out=a.out)


def _cmd_reference(a: argparse.Namespace) -> None:
    from symptommonster.reference.run import run_reference

    run_reference(dumps=a.dumps, signal=a.signal, out=a.out)


def _cmd_figures(a: argparse.Namespace) -> None:
    from symptommonster.figures.run import run_figures

    run_figures(
        which=a.which,
        signal=a.signal,
        benchmark=a.benchmark,
        reference=a.reference,
        matrix=a.matrix,
        out_dir=a.out_dir,
    )


def _cmd_tables(a: argparse.Namespace) -> None:
    from symptommonster.figures.run import run_tables

    run_tables(signal=a.signal, demographics=a.demographics, out_dir=a.out_dir)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
