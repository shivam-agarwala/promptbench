"""Benchmark prompting strategies on the support-triage task in dataset.csv.

    python evaluate.py --dry-run                    # mock model, no key, no cost
    python evaluate.py --list-providers             # what you can point it at
    python evaluate.py --limit 8                    # cheap live smoke test
    python evaluate.py --repeats 3 --concurrency 4

Writes three CSVs. predictions.csv is the one that matters; everything else can be
recomputed from it.
"""
import argparse
import csv
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from harness import CATEGORIES, STRATEGIES, SYSTEM, MockClient, ResponseCache
from metrics import (discordant_pairs, macro_f1, mcnemar_exact, mean_and_sd, per_class_prf,
                     percentile, wilson_ci)
from providers import PROVIDERS, build_provider_client

# Paths resolve against this file, not the cwd, so the script works from any directory.
HERE = os.path.dirname(os.path.abspath(__file__))

BUCKETS = ["wrong_category", "hallucinated_category", "malformed_output", "refusal_or_empty"]
STATUS_TO_BUCKET = {"hallucinated": "hallucinated_category", "malformed": "malformed_output",
                    "empty": "refusal_or_empty"}

# Matched against the error string, which begins with the exception class name. Both
# class names and server wording are listed; server wording varies by version.
ERROR_HINTS = [
    (("authenticationerror", "invalid x-api-key", "bad credentials", "401"),
     "The key was rejected. Check it for a typo, a stray quote or a trailing space."),
    (("notfounderror", "not_found", "unknown_model", "404"),
     "That model id doesn't exist for this key. Try one from the list below."),
    (("credit balance", "quota", "billing"),
     "The account has no credit. API billing is separate from any chat subscription."),
    (("permissiondenied", "forbidden", "403"),
     "The key is recognised but not allowed to do this. Usually the token is missing an "
     "inference permission, or the model needs terms accepted on its page. On Hugging "
     "Face the token needs 'Make calls to Inference Providers'. Run --list-models: if "
     "that also fails, it's the token; if it works, it's the model id."),
    (("ratelimit", "rate_limit", "429"),
     "Rate limited even after retries. Lower --concurrency or wait; the cache means "
     "completed calls aren't repaid."),
    (("402", "credit", "quota", "depleted", "payment"),
     "The account is out of inference credit. Top up, or switch provider: "
     "--provider ollama runs locally with no account at all."),
    (("410", "retired", "brownout", "no longer available"),
     "That service has been shut down. GitHub Models was retired on 30 July 2026. Pick "
     "another provider: python evaluate.py --list-providers"),
    (("urlerror", "connection", "timeout"),
     "Couldn't reach the endpoint. Check network, VPN, or that ollama serve is running."),
]


FATAL_SIGNS = ("402", "credit", "depleted", "quota", "401", "403", "410")
FATAL_STREAK = 4  # consecutive fatal-looking failures before we stop the whole run


class SweepAborted(Exception):
    """Raised when repeated unrecoverable errors mean the run is no longer collecting data."""


@dataclass
class Prediction:
    """One strategy's attempt at one item on one run. All reported metrics derive from
    these records, and the paired tests require them to be item-addressable."""
    run: int
    strategy: str
    item_id: str
    gold: str
    predicted: str
    status: str
    correct: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached: int
    attempts: int
    error: str
    raw_text: str


def load_dotenv(path):
    """Load KEY=value pairs from a .env file.

    Uses setdefault, so an existing environment variable takes precedence.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip("\"'")
            if value:
                os.environ.setdefault(key.strip(), value)


def load_dataset(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    missing = {"id", "message", "label"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path} is missing column(s): {sorted(missing)}")
    bad = [r for r in rows if r["label"] not in CATEGORIES]
    if bad:
        raise SystemExit(f"{len(bad)} row(s) labelled outside {CATEGORIES}, e.g. {bad[0]}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("dataset ids are not unique")  # would break the paired alignment
    return rows


def take_sample(rows, limit):
    """Return `limit` items balanced across classes.

    dataset.csv is grouped by label, so a plain slice would return a single class.
    Round-robin selection, deterministic.
    """
    if not limit or limit >= len(rows):
        return rows
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    picked, labels = [], list(by_label)
    while len(picked) < limit:
        before = len(picked)
        for lab in labels:
            if by_label[lab] and len(picked) < limit:
                picked.append(by_label[lab].pop(0))
        if len(picked) == before:
            break  # every class exhausted
    return sorted(picked, key=lambda r: int(r["id"]) if r["id"].isdigit() else r["id"])


def describe_classes(rows):
    counts = Counter(r["label"] for r in rows)
    return ", ".join(f"{k} {counts[k]}" for k in CATEGORIES if counts[k])


def preflight(client, model_tag):
    """Single probe call before the sweep. Exits on failure rather than running it."""
    probe = client.complete(SYSTEM, "Reply with the single word: ok", "zero_shot", "ok", 16)
    if not probe.error:
        return
    print(f"\nPreflight call to {model_tag} failed, so the sweep was not started.\n")
    print(f"  {probe.error}\n")
    lowered = probe.error.lower()
    for needles, hint in ERROR_HINTS:
        if any(n in lowered for n in needles):
            print(f"  Likely fix: {hint}\n")
            break
    if any(k in lowered for k in ("404", "403", "not_found", "notfound", "forbidden", "model")):
        models = client.list_models()
        if models:
            print("  Model ids this key can use:")
            for m in models:
                print(f"    {m}")
            print()
    print("  Everything except the live API still works: python evaluate.py --dry-run")
    raise SystemExit(1)


def run_pass(name, client, rows, run_idx, concurrency, model_tag, temperature, guard):
    strat = STRATEGIES[name]
    reset = getattr(client, "reset", None)
    if reset:
        reset(run_idx)  # mock only

    def one(row):
        if guard["stop"]:
            raise SweepAborted(guard["reason"])
        user = strat.build(row["message"])
        # run_idx is part of the key, so --repeats re-samples rather than replaying
        # a cached response and reporting sd=0.
        key = ResponseCache.key(model_tag, temperature, name, strat.max_tokens,
                                strat.stop, run_idx, user)
        c = client.complete(SYSTEM, user, name, row["message"], strat.max_tokens, key,
                            strat.stop)
        # Abort on a streak of unrecoverable errors rather than emitting a table of zeros.
        if c.error and any(s in c.error.lower() for s in FATAL_SIGNS):
            guard["streak"] += 1
            if guard["streak"] >= FATAL_STREAK:
                guard["stop"], guard["reason"] = True, c.error
        elif not c.error:
            guard["streak"] = 0
        label, status = strat.parse(c.text)
        gold = row["label"]
        return Prediction(
            run=run_idx, strategy=name, item_id=row["id"], gold=gold,
            predicted=label or "", status=status,
            correct=int(status == "ok" and label == gold),
            latency_ms=round(c.latency_ms, 1), input_tokens=c.input_tokens,
            output_tokens=c.output_tokens, cached=int(c.cached), attempts=c.attempts,
            error=c.error, raw_text=c.text)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, rows))  # map preserves order
    return [one(r) for r in rows]


def summarize(name, preds, n_items):
    buckets = Counter()
    confusion = defaultdict(Counter)
    for p in preds:
        if p.status == "ok":
            buckets["wrong_category"] += 1 - p.correct
            confusion[p.gold][p.predicted] += 1
        else:
            # No usable prediction: INVALID column, not a category cell.
            buckets[STATUS_TO_BUCKET[p.status]] += 1
            confusion[p.gold]["INVALID"] += 1

    # Latency from live successful calls only. Cached hits are instant; failed calls
    # include retry sleeps.
    lat = [p.latency_ms for p in preds if not p.cached and not p.error]
    per_run = [sum(p.correct for p in preds if p.run == r) / n_items
               for r in sorted({p.run for p in preds})]
    acc, acc_sd = mean_and_sd(per_run)
    # Interval uses n_items, not items x repeats: repeats of the same items are not
    # independent observations of the population.
    lo, hi = wilson_ci(round(acc * n_items), n_items)
    prf = per_class_prf(confusion, CATEGORIES)
    out = {
        "strategy": name, "n_items": n_items, "runs": len(per_run),
        "accuracy": acc, "accuracy_sd": acc_sd, "ci_low": lo, "ci_high": hi,
        "macro_f1": macro_f1(prf),
        "mean_ms": sum(lat) / len(lat) if lat else 0.0, "p95_ms": percentile(lat, 0.95),
        "input_tokens": sum(p.input_tokens for p in preds),
        "output_tokens": sum(p.output_tokens for p in preds),
        "api_calls": sum(1 for p in preds if not p.cached),
        "cache_hits": sum(p.cached for p in preds),
        "retries": sum(p.attempts - 1 for p in preds),
        "errors": sum(1 for p in preds if p.error),
        **{b: buckets[b] for b in BUCKETS},
        "_confusion": confusion, "_prf": prf,
    }
    out["unparseable"] = sum(out[b] for b in BUCKETS if b != "wrong_category")
    return out


def add_cost(summary, price_in, price_out):
    # Prices are supplied by the caller. A hardcoded table would go stale silently.
    if price_in is None or price_out is None:
        summary["cost_usd"] = ""
        return
    summary["cost_usd"] = round(summary["input_tokens"] / 1e6 * price_in
                                + summary["output_tokens"] / 1e6 * price_out, 6)


def pairwise_tests(by_strategy, run_idx=0):
    """McNemar for each pair of strategies, on a single run.

    One run only: repeats of the same items are not independent and pooling them would
    inflate discordant counts. Items are aligned by id, not list position.

    Items where either strategy's call failed are excluded. A transport failure is
    missing data; scoring it as incorrect produces spurious significance.
    """
    names = list(by_strategy)
    vectors, failed = {}, {}
    for n in names:
        run = [p for p in by_strategy[n] if p.run == run_idx]
        vectors[n] = {p.item_id: p.correct for p in run}
        failed[n] = {p.item_id for p in run if p.error}
    all_ids = set.intersection(*(set(v) for v in vectors.values()))
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            usable = sorted(all_ids - failed[a] - failed[b])
            only_a, only_b = discordant_pairs([vectors[a][k] for k in usable],
                                              [vectors[b][k] for k in usable])
            out.append({"strategy_a": a, "strategy_b": b, "a_right_b_wrong": only_a,
                        "b_right_a_wrong": only_b, "discordant": only_a + only_b,
                        "p_value": round(mcnemar_exact(only_a, only_b), 4),
                        "n_compared": len(usable),
                        "excluded_failed": len(all_ids) - len(usable)})
    return out


def _print_rows(headers, widths, rows):
    print("".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for r in rows:
        print("".join(str(c).ljust(w) for c, w in zip(r, widths)))


def print_summary(results, show_cost):
    headers = ["rank", "strategy", "acc", "95% CI", "sd", "macroF1", "mean_ms", "p95_ms",
               "wrong", "halluc", "malfm", "empty", "in_tok", "out_tok"]
    widths = [5, 18, 7, 16, 7, 9, 9, 9, 7, 8, 7, 7, 9, 9]
    if show_cost:
        headers.append("usd")
        widths.append(10)
    rows = []
    for i, r in enumerate(results, start=1):
        row = [i, r["strategy"], f"{r['accuracy']:.3f}",
               f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]", f"{r['accuracy_sd']:.3f}",
               f"{r['macro_f1']:.3f}", f"{r['mean_ms']:.0f}", f"{r['p95_ms']:.0f}"] \
              + [r[b] for b in BUCKETS] + [r["input_tokens"], r["output_tokens"]]
        if show_cost:
            row.append(r["cost_usd"])
        rows.append(row)
    print("\nSUMMARY  (ranked by accuracy, mean latency breaks ties)")
    _print_rows(headers, widths, rows)


def print_per_class(result):
    print(f"\nPer-class metrics: {result['strategy']}")
    rows = [[c, f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}",
             v["support"], v["predicted"]] for c, v in result["_prf"].items()]
    _print_rows(["category", "prec", "recall", "f1", "gold_n", "pred_n"],
                [12, 8, 8, 8, 8, 8], rows)


def print_confusion(result):
    cols = CATEGORIES + ["INVALID"]
    print(f"\nConfusion matrix: {result['strategy']}  (rows = gold, cols = predicted)")
    print("  " + "gold".ljust(11) + "".join(c[:9].rjust(10) for c in cols))
    for gold in CATEGORIES:
        counts = result["_confusion"][gold]
        print("  " + gold.ljust(11) + "".join(str(counts.get(c, 0)).rjust(10) for c in cols))


def print_pairwise(pairs, n_items):
    print(f"\nPAIRED COMPARISONS  (exact McNemar on run 0, {n_items} items)")
    rows = [[p["strategy_a"], p["strategy_b"], p["n_compared"], p["a_right_b_wrong"],
             p["b_right_a_wrong"], p["discordant"], f"{p['p_value']:.4f}",
             "significant" if p["p_value"] < 0.05 else "not distinguishable"]
            for p in pairs]
    _print_rows(["strategy A", "strategy B", "n", "A only", "B only", "discord", "p",
                 "verdict"], [18, 18, 5, 8, 8, 9, 9, 20], rows)
    dropped = max((p["excluded_failed"] for p in pairs), default=0)
    if dropped:
        print(f"  {dropped} item(s) excluded from some pairs: the call never reached the "
              f"model, which is missing data rather than a wrong answer.")
    print("  'A only' = items A got right and B got wrong. Concordant items are excluded.")
    print("  Below about 6 discordant items no split can reach p < 0.05 at all, so small")
    print("  discordant counts mean the dataset can't separate the strategies.")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def show_providers():
    print("\nProviders. Pick one with --provider, override the model with --model.\n")
    for name, cfg in PROVIDERS.items():
        have = "set" if os.environ.get(cfg["key_env"]) else "NOT set"
        print(f"  {name:<12} env {cfg['key_env']:<20} ({have})")
        print(f"  {'':<12} default model: {cfg['model']}")
        print(f"  {'':<12} {cfg['note']}\n")
    print("  custom       --base-url URL --model ID [--api-key-env VAR]\n")


def build_client(args, cache):
    client = build_provider_client(args.provider, model=args.model, base_url=args.base_url,
                                   key_env=args.api_key_env, temperature=args.temperature,
                                   cache=cache)
    return client, f"{args.provider}:{client.model}"


def main():
    ap = argparse.ArgumentParser(description="Prompt strategy benchmark")
    ap.add_argument("--dry-run", action="store_true", help="mock client, no API calls")
    ap.add_argument("--dataset", default=os.path.join(HERE, "dataset.csv"))
    ap.add_argument("--provider", default="ollama",
                    help="ollama | openai | openrouter | huggingface | custom")
    ap.add_argument("--model", default=None, help="model id, defaults per provider")
    ap.add_argument("--base-url", help="--provider custom only")
    ap.add_argument("--api-key-env", help="--provider custom only: env var holding the key")
    ap.add_argument("--list-providers", action="store_true")
    ap.add_argument("--list-models", action="store_true",
                    help="ask the provider which model ids your key can use, then exit")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="cap items, 0 = all")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, ".cache"))
    ap.add_argument("--price-in", type=float, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, help="USD per million output tokens")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    load_dotenv(os.path.join(HERE, ".env"))
    if args.list_providers:
        show_providers()
        return

    if args.list_models:
        client, _ = build_client(args, None)
        models = client.list_models()
        print(f"\nModels available to your key on {args.provider}:\n")
        for m in models:
            print(f"  {m}")
        if models:
            print(f"\nPick one with --model. Listed does not always mean callable: some "
                  f"route through providers your account can't reach, which shows up as a "
                  f"403 on the first call.\n")
        else:
            print("  (the endpoint returned nothing)\n")
        return

    rows = take_sample(load_dataset(args.dataset), args.limit)
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for n in names:
        if n not in STRATEGIES:
            raise SystemExit(f"unknown strategy {n!r}, pick from {list(STRATEGIES)}")

    concurrency = args.concurrency
    if args.dry_run:
        client, model_tag = MockClient(), "mock"
        if concurrency > 1:
            # The mock's RNG is consumed in call order; threads would break reproducibility.
            print("note: forcing concurrency=1 for --dry-run to keep the mock reproducible")
            concurrency = 1
    else:
        cache = None if args.no_cache else ResponseCache(args.cache_dir)
        client, model_tag = build_client(args, cache)
    if concurrency > 1:
        print("note: concurrency > 1 makes the latency columns unreliable, since calls "
              "queue against each other.")

    if not args.dry_run:
        preflight(client, model_tag)

    total = len(rows) * len(names) * args.repeats
    print(f"{len(rows)} items ({describe_classes(rows)}) x {len(names)} strategies "
          f"x {args.repeats} run(s) = {total} calls against {model_tag}")

    by_strategy = defaultdict(list)
    guard = {"stop": False, "streak": 0, "reason": ""}
    try:
        for run_idx in range(args.repeats):
            for name in names:
                preds = run_pass(name, client, rows, run_idx, concurrency, model_tag,
                                 args.temperature, guard)
                by_strategy[name].extend(preds)
                failed = [p.error for p in preds if p.error]
                note = f", {len(failed)} call(s) failed after retries" if failed else ""
                print(f"  run {run_idx}  {name:<18} "
                      f"acc {sum(p.correct for p in preds) / len(preds):.3f}{note}")
                for msg in sorted(set(failed))[:3]:
                    print(f"      {msg[:200]}")
    except SweepAborted as stop:
        print(f"\nStopped after {FATAL_STREAK} consecutive failures of the same kind.\n")
        print(f"  {stop}\n")
        lowered = str(stop).lower()
        for needles, hint in ERROR_HINTS:
            if any(n in lowered for n in needles):
                print(f"  {hint}\n")
                break
        print("  Nothing was written. Partial results here would be worse than none: the\n"
              "  strategy that ran first keeps whatever quota was left, so the ranking\n"
              "  would reflect call order rather than prompt quality.")
        raise SystemExit(1)

    results = [summarize(n, by_strategy[n], len(rows)) for n in names]
    for r in results:
        add_cost(r, args.price_in, args.price_out)
    results.sort(key=lambda r: (-r["accuracy"], r["mean_ms"]))
    show_cost = args.price_in is not None and args.price_out is not None

    total_errors = sum(r["errors"] for r in results)
    if total_errors:
        print(f"\nWARNING: {total_errors} call(s) never reached the model. They score as "
              f"incorrect,\nso accuracy understates every affected strategy, and failures "
              f"rarely hit strategies\nevenly. Treat the ranking below as unusable until "
              f"the errors are fixed.")

    print_summary(results, show_cost)
    for r in results:
        print_per_class(r)
        print_confusion(r)
    pairs = pairwise_tests(by_strategy)
    print_pairwise(pairs, len(rows))
    if not show_cost:
        print("\nno cost column: pass --price-in and --price-out (USD per million tokens).")

    os.makedirs(args.outdir, exist_ok=True)
    summary_fields = ["strategy", "n_items", "runs", "accuracy", "accuracy_sd", "ci_low",
                      "ci_high", "macro_f1", "mean_ms", "p95_ms", "unparseable"] + BUCKETS \
                     + ["input_tokens", "output_tokens", "cost_usd", "api_calls",
                        "cache_hits", "retries", "errors"]
    write_csv(os.path.join(args.outdir, "results.csv"), results, summary_fields)
    write_csv(os.path.join(args.outdir, "predictions.csv"),
              [asdict(p) for name in names for p in by_strategy[name]],
              list(Prediction.__dataclass_fields__))
    write_csv(os.path.join(args.outdir, "pairwise.csv"), pairs,
              ["strategy_a", "strategy_b", "n_compared", "a_right_b_wrong",
               "b_right_a_wrong", "discordant", "p_value", "excluded_failed"])


if __name__ == "__main__":
    main()
