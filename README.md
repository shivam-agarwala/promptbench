# promptbench

![tests]([https://github.com/shivam-agarwala/promptbench/actions/workflows/tests.yml/badge.svg])

Benchmarks five prompting strategies on one classification task: 
Sorting (short customer support messages) into `billing`, `technical`, `account` or `other`.

The point isn't to produce a ranking. It's to produce a ranking you can believe.
Most prompt comparisons report two accuracy numbers and declare a winner. 
On 40 examples that's noise, and this repo says so in its own output.

## Quickstart

```bash
python evaluate.py --dry-run           # mock model, no key, no network, no cost
python -m unittest test_harness        # 52 tests, all offline
```

For real calls, the default provider is a local Ollama :

```bash
ollama serve                           # one terminal
ollama pull llama3.2                   # another
python evaluate.py --limit 8           # 40 calls, quick check
python evaluate.py --repeats 3         # full run, 600 calls
```

Hosted providers work too. `python evaluate.py --list-providers` shows each one and the
env var it reads. Export it, or put it in a `.env` file next to `evaluate.py`.

## Providers

Anything speaking the OpenAI `/chat/completions` format works through one client in
`providers.py`, including Ollama.

| `--provider` | Endpoint | Env var | Notes |
|---|---|---|---|
| `ollama` (default) | `localhost:11434/v1` | none | Local and free. `ollama serve`, then `ollama pull llama3.2`. |
| `openai` | `api.openai.com/v1` | `OPENAI_API_KEY` | Paid. |
| `openrouter` | `openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | Some ids have a `:free` suffix. |
| `huggingface` | `router.huggingface.co/v1` | `HF_TOKEN` | Monthly credits run out fast. |
| `custom` | `--base-url` | `--api-key-env` | vLLM, llama.cpp, LiteLLM, anything compatible. |

```bash
python evaluate.py --provider ollama --model mistral
python evaluate.py --provider openrouter --model meta-llama/llama-3.3-70b-instruct
python evaluate.py --provider custom --base-url http://localhost:8000/v1 --model my-model
```

## Files

```
harness.py        prompts, parsers, response cache, mock client
providers.py      one client for any OpenAI-compatible endpoint
metrics.py        Wilson intervals, exact McNemar, per-class P/R/F1
evaluate.py       run loop, aggregation, reporting
test_harness.py   52 tests
dataset.csv       40 labelled items, 10 per class
```

`evaluate.py` imports from the other three. None of them import each other or know about
the run loop. `metrics.py` does no I/O, which is why every statistic in it is testable.

## What it measures

Accuracy with a 95% Wilson interval, standard deviation across repeats, macro F1,
per-class precision and recall, mean and p95 latency, token counts, optional cost, a 4x5
confusion matrix, and four failure buckets:

- `wrong_category`: a valid category, wrong answer
- `hallucinated_category`: a short label outside the four allowed
- `malformed_output`: non-empty text with no readable decision in it
- `refusal_or_empty`: nothing came back, or the call died after every retry

Plus exact McNemar on every pair of strategies, which is the part most prompt benchmarks
skip.

## Results

See [RESULTS.md](RESULTS.md). Short version, llama3.2 via Ollama, 600 calls: the top three
strategies are statistically indistinguishable from each other, chain-of-thought costs 26x
the output tokens of the cheapest comparable strategy for no measurable gain, and prompt
format turns out to matter more than the examples themselves.

## Reading the output

```
rank strategy          acc    95% CI          sd     macroF1  wrong  halluc  malfm  empty
1    few_shot          0.750  [0.60, 0.86]    0.066  0.836    5      14      8      3
```

Read it right to left. The failure columns tell you *how* a strategy fails, which is more
actionable than the accuracy: `wrong` means the prompt isn't discriminating, `malfm` means
the output contract isn't being honoured, and those have different fixes. Then `sd` is
run-to-run noise. Then the interval. The point estimate is the least reliable number on
the line.

Check `pairwise.csv` before believing any ordering.

## Outputs

`predictions.csv` is the one to keep: one row per strategy per item per run, with the raw
model text, tokens, latency, retry count and failure status. Every table above is derived
from it, so you can re-cut the analysis without paying for the calls again.
`results.csv` and `pairwise.csv` are summaries.

## Why these statistics

**Paired testing.** Every strategy sees the same items, so what matters is the discordant
pairs: items A got right and B got wrong, and vice versa. Items both got right say nothing
about which is better. Comparing two independent accuracy figures throws the pairing away
and loses most of the power.

**Exact binomial, not chi-square.** The discordant count here is usually single digits,
where chi-square isn't reliable.

**Wilson, not Wald.** The textbook interval returns bounds outside [0,1] at small n and
has zero width at p=1, which would claim certainty from 40 examples.

**Unparseable responses count against recall.** They sit in an `INVALID` column, never in a
real category cell, so a strategy that breaks format can't hide in the confusion matrix.
They never count toward precision, since a broken response isn't a claim about a category.

**Latency skips cached hits and failed calls.** Cached hits are instant, failed calls carry
the retry sleeps.

**Failed calls are excluded from the paired tests.** A call that never reached the model is
missing data, not a wrong answer. Scoring it as wrong manufactures significant-looking
differences out of an outage, and because failures hit whichever strategy happens to run
when the quota dies, the effect looks exactly like a real result. The `n` column in the
pairwise table shows how many items each comparison actually used.

**The sweep aborts on repeated unfixable errors.** Four consecutive failures of the same
kind (402, 401, 403, 410) stop the run and write nothing. Partial output is worse than none
here: the strategy that ran first keeps whatever quota was left, so the ranking would
reflect call order rather than prompt quality.

**The interval is on 40, not 40 × repeats.** Rerunning the same items doesn't tell you
anything new about the population they came from.

## Provider history

This started against the Anthropic API, then GitHub Models, then the Hugging Face router,
and ended on a local Ollama. Not by plan:

- The Anthropic SDK build in use no longer accepted `temperature` on `messages.create`, so
  every call died on a `TypeError` before reaching the network.
- GitHub Models was retired on 30 July 2026, mid-project.
- Hugging Face monthly inference credits ran out partway through a sweep, which produced a
  false `p = 0.0156` because failed calls were being scored as wrong answers.

Each move cost one flag rather than a rewrite, because nothing above the client layer knows
which provider is behind it. The Anthropic-specific client was removed once Ollama became
the default: it needed an optional dependency and a separate code path for an API nothing
here uses any more. The third failure is the one that changed the design; see the note on
failed calls under "Why these statistics".

## A bug worth knowing about

`few_shot` originally scored 0.000 against llama3.2. The cause was not the prompt being
bad. The prompt ends with `Category:`, which invites the model to continue the pattern, so
it echoed the first demonstration back and started inventing new examples. The parser takes
the last line, which by then was garbage, and `max_tokens=16` truncated it mid-sentence.

Three faults stacked into one number that looked like a finding about prompting. It was
found by reading `raw_text` in `predictions.csv`, which is why that column exists. The fix
was stop sequences plus a larger token budget, and `few_shot_instruct` was added as a
control: same examples, instruction format, no completion trap. Comparing the two separates
"do examples help" from "does the format make the model run on".

## Limitations

**40 examples is a toy sample.** The interval at 85% accuracy spans about 20 points. A
strategy scoring 0.90 against one scoring 0.85 differs by two items.

**Below six discordant pairs, significance is arithmetically impossible.** No split of five
reaches p < 0.05 under a two-sided exact test. If the pairwise table shows small discordant
counts, the honest reading is "this dataset can't separate them", not "they're equivalent".
For gaps of a few points you'd want several hundred items.

**Run-to-run noise is bigger than you'd guess.** `--dry-run --repeats 3` produces an sd
around 0.066 from injected noise alone, which is larger than most gaps in the table.

**Temperature 0 is not a guarantee of determinism.** It happens to be deterministic on
llama3.2 through Ollama, which the `sd = 0.000` column confirms, but that's a property of
that setup rather than something the flag promises. Hosted endpoints also change models
under a fixed name over time. Check the `sd` column instead of assuming.

**Latency is wall clock from your machine**, including network and retry sleeps. Comparable
between strategies in one serial run, not across runs or regions. With `--concurrency > 1`
calls queue against each other and the latency columns stop meaning much; the harness warns.

**No price table.** Prices drift and a stale hardcoded one produces a confident wrong
number. Pass `--price-in` and `--price-out`, or get a blank column.

**Parser leniency is part of the benchmark.** `parse_json` strips markdown fences. That's a
judgement call that flatters `strict_json`. Change a parser and old results stop being
comparable.

**The mock isn't a model.** `--dry-run` is a keyword matcher with seeded failure injection.
A healthy dry run prints identical rows for all four strategies, because within a run they
meet the same failures and differ only by parser. Non-flat means a parser diverged, which
is a finding. Flat means the plumbing works and you've learned nothing about prompting.

**Single-label gold data caps everything.** "My account was suspended and I don't know why"
is filed as `account` and could reasonably be `billing`. Some of the residual error is
label disagreement, not model failure. Two annotators and a measured agreement rate would
separate the two. Not implemented.

**One task, one dataset.** Nothing here generalises. A ranking on support triage says
little about extraction or summarisation.

## When a live run fails

A live run makes one preflight call before the sweep. If it fails, it stops there and
prints the error, a likely fix, and, when the model id looks like the problem, the ids your
key can use. Usual causes: mistyped key, a model the key can't access, no API credit.

GitHub Models used to be the default provider here. It was retired on 30 July 2026, so it
has been removed. Providers disappear; that's part of why the client is generic.

Errors inside a sweep are printed per pass and stored in full in `predictions.csv`:

```bash
python -c "import csv;r=next(csv.DictReader(open('predictions.csv')));print(r['error'])"
```

## VS Code

Open the folder, pick an interpreter, press F5. Five launch configs: dry run, dry run with
repeats, a limited live run, a full live run, and tests. Default paths hang off the script
rather than the shell's cwd, so it works regardless of what VS Code sets as the working
directory.

## Extending it

**Add a strategy.** Write a builder that takes a message and returns a prompt, reuse or
write a parser returning `(label, status)`, add one row to `STRATEGIES` in `harness.py`.
`few_shot_instruct` was added exactly this way, in about ten lines.
Nothing else changes. Add tests for the new parser's four statuses; the failure buckets are
only as good as the parser feeding them.

**Swap the task.** Replace `dataset.csv` with your own `id,message,label` rows, update
`CATEGORIES`, rewrite the prompt text. The metrics don't care what the categories are. The
only task-specific leftover is `MockClient.KEYWORDS`, which only exists to make `--dry-run`
produce sensible labels.

**Swap the provider.** Write a class with the same `complete(...) -> Completion` signature.
The run loop doesn't know what's behind it.

## License

MIT.
