# Results

llama3.2 via Ollama, 40 items, 5 strategies, 3 runs, 600 calls, temperature 0.

```
rank strategy           acc    95% CI         sd     mean_ms  out_tok  wrong  halluc  malfm
1    chain_of_thought   0.950  [0.83, 0.99]   0.000  1340     6204     6      0       0
2    strict_json        0.900  [0.77, 0.96]   0.000  265      840      12     0       0
3    few_shot_instruct  0.875  [0.74, 0.95]   0.000  134      240      15     0       0
4    zero_shot          0.725  [0.57, 0.84]   0.000  129      240      33     0       0
5    few_shot           0.550  [0.40, 0.69]   0.000  478      2037     27     3       24
```

## There is no winner, and that is the result

The top three are mutually indistinguishable under exact McNemar:

| pair | discordant | p |
|---|---|---|
| few_shot_instruct vs chain_of_thought | 3 | 0.2500 |
| few_shot_instruct vs strict_json | 5 | 1.0000 |
| chain_of_thought vs strict_json | 6 | 0.6875 |

A 7.5 point gap between chain_of_thought and few_shot_instruct comes down to 3 items out of
40. Reporting that as "chain-of-thought wins" would be reading noise.

What is significant: all three beat zero_shot (p = 0.0312, 0.0039, 0.0391) and all three
beat few_shot (p = 0.0023, 0.0001, 0.0013).

## Chain-of-thought costs 26x the output tokens for no measurable gain

Against few_shot_instruct it is 10x slower and uses 26x the output tokens, for a difference
that does not reach significance. Against strict_json, 5x slower and 7.4x the tokens, also
not significant. On short single-label classification, reasoning tokens are close to pure
cost.

The Pareto frontier is few_shot_instruct at 134 ms, strict_json at 265 ms, and
chain_of_thought at 1340 ms. zero_shot and few_shot are dominated: something on the
frontier is both faster and more accurate.

## Three examples repair one specific confusion

zero_shot has billing recall 0.300. It sends billing messages to `account` 12 times and to
`technical` 6 times. few_shot_instruct, same model and three examples, lifts billing recall
to 0.800.

The gain is not spread evenly across classes. It is one confusion being fixed, which is the
kind of thing an aggregate accuracy number hides and a per-class table does not.

## Prompt format matters more than the examples themselves

few_shot and few_shot_instruct carry identical examples and differ only in format. The
completion-style version ends with a dangling `Category:`; the instruct version ends with an
explicit instruction.

- few_shot vs few_shot_instruct: p = 0.0023, 15 items go to the instruct version
- few_shot vs zero_shot: p = 0.1435, not distinguishable

So badly formatted examples are worth no more than no examples at all. few_shot also leaves
27 of 120 responses unparseable and drops account recall to 0.100, because the model
anchors on the first demonstration instead of classifying.

Before stop sequences were added, few_shot scored 0.000: llama3.2 echoed the first
demonstration back and began inventing new ones, and `max_tokens=16` truncated it mid-flow.
That was found by reading `raw_text` in `predictions.csv`.

## Determinism

`sd = 0.000` for every strategy across three runs. llama3.2 at temperature 0 through Ollama
reproduces exactly, down to identical confusion matrices. The repeats establish that all
remaining uncertainty is item sampling, which the Wilson intervals cover.

Accuracy figures are also stable across code revisions: a refactor that removed the
Anthropic client and rewrote the sampling logic reproduced every number in this table,
with only wall-clock latency differing.

## What these numbers do not support

One model, one task, 40 items, single-label gold data. Nothing here transfers to a
different task, a larger model, or a larger label space. The confidence intervals span
10 to 15 points; anything inside that is not a finding.
