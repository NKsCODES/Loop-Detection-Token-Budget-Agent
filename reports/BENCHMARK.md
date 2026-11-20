# Stuck-rate benchmark

50 tasks: 20 healthy, 30 loop traps (6 each: retry after 503, polling a job that never
finishes, empty search, circular records, runaway pagination). Offline scripted model,
seeded, so every run gives the same numbers. Reviewer decisions come from a fixed policy.

| Variant | Stuck | Stuck rate | Passed | Tokens | Cost (USD) | Reviews | Loop false positives |
|---|---|---|---|---|---|---|---|
| Baseline ReAct (recursion_limit=25) | 30/50 | 60% | 20/50 | 492,574 | 1.59 | 0 | 0 |
| Step limit only (10 rounds) | 31/50 | 62% | 19/50 | 376,291 | 1.22 | 0 | 0 |
| LoopGuard (full) | 0/50 | 0% | 39/50 | 304,169 | 0.99 | 44 | 1 |

LoopGuard cut total tokens by 38% and tokens spent on trap tasks by 48%. Median detection step on traps: 3.0.
Traps resolved with an honest answer after one hint: 19/30. Traps stopped by the reviewer: 11.

## LoopGuard by task category

| Category | Tasks | Stuck | Passed | Avg tokens | Avg steps | First detector |
|---|---|---|---|---|---|---|
| healthy_gst | 4 | 0 | 4 | 2,227 | 2 | none |
| healthy_kb | 3 | 0 | 3 | 1,449 | 1 | none |
| healthy_long_docs | 2 | 0 | 2 | 22,604 | 4 | none |
| healthy_multi_lookup | 4 | 0 | 4 | 7,442 | 6.8 | tool_overuse x1 |
| healthy_org | 2 | 0 | 2 | 1,393 | 1 | none |
| healthy_pagination | 1 | 0 | 1 | 3,066 | 3 | none |
| healthy_polling | 2 | 0 | 2 | 3,032 | 3 | none |
| healthy_weather | 2 | 0 | 2 | 1,396 | 1 | none |
| trap_empty_search | 6 | 0 | 4 | 3,782 | 4 | stagnation x6 |
| trap_ping_pong | 6 | 0 | 4 | 8,118 | 8 | cycle x6 |
| trap_polling | 6 | 0 | 4 | 3,781 | 4 | exact_repeat x6 |
| trap_retry_error | 6 | 0 | 4 | 3,833 | 4 | exact_repeat x6 |
| trap_runaway_pagination | 6 | 0 | 3 | 14,024 | 12 | tool_overuse x6 |

## Repeat threshold sweep

N = repeats before a loop is flagged (exact repeat, stagnation and cycle).

| N | Stuck | Passed | Tokens | Loop false positives | Median detection step |
|---|---|---|---|---|---|
| 2 | 0 | 39 | 265,417 | 3 | 2.0 |
| 3 | 0 | 39 | 304,169 | 1 | 3.0 |
| 4 | 0 | 39 | 347,752 | 1 | 4.0 |
