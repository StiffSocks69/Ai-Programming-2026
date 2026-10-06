# Degrees of Separation – Inlämningsuppgift 1

## Files
| File | Purpose |
|---|---|
| `degrees.py` | Grade 3: Breadth-First Search (`shortest_path`) |
| `compare.py` | Grade 5: A* search and comparison with BFS |
| `test_degrees.py` | Tests for BFS and A* |
| `util.py` | Node and frontier classes from the lectures |
| `small/`, `large/` | Data from the course |

## Setup
Python 3, no extra packages. Keep `small/` and `large/` in the same folder as the `.py` files.

```bash
python degrees.py small               # BFS only (grade 3)
python compare.py small               # BFS vs A* (grade 5)
python -m unittest test_degrees.py -v  # run the tests
```
Use `large` instead of `small` for the full dataset.

## Heuristic
There is no distance like Manhattan distance between actors, so we use what we know about the target:

- the target is 0 steps away
- anyone who starred in a movie with the target is 1 step away
- everyone else is at least 2 steps away

This never overestimates the real distance, so A* still finds a shortest path.

## Metrics
- **Degrees found**: checks that A* finds a path as short as BFS
- **People explored**: how much searching each algorithm does
- **Time (ms)**: actual running time

## Results
*(fill in after running `python compare.py large` on a few pairs)*

| Pair | BFS explored | A* explored | BFS time | A* time |
|---|---|---|---|---|
| | | | | |

## Discussion
The heuristic only has the values 0, 1 and 2, so it can only help in the last step
of the search: A* explores co-stars of the target first, which finds the target at once.
In the earlier steps, every person gets h = 2, so A* searches roughly like BFS.
A* also uses a priority queue, which is slower per step than BFS's queue, and it
spends one extra lookup to find the target's co-stars. Because of this, A* is
often not faster than BFS here. When two actors are not connected, both
algorithms must explore everyone reachable, so the heuristic does not help at all.