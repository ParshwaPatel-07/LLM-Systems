# Shared code

Assignments build on each other:

- **Source of truth for the LM** lives in [`../assignment1_basics/cs336_basics/`](../assignment1_basics/cs336_basics/) (`tokenizer.py`, `model.py`, …).
- **Assignment 2** consumes that implementation via [`../assignment2_systems/cs336-basics/`](../assignment2_systems/cs336-basics/) (a copy of the A1 package, so `uv` can depend on it locally).

When you change the Transformer / tokenizer in A1 and want A2 to pick it up:

```bash
rsync -a --include '*/' --include '*.py' --exclude '*' \
  ../assignment1_basics/cs336_basics/ \
  ../assignment2_systems/cs336-basics/cs336_basics/
```

Later, if a true shared package is worth it, move stable modules here and depend on `common` from both assignments.
