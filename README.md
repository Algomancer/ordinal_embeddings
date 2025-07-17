# ordinal_embeddings

Tiny, reversible embedding of big ints => unit vectors (n‑dim sphere) via multi‑prime sin/cos phases + Garner CRT.

```
from ordinal import OrdinalEmbedding
emb = OrdinalEmbedding(max_value=2**32-1, n_dim=18)

vec = emb(torch.tensor([123456789]))  # → 18‑D unit vector
num = emb.decode(vec)                 # → 123456789
```

# How it works
pick co‑prime periods (primes)

encode int as phases 2π·x/p → [cos, sin] pairs

normalize; decode with Garner CRT

That’s it. Vector is dope for outputs - slots neatly into von mises‑style density/mixture density estimation and spherical invertible flows if you need a continous ordinal target.
