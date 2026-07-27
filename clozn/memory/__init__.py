"""Memory cards and prompt-mode settings.

SCOPE, after the 2026-07-27 cut: memory is PLUMBING, not a feature. Cards are compiled into a prompt
system block and nothing else. There is no memory page, no CLI verb, no headline -- steering is the
one advertised personalization surface. `clozn.profiles` and `clozn.receipts.explain` are the reason
this package still exists; both read cards to report what was active on a run.

REMOVED here, with the measurements that decided it (notes/ANCHORED_MEMORY_FINDINGS.md):
  * anchored memory -- k-sparse {token: alpha} bags injected as one direction at L21. Measured on its
    OWN qualified config (Qwen2.5-7B-Instruct Q4_K_M, L21, k=4, s=0.5): no topic bleed, but it could
    not RECALL. The bag fitted for "My cat is a grey tabby named Miso who was adopted in Kyoto"
    stored {cat, grey, named, adopted} -- every proper noun dropped, because dir(c) requires
    single-token words. Asked the cat's name, the injection produced "your cat named Qwen": a
    fabrication that was ABSENT without it. The limit is the representation, not the tuning.
  * topic_gate -- the MiniLM relevance gate. It existed to stop the *trained soft prefix* bleeding,
    needed sentence-transformers (not a product dependency), and therefore returned 1.0 in the
    shipped configuration -- gating nothing.
  * markdown_cards -- its CLI verb went in the 41->33 cut; zero remaining callers.

The facts/slot-memory tier already lived in clozn/lab/slotmem_qwen (torch, research-only, never
wired to change a product reply).
"""

from . import cards, mode

__all__ = ["cards", "mode"]
