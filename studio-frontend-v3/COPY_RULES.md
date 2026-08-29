# v3 copy review

Before adding persistent visible text, ask:

- Is this precise?
- Is this redundant?
- Does the user need it here, or does it belong in onboarding, docs, or help?
- Will this fatigue someone who sees the interface every day?

If removing text does not make the interface less precise or less safe, remove it.

Valid copy identifies real persisted objects and facts: session IDs, titles, dates, counts, model names,
states, and recorded prompt or response summaries. Avoid narration, invented causality, and headings that
sound like a prototype explaining itself.

Review checklist:

- [ ] The four questions above were applied to every new visible string.
- [ ] Visible labels identify a persisted object, a measured fact, or a necessary control.
- [ ] No text implies that a branch, diagnostic, or change explains another turn unless the backend says so.
- [ ] Empty, unavailable, and malformed-contract states remain explicit and distinct.
