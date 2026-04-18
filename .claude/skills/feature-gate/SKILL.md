---
name: feature-gate
description: Scaffold the four Discovery Gate Serena memories for a new VoiceLayer feature under features/<name>/. Creates FEATURE_BRIEF, ACCEPTANCE_CRITERIA, ASSUMPTIONS, and RISKS templates in Chinese. Use before starting any non-trivial feature.
disable-model-invocation: true
---

Feature slug: $ARGUMENTS

Scaffold four Serena memories under `features/<slug>/`, where `<slug>` is the kebab-case form of the argument. Use `mcp__serena__write_memory` for each memory. Write bodies in Chinese, following the Discovery Gate convention in the global development workflow.

Before writing any memory, ask the operator for:

- The one-sentence user story or problem statement.
- Known constraints (desktop environment, latency budget, provider selection, privacy posture).
- Critical open questions that must be resolved before implementation.

Do not fabricate scope, acceptance criteria, or risk scenarios. When details are missing, leave a `TODO:` line calling out exactly what is required from the operator.

Create these four memories:

1. `features/<slug>/FEATURE_BRIEF` — scope, goals, non-goals, constraints.
2. `features/<slug>/ACCEPTANCE_CRITERIA` — BDD Given/When/Then plus concrete examples.
3. `features/<slug>/ASSUMPTIONS` — validation plan and expiry for each assumption.
4. `features/<slug>/RISKS` — risk register with mitigations and test hooks.

The gate only passes when clarity is at or above 80% and every critical question has an answer. After writing, summarize open items the operator must resolve before implementation starts.
