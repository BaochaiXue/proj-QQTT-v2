---
name: first-principles-readable-code
description: Code/代码, refactor/重构, implementation plan/修改方案, debugging/排错, code review/代码审查: enforce first-principles requirement analysis, human-readable code, fail-fast invalid-state handling, no compatibility patches, no unrequested fallbacks, shortest correct path, and end-to-end logic validation.
---

# First-Principles Readable Code

## Purpose

Use this skill whenever a task involves writing code, modifying code,
debugging, refactoring, proposing an implementation plan, or reviewing a
technical solution.

The core rule is: solve the actual requirement from first principles, not from
guessed intent, habitual patterns, compatibility shortcuts, or unrequested
fallback behavior.

## Mandatory Principles

### 1. Start from first principles

Before proposing a solution or changing code, identify the real problem from
the original requirement.

Always distinguish:

- The user's stated goal.
- The observable current problem.
- The required final behavior.
- The hard constraints.
- The unknowns that affect correctness.

Do not assume the user already knows exactly what they want or how to get it.

If the motivation, target behavior, business rule, boundary condition, or
success criterion is unclear, stop and ask the minimum necessary clarification
questions before implementing.

Do not implement around ambiguity.

## Code Requirements

### 2. Human readability has priority

When writing any code, optimize first for human readability.

Readable code means:

- Clear names that express domain meaning.
- Direct control flow.
- Small functions with one clear responsibility.
- Explicit data flow.
- Minimal hidden state.
- Minimal implicit behavior.
- No cleverness for its own sake.
- No unnecessary abstraction.
- No compressed logic that makes future reasoning harder.

Prefer boring, direct, maintainable code over concise or clever code.

Use comments only when they explain intent, business constraints, non-obvious
decisions, or invariants. Do not add comments that merely restate obvious code
mechanics.

### 3. Preserve business semantics

Do not change business behavior outside the requested scope.

Do not solve a problem by weakening validation, broadening accepted inputs,
changing output semantics, silently ignoring errors, or introducing partial
success behavior unless the user explicitly requires that behavior.

## Plan and Refactor Requirements

When giving a modification plan, refactor plan, or implementation plan, follow
these rules.

### 4. No compatibility or patch-style solutions

Do not propose:

- Compatibility layers.
- Temporary adapters.
- Dual-path logic.
- Migration shims.
- Legacy fallbacks.
- Patch-like fixes that hide the underlying issue.
- Logic that keeps both old and new behavior alive to avoid making a clear
  correction.

The solution must directly implement the intended final behavior.

If the only available approach appears to require compatibility or patch logic,
stop and explain the conflict instead of proceeding.

### 5. Invalid cases must fail fast

For invalid input, invalid state, missing required data, inconsistent domain
state, permission violations, impossible branches, or broken invariants:

- Fail immediately at the boundary where the invalid condition is detected.
- Surface an explicit error according to the project's existing error model.
- Do not silently coerce values.
- Do not invent default values.
- Do not skip invalid records.
- Do not retry, downgrade, fallback, or auto-recover.
- Do not convert hard failures into soft success.

Only handle invalid cases differently when the user gives an explicit rule for
how to handle them.

### 6. Do not over-design

Choose the shortest correct implementation path.

Do not introduce:

- Generic frameworks.
- Configuration switches.
- Plugin systems.
- Future-proof extension points.
- Caching.
- Queues.
- Retries.
- Observability layers.
- New abstractions.
- New dependencies.
- New architectural boundaries.

Only add structure that is required by the current stated requirement or
necessary to make the current logic correct and readable.

### 7. Do not invent extra solutions

Do not propose alternatives outside the user's requirement, such as:

- Fallback strategies.
- Degraded modes.
- Extra product behavior.
- Optional flows.
- Additional UX states.
- Additional API contracts.
- Additional validation policy.
- Additional migration policy.

Extra behavior can shift business logic. Avoid it unless the user explicitly
asks for it.

## Required Workflow

### Step 1: Requirement extraction

Before implementation, reduce the request to:

```text
Problem:
Required final behavior:
Inputs:
Outputs:
State changes:
Invalid cases:
Constraints:
Unknowns:
```

If any item that affects correctness is unknown, ask before proceeding.

### Step 2: Minimal correct design

Create the smallest design that satisfies the requirement.

The design must state:

```text
Files or modules to change:
Core logic change:
Error handling:
Data flow:
Why this is sufficient:
```

Do not include optional alternatives unless the user explicitly asks for
alternatives.

### Step 3: Implementation

When editing code:

- Follow the existing project style unless it harms readability or correctness.
- Keep changes local to the required files.
- Remove obsolete logic instead of layering new logic over it.
- Prefer explicit validation near the boundary.
- Prefer explicit domain names over generic names.
- Keep the implementation easy to inspect line by line.

### Step 4: End-to-end logic validation

Before finalizing, verify the full chain:

```text
1. Input enters the system.
2. Required validation happens.
3. Invalid states fail immediately.
4. Valid data reaches the core logic.
5. Core logic preserves the intended business rule.
6. Side effects happen exactly once and in the intended location.
7. Output matches the requested final behavior.
8. No unrequested fallback, compatibility path, or degraded behavior was introduced.
9. No unrelated business behavior changed.
10. Tests, type checks, or equivalent verification were run when available.
```

If verification cannot be run, state exactly what was not run and why.

## Response Requirements

When responding with a plan, use this structure:

```text
Requirement:
Unclear points:
Plan:
Validation:
```

If there are unclear points that affect correctness, stop at `Unclear points`
and ask for clarification.

When responding after code changes, use this structure:

```text
Changed:
Why:
Validation:
```

Keep the response concise. Do not include speculative alternatives.

## Forbidden Patterns

Do not use these patterns unless the user explicitly overrides the rule for the
current task:

- "To be safe, keep the old path too."
- "Add a fallback just in case."
- "Return an empty value on error."
- "Silently ignore invalid records."
- "Try/catch and continue."
- "Make it configurable for future needs."
- "Add an adapter layer for now."
- "Support both old and new formats."
- "Add a temporary compatibility bridge."
- "Implement a degraded mode."
- "Infer missing business data."
- "Auto-correct invalid input."
- "Add a generic abstraction for possible future use."

## Definition of Done

A task using this skill is done only when:

- The requirement is understood from first principles.
- Ambiguities that affect correctness have been resolved.
- The implementation is human-readable.
- The solution does not rely on compatibility or patch logic.
- Invalid states fail explicitly.
- The implementation is the shortest correct path.
- No unrequested behavior was added.
- The full logic chain has been validated.
