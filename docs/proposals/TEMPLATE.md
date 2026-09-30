<!--
  TEMPLATE — delete every HTML comment and every <placeholder> before submitting.
  Keep the section order unless there is a reason to deviate; consistent structure
  is what makes proposals comparable and reviewable.
-->

# Proposal: <short imperative title>

| Field | Value |
|---|---|
| **Scope** | <one line: what this document decides> |
| **Out of scope** | <one line: what it deliberately does not decide, and where that lives instead> |

---

## 1. Summary

<!--
  The decision, stated up front, in 3-5 sentences. A reader who stops here should
  know what is being proposed and why. Do NOT make them read to section 7 to find
  out what the document is arguing for.
-->

<What is being proposed, in plain language, and the single strongest reason for it.>

## 2. Context and motivation

<!--
  The problem, and why it needs solving now. Reference prior work, incidents,
  constraints, or deadlines. If this proposal is a response to another document,
  link it and state the specific deficiency being addressed.
-->

<The situation before this change. What is currently done, and what is wrong or
missing about it.>

## 3. Goals and non-goals

**Goals** — what a successful outcome looks like:

- <goal 1, phrased so it is verifiable>
- <goal 2>

**Non-goals** — explicitly out of scope, and why:

- <thing a reader might expect to find here, and the reason it is excluded>

## 4. Assumptions and constraints

<!--
  Everything the proposal depends on being true. Anything unverified belongs here
  rather than being quietly assumed in the design. Mark status per row.
-->

| Assumption | Status | If it fails |
|---|---|---|
| <assumption> | verified \| assumed \| unknown | <contingency> |

**Hard constraints:**

- <constraint that rules out whole classes of solution>

## 5. Design

### 5.1 Overview

<!--
  The shape of the solution. A diagram, a schema, a table — whatever makes the
  structure obvious at a glance. Keep it simple enough to be correct.
-->

<Diagram or summary table>

### 5.2 <Component or stage 1>

<!--
  Repeat per component. For each, state:
    - what it does
    - why, specifically
    - what it costs, and what it gives up
    - what happens if it is wrong
  A component with no stated cost is usually an unjustified decision.
-->

<Description, rationale, trade-offs>

### 5.3 <Component or stage 2>

<Description, rationale, trade-offs>

## (Optional) Verification

<!--
  How this will be shown to work. Name the metrics and the acceptance criteria
  BEFORE the results, so the evaluation cannot be retrofitted to the outcome.
  Include the baseline to beat, and any threshold selection procedure.
-->

**Acceptance criteria** — this proposal succeeds if:

- <measurable condition>

**Method:**

- **Baseline to beat:** <what a trivial approach achieves>
- **Primary metric:** <metric, and why that one>
- **Selection procedure:** <how thresholds and hyperparameters are chosen, and
  on which split. State explicitly what must never be tuned on test data.>
- **Reporting:** <which metrics are reported together, and what is deliberately
  not reported as a headline>


## 6. Next Steps

<!--
  Optional — delete if not applicable. For changes that affect running systems,
  cover staging, migration, and rollback. For a standalone artifact, replace
  with "how this gets built and where it is published."
-->

- **Step 1:** <first step, with the gate for moving on>
- **Step 2:** <next step>
- **Rollback:** <how to revert, and what would trigger it>