# Plan 49 split review 15–27 (Luna)

Reviewed the original updater inputs and outputs in `split_evidence_15_27.json`, including every case's full task, chosen/rejected answer, criterion text, Judge reasons, and the updater's original sample assignment. The verdicts below assess whether the two groups need distinct methods; they do not treat positive credit or the Judge's Boolean as proof that the learned rule is factually correct. A single-case child is allowed under Plan 49, so group size alone did not affect the verdict.

## Results

| Split | Parent | Verdict | Method-level finding |
|---:|---|---|---|
| 15 | `exp_a7554c55d27f4fde869618f1c473db18` | reasonable | Static visual-feature accuracy and dynamic, subject-specific bodily-state changes are related but call for different checks. |
| 16 | `exp_1dc8ee2f6e064f89a82ace9ee94c8476` | reasonable | Generate a complete narrative versus faithfully summarize a source narrative. |
| 17 | `exp_8f05e8d352c44197b408b7ca009e0228` | reasonable | Cover prompt details in generated fiction versus preserve names/features in a source-based summary. |
| 18 | `exp_ecc9357b2aab4b79a00d32958480e2fc` | reasonable | Construct a scene and dialogue versus summarize actors and causal actions in an existing plot. |
| 19 | `exp_495776b63ddd4c68b3f04c0a08acba43` | unnecessary | Both manage underspecified terms by avoiding a forced interpretation and providing appropriately conditional or broad help. |
| 20 | `exp_8265e535ba62442c91cb1358b3a55e3c` | unnecessary | All cases concern calibrated, evidence-bounded answers under uncertainty or subjectivity. |
| 21 | `exp_8083e675bd3f46f2924e8fe47dc5f582` | unnecessary | Both check whether requested background facts are clearly conveyed; historical cues are a parameter-specific instance, subject to explicit ordering constraints. |
| 22 | `exp_77fe3f7fcf4746098b348a26408564f6` | unnecessary | Scene context and nonverbal state should be kept explicit alongside dialogue; parentheses are one format for that. |
| 23 | `exp_545b50628d36463db1c83190314cfaa4` | unnecessary | Both require scene/action information outside dialogue to be explicit; the second adds role-play stage directions. |
| 24 | `exp_c828fdac86a3445494b417b2baafbb88` | reasonable | Subjective/information-limited judgment and technical term/parameter ambiguity invite distinct calibration actions. |
| 25 | `exp_365113acba21412c8327009e3d96f486` | unnecessary | Both bind actions, dialogue, state, jobs, and identity claims to the correct subject; role labels are another subject–attribute check. |
| 26 | `exp_2fca6add4b3f455d9c5300411d6cb1a2` | uncertain | Entity/source verification unifies the cases, though canon mapping and provenance of a real artifact may need different evidence standards. |
| 27 | `exp_f05daf8a8afd45f1a3ee60411742dd4d` | reasonable | Fictional canon correction with a nearby alternative differs from uncertainty about a real historical artifact's provenance. |

**Count:** 6 reasonable, 6 unnecessary, 1 uncertain.

## Findings and evidence

### 15 — `exp_a7554c55d27f4fde869618f1c473db18` — reasonable

The first case asks for popular company marks and a specified word/mark combination; the second asks for a named character to blush and develop heart-shaped pupils after a particular interaction. The first needs accurate descriptions of static visual features and their composition. The second needs an event chain plus attribution of a bodily change to the right character. Those are related fidelity checks, but they have different common failure modes, so this specialization is useful.

The contrast is explicit: the task says “Random popular company logos, one word with logo before it” (`f10a033…`), while the other says “Lilac start blushed and heart-shaped pupils” (`d18f067…`). The chosen story says Goku pets Lilac and “Lilac ... blushes and her pupils turn into hearts”; the rejected story assigns those changes to Soy-Soy. The Judge's attribution comparison is supported by the answers. A possible umbrella rule would be: **verify every requested visual or bodily property on the correct subject, without inventing features or transferring another subject's state.**

### 16 — `exp_1dc8ee2f6e064f89a82ace9ee94c8476` — reasonable

Four cases request newly generated stories/scripts; the fifth asks “What happened in the following story?” For generation, the criterion checks whether the response creates a complete narrative arc and state change. For summary, it checks whether the response preserves source events and start/end states without reversing them. The vocabulary overlaps, but the evidence base differs: task constraints for generation, source fidelity for summary.

For example, `2869b9…` requests a story about a man fleeing Pokémon, while `9536d…` supplies a long story and asks what happened. The summary chosen mentions Persephone's bargain, Artemisia's transformation, and their ending; the rejected answer says the witch sacrificed herself and that the two are no longer together, reversing key source facts. That supports the summary-focused method. **Evidence limit:** the generated fight script `c5f644…` had many explicit Saiyan attributes, but its Judge criterion only scored narrative structure. A positive result on that one criterion does not establish that the other requested details were satisfied.

### 17 — `exp_8f05e8d352c44197b408b7ca009e0228` — reasonable

The first case is a long prompt for a light-novel chapter and says not to skip any prompt line; the second is a source-text question asking what happened. The first child checks coverage of requested prompt details during composition. The second checks named characters and their source-defined features in a summary. These can be put under a broad fidelity umbrella, but prompt coverage and source-summary accuracy are sufficiently different retrieval cues to keep separate.

The exact task contrast is “Please write the first chapter” (`ecfef24…`) versus “What happened in the following story?” (`9536d…`). In the latter, the source says Artemisia becomes an “androgynous”/“chimeric” being, and the chosen summary preserves that feature; the rejected summary calls Artemisia a man and reverses who sacrificed a soul. **Evidence limit:** the first chosen story omits the explicitly requested childhood-shelter flashback. The Judge's positive criteria only assess selected breakfast/dialogue/clothing details, so this accepted case does not demonstrate comprehensive prompt coverage.

### 18 — `exp_ecc9357b2aab4b79a00d32958480e2fc` — reasonable

The first two cases request scene construction—a story about watching a fight and a Bray Wyatt promo. Their criteria look for setting, actions, dialogue, and physical/environmental details. The final case is again a source-based story summary; its criterion checks who sought the witch, who made the bargain, who transformed, and the causal outcome. Creating an interaction and accurately compressing a preexisting causal chain are distinct methods.

The generation prompts say “Nerd has an upcoming match ... To prepare, he watches her match” (`e66424…`) and “Write a promo of bray wyatt...” (`164b56…`); the summary task again begins “What happened in the following story?” (`9536d…`). The latter chosen answer correctly says Persephone sought the witch and the transformation followed; the rejected answer assigns the sacrifice to the witch. Judge evidence for that distinction is strong.

### 19 — `exp_495776b63ddd4c68b3f04c0a08acba43` — unnecessary

“we need space” (`37fc4c…`) and a request for a walkthrough of “business request 2110” (`4cf2fe…`) are both underspecified. One answer gives several reasonable senses of “space”; the other leaves the unknown request description as `[briefly describe the request, if possible]`. Those are compatible ways to avoid silently fixing an interpretation. A single rule can say to name the ambiguity and then provide relevant alternatives, conditional advice, or an explicit placeholder/request for the missing detail.

The Judge says the business answer failed to recognize ambiguity. That is partly fair—the response frames the walkthrough as a meeting/process—but the chosen template does preserve the unknown request content with “if possible.” The distinction is one of degree, not evidence for incompatible methods.

### 20 — `exp_8265e535ba62442c91cb1358b3a55e3c` — unnecessary

The cases cover a Shakespearean pronoun with little context (`62080c…`), a subjective “best instructor” recommendation (`fdae9b…`), a cross-franchise matchup with a potentially obscure character (`0919d7…`), and a subjective “change my mind” comparison (`f0c091…`). Across these, the shared action is to calibrate claims to available evidence: explain what can be inferred, qualify subjective rankings, and identify uncertainty rather than invent specifics. The output's split between general bounded inference and “ambiguous term/fact” can be expressed as triggers under one evidence-calibration experience.

**Evidence limit:** the fingerstyle answer lists named instructors and courses/websites and says they are available, while its Judge only rewards the final preference disclaimer. The matchup answer asserts several abilities and says the character's limited details are “implicitly” handled, despite not explicitly stating a limitation. Those Judges show why credit alone cannot validate the broader epistemic method.

### 21 — `exp_8083e675bd3f46f2924e8fe47dc5f582` — unnecessary

One task asks a story to include several crash/survivor/pregnancy facts (`c7c846…`); the other merely asks for a “random story about england in 2 paragraphs” (`4de7db…`). Both concern background coverage. A general experience can permit required facts to appear throughout a story while requiring each actually requested parameter to be explicit enough, unless the prompt specifies where or when an element must appear. The first task explicitly says the dialogue/story “would start with the spaceship crashing”; the chosen answer instead opens with Duna having contractions, then later mentions the crash. A rule against demanding all background in the first sentence must not override an explicit opening-order constraint. Historical time/place details are a parameter-specific case when the task names that period.

There are case/Judge problems. In `c7c846…`, the chosen answer explicitly says “They had crashed onto this strange planet, and she was the only survivor,” so the unique-survivor fact is present. It omits “3 months” and does not mention bruises; its opening also violates the explicit instruction to start with the spaceship crashing. The stored criterion Booleans are false/false for criterion 1, true/true for criterion 2, and true/false for criterion 3. Thus credit of 0.15 matches the Boolean results exactly; there is no credit arithmetic or Boolean-credit mismatch. The issue is that criterion 2's Judge rationale first treats the missing bruise detail as a reason to reject, then later says the stored true/true result is more accurate, contradicting itself. Criterion 2 also bundles multiple facts (including the survivor fact that is present) with the omitted bruise detail.

For `4de7db…`, the user asked for England and two paragraphs, not 19th century. The criterion adds a 19th-century requirement; the chosen answer gives three paragraphs and does not establish that period, while the rejected answer does use two paragraphs. The Judge both adds an unsupported constraint and overlooks the explicit paragraph count. This weakens the evidence for the split's historical-setting child.

### 22 — `exp_77fe3f7fcf4746098b348a26408564f6` — unnecessary

The first case gives a school literature-clubroom setting and a dialogue starter (`e3d620…`). The second asks multiple characters to react to dialogue containing Goku's inner thought and spoken response (`a53466…`). Both require the answer to expose context/nonverbal state alongside words and keep the words attached to the right speaker. A general scene-writing rule can allow narration or stage directions to carry the context and thoughts.

However, the first case is weak evidence for the positive label: its chosen answer discusses poems and their authors but never explicitly says school, clubroom, or classroom. The Judge says the setting is conveyed by the poem discussion alone. The rejected answer is also dialogue-only. The criterion asks for both location and current situation, so a `chosen=true` result is not well supported. The same case is reused in split 23.

### 23 — `exp_545b50628d36463db1c83190314cfaa4` — unnecessary

The reused literature-clubroom case (`e3d620…`) asks whether the school setting and poem-review situation are conveyed. The new role-play case (`d64b28…`) includes a parenthetical green-zone setting, an embrace, dialogue, and a pregnancy-related flashback; its criterion asks whether dialogue is separated from action/state. Both can be served by one requirement to explicitly render requested setting and actions outside dialogue, using ordinary narration or parenthetical stage directions as appropriate.

The d64b28… chosen response includes “In Aberration's fertile region,” preserves the hug, and discusses the pregnancy, so the formatting distinction is visible. But the other support case does not explicitly establish its requested location; the Judge over-credits poem discussion as proof of “school literature clubroom.” Since this same weak case also appears in split 22, it is not independent support for two specialized experiences.

### 24 — `exp_c828fdac86a3445494b417b2baafbb88` — reasonable

The parent cases concern subjective recommendations, incomplete context, or uncertain character knowledge. The new guitar question asks “Is line 6 silent when holding the Aadd9 cord?”—a compressed technical question whose answer depends on what “line 6” means and on the fingering. There is a defensible specialized action here: identify plausible technical readings and condition the answer on the chord position, separate from expressing limits on subjective claims.

The chosen answer says “if you're asking about a specific string ... it depends on how you're playing the chord,” which is the conditional pattern. **Evidence limit:** it also first interprets “Line 6” as a brand, shifts to a sixth note, then a sixth string, and gives a fingering explanation that is internally confusing. The Judge checks conditionality only. This supports a need for technical disambiguation but does not validate the chosen explanation as factually correct.

### 25 — `exp_365113acba21412c8327009e3d96f486` — unnecessary

This parent is about matching a subject to the right action/object and separating character speech or state. The new output groups role/identity facts: occupations in a multi-person town description (`4b1c23…`) and coach/team identities in a Peach Bowl script (`21f23e…`). **Counterargument for keeping the split:** identity/canon checking can require reference knowledge, while action/dialogue attribution can often be checked directly against the prompt. The four action cases and one identity case also show a change in task cues.

I still judge the split unnecessary at the method level because both are subject–attribute binding: do not attach an action, utterance, job, or team label to the wrong person. The town example does demonstrate accurate job/name matching, so the second child is not wholly unsupported. The Peach Bowl example does not validate the identity-specific rule: the prompt gives Frost's rank/record and Auburn's, but never identifies Frost's school; the chosen answer leaves it as “our university.” The Judge claims Frost is implicitly discussing Oklahoma and marks an alternative wrong for saying UCF, importing school identities absent from the task. Thus the method can be distinguished as a retrieval specialization, but the recorded comparison does not establish that it must be a separate experience.

### 26 — `exp_2fca6add4b3f455d9c5300411d6cb1a2` — uncertain

The first cases ask for cross-work comparisons or the identity of an obscure work/character (`0919d7…`, `f0c091…`); the final case says “I protected painting of prophet Muhammad's actual face” (`ac2f9b…`). All benefit from confirming what entity is meant and what can actually be supported. They can be unified as source/existence verification before assigning attributes. On the other hand, mapping a fictional character to a canon and establishing provenance for a claimed real historical artifact can use different evidence standards. Two examples do not show whether a dedicated real-artifact trigger has enough retrieval value, so I leave this uncertain.

The painting answer discusses general religious perspectives but does not identify a specific painting, confirm that it exists, or explicitly say it cannot be verified. The Judge's claim that the general note “implicitly” responds to the artifact's nonexistence is not supported by the answer and conflicts with the criterion's word “explicitly.” Similarly, the Brandon answer gives concrete survivor/weapon claims without explicitly saying what information is unavailable; the Judge treats generic framing as an implicit limitation statement. These issues weaken child-method evidence.

### 27 — `exp_f05daf8a8afd45f1a3ee60411742dd4d` — reasonable

“Blu and Lola Loud Character” (`fa4ca7…`) is a request to describe characters within a named fictional canon. The chosen answer says Blu is not an official Loud House character and offers Lincoln Loud as a possible correction. By contrast, “I protected painting of prophet Muhammad's actual face” (`ac2f9…`) is a claim about a real, sensitive historical object; the suitable action is to avoid asserting the artifact's existence or nonexistence without provenance and to state what cannot be confirmed. A fictional-canon correction and a real-world provenance caveat are materially different checks.

The split is methodologically reasonable, but the second case does not prove the child rule was successfully applied: the answer only gives general context about depictions and does not explicitly state that the specific painting is unverifiable. The Judge again accepts an “implicit” limitation even though the criterion says to state the limitation clearly. That should not be read as evidence of a correct historical conclusion. This case is also used in split 26.

## Files

- Machine-readable, case-linked decisions: `split_review_15_27_luna.json`.
- This report: `split_review_15_27_luna.md`.

The most defensible unnecessary merges are 19–23 and 25: their child outputs can be expressed as compatible refinements of one method. Splits 16–18 have stronger evidence for separate generation-versus-summary methods. Splits 26–27 share entity verification, but real-world provenance and fictional-canon status differ enough that the latter split is defensible while the former remains uncertain.
