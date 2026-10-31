# Three-minute pitch

## Problem — 0:00–0:20

Every student gets the same explanation, even though students approach maths
differently. But matching teaching to supposed "learning styles" is not
evidence-based, and a student's favourite explanation is not necessarily the
one that helps them get the next question right.

## Idea — 0:20–0:40

The student asks a maths question in a chat. If they have already chosen a
preferred explanation style, the tutor uses it. Otherwise, it offers a
step-by-step explanation, an analogy, and a worked example, and remembers the
student's choice. For non-definition questions, it follows with a short
practice prompt. While the student learns maths, our agent learns how they
prefer to learn.

## Live demo — 0:40–2:40

Ask the tutor a question. With no saved preference, the three explanations
appear side by side. Choose one and it becomes the saved style for the next
question. Ask for a definition and there is no unnecessary knowledge check;
ask how to solve a problem and the tutor asks a related follow-up. The Gemini
provider generates the explanations and follow-up question using the
`GOOGLE_API_KEY` from `.env`; the UI warns that prompts and saved
style preferences are sent to Google. A local demo provider remains available
for offline UI testing.

## Honest close — 2:40–3:00

AI tutoring has promising evidence, but whether a long-term, self-correcting
student profile makes tutoring better is still an open question. We are not
claiming to have solved that. This is our small, testable version of the experiment: remember the student's
explicit presentation preference without claiming that preference alone proves
what improves learning.

## If a judge asks, "Isn't the simulation circular?"

No. The simulated students' choices and correctness come from fixed rules in
the simulator, not from the tutor agent. The agent has to infer those hidden
rules from observed rounds. The chat demo is a separate way to try the
student-facing tutoring experience.
