Task Workflow
Before starting any task:

Check tasks/lessons.md for relevant past mistakes
Write plan to tasks/todo.md with checkboxes
Check in with user before implementation begins

During:

Mark items complete as you go
Write a high-level summary after each step

After:

Add a review section to tasks/todo.md
If the user corrected you: log the pattern and rule to tasks/lessons.md immediately


Verification
Never mark a task complete without:

Running the relevant notebook cell or script end-to-end without errors
Checking output shape, dtype, and a sample of values when transforming data
Confirming model metrics are written to tasks/todo.md review section
Asking: "Would a staff ML engineer approve this?"


Bug Fixing
When given a bug report, error, or failing output:

Read the traceback completely before touching any code
Identify root cause — do not patch symptoms
Fix it. Do not ask for handholding unless the root cause is genuinely ambiguous
Log what caused it and the fix to tasks/lessons.md


Code Standards

Prefer simple, readable code over clever code — this pipeline will be read and modified later
No hardcoded values in src/ — use config.yaml
Comments explain why, not what
Minimize lines of code without sacrificing clarity
Keep functions single-purpose and short