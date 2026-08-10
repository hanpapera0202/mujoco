# Nova5 project memory

Read `docs/project_memory_zh.md` before changing the simulator, controller,
MuJoCo model, dashboard, or benchmark.  That file is the durable requirement
ledger for this repository.

## Non-regression contract

Before reporting or pushing a simulator result, run:

```powershell
python -m unittest discover -s tests -v
```

The fixed-seed physical regression must continue to prove all of the following:

- Seed 42 assigns both equal-peer arms and moves them concurrently.
- `part_01` and `part_02` are grasped by two physical finger-pad contacts.
- No grasp weld/equality constraint is activated.
- No arm-arm collision, safety stop, or false placement is accepted.
- Feed rate and active-part limits remain explicit configuration parameters.

Do not weaken an assertion, widen a physical tolerance, reactivate a weld, or
rename a failure into success to make a test pass.  If an intentional research
change invalidates a requirement, update the requirement ledger with the user,
add the replacement acceptance test, and record the decision in the commit.

## Change workflow

1. Add a new requirement ID to `docs/project_memory_zh.md` before implementing
   a new behavior.
2. Preserve all requirements marked `PASS` unless the user explicitly changes
   them.
3. Add or update an automated test for every requirement that can be measured.
4. Run the complete test command and a relevant fixed-seed headless replay.
5. Follow `docs/versioning_zh.md`, update `VERSION`, then commit, tag, and push
   only after the passing evidence is recorded.

