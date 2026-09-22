# finetuning

`openpi/` and `seer/` are **git subtrees** (not submodules). Their files are
tracked directly in this repo, so local edits are ordinary commits here.
Upstream changes are merged in with a real three-way merge, which keeps local
edits and only conflicts where upstream touched the same lines.

Upstreams:

| directory | upstream | remote name |
|-----------|----------|-------------|
| `openpi/` | https://github.com/Physical-Intelligence/openpi | `openpi-upstream` |
| `seer/`   | https://github.com/OpenRobotLab/Seer            | `seer-upstream`   |

## One-time setup on a new clone

Remotes are not cloned, so add them once per machine:

```bash
git remote add openpi-upstream https://github.com/Physical-Intelligence/openpi
git remote add seer-upstream   https://github.com/OpenRobotLab/Seer
```

## Pull upstream changes

```bash
git subtree pull --prefix=finetuning/openpi openpi-upstream main --squash
git subtree pull --prefix=finetuning/seer   seer-upstream   main --squash
```

Resolve any conflicts like a normal merge, then `git commit`.

## Rules

- Never create a `.git` directory inside `openpi/` or `seer/`. That turns the
  directory back into a broken submodule pointer.
- Always use `--squash` so upstream history is not replayed into this repo.
