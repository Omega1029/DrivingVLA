# External reference copies

Read-only snapshots of harness scripts that live at `~/neuroncap/neuro-ncap/*.sh` on the
machine this project runs on — outside this git repo, and outside what a cloud-sandboxed
routine can see. Copied here **only** so they can be audited for the renderer-reuse bug class
documented in `PROGRESS.md` (2026-08-26/27 entries) and `papers/four_bits_without_loss.tex`.

**Do not run these from here.** They reference paths (`data/nuscenes`, `checkpoints/`,
`neuro_ncap`) that only resolve from their original location, and they are not kept in sync
automatically — if the originals change, these go stale until someone re-copies them. Treat
as a snapshot taken 2026-08-27, not a live mirror.

Snapshot taken: 2026-08-27. Audit status of each is tracked in `AUTOLOOP.md`'s task queue.
