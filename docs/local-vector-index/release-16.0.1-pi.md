# Local vector search and persistent Pi memory on 16.0.1

The integration branch is `release/local-vector-pi-v16.0.1` on
`ezoushen/claude-mem`. It starts at upstream `v16.0.1` (`e22c58353`), merges the
existing `local-vector-index` branch (`34ad425f`), and merges
`fix/pi-persistent-memory-20261120`. The original vector branch remains unchanged.
Source and tests precede the final regenerated-plugin commit. Generated bundles
are rebuilt from source and are never edited directly.

## Contracts retained

- Local vector search is opt-in. With the option off, stock search remains the
  default. With it on and Chroma off, keyword and observation-vector results are
  fused. Existing `vec_observation_docs` rows keep their model identity
  `Xenova/bge-m3/q8/cls/l2/512` and the pinned model revision
  `4de13258303883538bd53b696b452bf8099f0858`.
- Pi admits prompts through the existing worker privacy checks. Its initial
  memory snapshot is persisted once in the selected branch. Reload restores those
  exact leading-system bytes; changed notes are persisted after a new user prompt.
  Clock/age-only changes do not add snapshots. Tool continuations defer new notes,
  and compaction retains the latest pruned snapshot. See
  [Pi memory across resume](../pi-native-capture.md#memory-across-resume).
- Memory snapshots contain conversation context text in Pi's private transcript.
  The separate `pi-prefix-stabilizer` diagnostics remain hash-only. The stabilizer
  is maintained in `ezoushen/pi-extensions`; it is not bundled into Claude-Mem.

## Build and qualify

Use Bun **1.4.4**, the version pinned in this release's CI, and a concrete supported
Node executable on `PATH`. A version-manager shim can fail in the temporary home
directories used by installer tests. This machine's old Bun 1.4.0 also reproduces
HTTP shutdown failures in unmodified upstream tests.

```sh
bun install --frozen-lockfile
npm run build
npm run typecheck
bun test tests/vector tests/services/vector tests/integrations/pi-extension-contract.test.ts tests/integrations/pi-private-capture.test.ts
npm run test:all
```

The build produces both `plugin/` and `dist/pi-extension/`. Installing only the
worker does not update Pi's separately installed adapter. Check the adapter hash
against this build after installing its seven owned files and license notices.
An already-running Pi process loads the adapter on its next reload/reopen.

Before adopting a new integration tip:

1. Save a consistent SQLite backup with the online backup API; keep it private.
2. Start the candidate on a backup copy with a separate data directory, home and
   port. Disable observation capture, transcript watching, My Work agents and
   worker automatic updates for that trial.
3. Check database integrity, vector counts/model identity, readiness and the
   existing graded search fixture against the current worker's baseline.
4. Validate Pi through fresh adapter activations and the actual provider
   serializer. Clock-only reload must preserve the whole request; real-note
   refresh must preserve the older wire-message prefix. Do not send replay
   capture/init requests to the live worker.

## Install this release on the machine

Publish the feature and integration branches to the fork before installing. Keep
`local-vector-index` at its original tip; do not rebase or force-push it for this
release. The marketplace named `thedotmack` must keep source repository
`ezoushen/claude-mem` and use ref `release/local-vector-pi-v16.0.1`, with
`autoUpdate: false`. Also set `CLAUDE_MEM_AUTO_UPDATE=false`: newer upstream
workers have a self-updater separate from marketplace auto-update.

Follow the outer project's update runbook for database/config backups, marketplace
refresh, plugin update, model-cache copying and worker restart. Verify the new
plugin cache against this branch's tracked `plugin/` files before starting it.
Keep the old version folder and original branch available.

Pi's installer reads a built `dist/pi-extension` from the marketplace root first.
Copy that generated directory from this exact release build into the updated
marketplace checkout before calling `installPiExtension()` from
`src/services/integrations/PiInstaller.ts`. Its normal preflight validates the
package attributions and atomically replaces only the owned Pi extension files.
Record the release commit, adapter SHA-256, worker bundle SHA-256, package version
and runtime version in the local deployment receipt. Never put the database,
credentials or captured memory text into a public receipt or commit.

After restart, check worker readiness and search, then verify that normal startup
selects the same installed release. Preserve existing provider/account settings,
vector options and model identity. Do not restore an older database merely to
roll back code; it would discard memories written after the backup.

No GX10 model runtime, networking or serving configuration is part of this release.
