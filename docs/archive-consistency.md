# Finalizing a sliding SfM archive

The streaming GPU session optimizes a local window, but exports the historical cameras and
points too. An archived observation is not a constraint in a later local solve. Replacing its
point or observing camera can therefore increase its reprojection error even when the local
window's inlier error improves. The local diagnostic must not be described as the error of the
complete exported map.

`IncrementalSfmSession.finish()` now gives a rolled archive one final complete-observation BA.
It preserves the observations, feature identities and point count while jointly refining camera
poses and points. The GPU solver already streams observation/point scratch to its working budget;
the session's 64-view input-window bound is not a limit on this final solve. Disconnected factors
use the existing fixed-boundary partition plan. A mandatory resident-floor refusal preserves the
unsolved component and marks the archive refinement deferred. Device loss and cancellation still
propagate. Invalid archived cheirality is retained and reported with a deferred solve and an
infinite whole-output error, rather than silently pruned to claim improvement.

For an automatic shared focal with matching image aspect, pixel aspect and distortion, final
calibration can initialize all cameras at one common physical focal ratio **inside the joint
pose/point/focal solve**. An early search winner at its grid boundary schedules at most three
starts: the carried estimate and up to two candidates from the original grid, once new views beyond the original calibration subset arrive.
This late calibration also applies to maps that still fit entirely inside the local window. Each start uses the same full observations; only a
candidate that does not increase their original Huber objective is accepted. This does not
overwrite focal numbers after optimization or assert a ground-truth focal. Supplied or grouped
camera intrinsics are not forced into this common model. Multiple focal groups use fixed-
intrinsics archive BA because their joint GPU focal solve is unsupported. Pending directional
factors also defer archive optimization instead of changing their geometry with a finite-only
objective.

`historyBundleIterations` bounds each final archive solve (0..100). Its default inherits
`finalBundleIterations`, except a boundary-selected automatic focal gets at least 32 iterations
per start. An explicit `finalBundleIterations: 0` skips both local and archive BA. Regardless of
optimization, the final exported mean is measured across **every exported observation**, including
outliers; it is not the local window's inlier-only mean. The result's `historyRefinement` diagnostic
records the full view/observation counts, before/after robust cost and deferred status. Accepted
geometry is synchronized back to local track identities and image cameras so a later finish or
add cannot restore the stale local solution.
The Lab also preserves this solved camera when attaching frame timing and image metadata;
the original frame's supplied camera must not overwrite the accepted calibration.

Historical tracks that disagree on the feature in an archived camera cannot be merged into a
single BA point. Their conflicting archive stays a separate point when it still has two views.
Current-window filtering continues to own the active observations, including rejected features.

`tests/history-refinement.test.ts` covers fixed-population refinement, whole-output metrics,
rollback, camera gauges, partition/defer behavior, device faults and local-state synchronization.
The streaming CPU regression uses 10 cameras with an eight-camera window and executes finish,
finish, add and finish; it actually evicts cameras. `scripts/replay-history-refinement.mjs` runs
the same finalizer on a local Lab export using WebGPU, audits any legacy duplicate-camera
observations once, and independently evaluates the result against all original observations.

Lower reprojection cost is an internal consistency improvement. It does not, by itself,
establish correct focal calibration, metric depth, surface boundaries or native GPU speed.
