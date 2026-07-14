"""Thin RTG-SLAM shim around gs_stats.WorkloadRecorder.

RTG-SLAM's recommended slam.py path is already single-process/synchronous,
and every rasterizer call funnels through SLAM.render.Renderer.render — so
recording happens there, tagged by a `stats_tag` passed from each call site
plus a per-frame context set in slam.py.

Phases (all render sites, per user decision):
  frame_render        — per-frame full render of global params (feeds ICP)
  map_local           — unstable optimization iteration (partial tiles)
  map_global          — stable/global optimization iteration (partial tiles)
  eval_range_unstable / eval_range_stable — T-map/tile-mask prep renders
  error_remove        — error-accumulation render (feeds stable prune)
  attach              — surface-attach render for new points
  eval                — eval.py metric renders

Rows carry `gaussian_set` (unstable | stable | global) — Gaussian indices
are only comparable within the same set, which the recorder's mask-size +
version gate enforces naturally.

Note on BVH frustums: RTG applies the principal point (cx/cy) inside the
CUDA kernel, not in full_proj_transform; for non-centered cameras the BVH
frustum is approximate (recorded in meta).
"""
import os

_recorder = None
_frame_id = 0
_itr = 0
_bvh_phases = ("frame_render",)


def init(args):
    global _recorder
    cfg = getattr(args, "arch_stats", None) or {}
    if not cfg.get("enabled", False):
        return
    try:
        from gs_stats import WorkloadRecorder, is_bvh_available
    except ImportError:
        print("arch_stats: gs_stats not installed in this env; disabled")
        return
    import diff_gaussian_rasterization_depth as dgr

    dgr.set_stats_mode(True)
    bvh_cfg = {
        "enabled": bool(cfg.get("bvh_enabled", True) and is_bvh_available()),
        "scale_factor": float(cfg.get("bvh_scale_factor", 3.0)),
        "guard_band": float(cfg.get("bvh_guard_band", 0.0)),
        "phases": list(cfg.get("bvh_phases", _bvh_phases)),
    }
    meta = {
        "app": "RTG-SLAM",
        "save_path": args.save_path,
        "single_process": True,
        "rasterizer_gs_stats_build": bool(dgr.gs_stats_compiled()),
        "notes": {
            "gaussian_set": "unstable|stable|global — overlap only within "
                            "one set (enforced by mask-size/version gate)",
            "partial": "tile_mask-restricted render; n_inview_pairs counts "
                       "masked pairs only",
            "bvh_frustum": "cx/cy applied in-kernel, not in the projection "
                           "matrix — frustum approximate off-center",
        },
    }
    _recorder = WorkloadRecorder(
        out_dir=os.path.join(args.save_path, "arch_stats"),
        meta=meta,
        bvh_cfg=bvh_cfg,
        flush_every=int(cfg.get("flush_every", 200)),
    )


def enabled():
    return _recorder is not None


def set_frame(frame_id):
    global _frame_id, _itr
    _frame_id = int(frame_id)
    _itr = 0


def set_itr(itr):
    global _itr
    _itr = int(itr)


# which Gaussian set each phase renders (index space of the masks)
SET_BY_PHASE = {
    "frame_render": "global", "map_local": "global", "map_global": "stable",
    "eval_range_unstable": "unstable", "eval_range_stable": "stable",
    "error_remove": "global", "attach": "stable", "eval": "global",
}


def record_render(results, stats_tag, camera, n_total, model_like,
                  partial, n_masked_tiles):
    if _recorder is None or stats_tag is None:
        return
    _recorder.record(results, stats_tag, _frame_id, view_id=_frame_id,
                     itr=_itr, model=model_like, camera=camera,
                     n_total=n_total, partial=partial,
                     extra={"gaussian_set": SET_BY_PHASE.get(stats_tag),
                            "n_masked_tiles": n_masked_tiles,
                            "num_tile": results.get("num_tile")})


def notify_scene_change(reason=""):
    if _recorder is not None:
        _recorder.bump_scene_version(reason)


def notify_param_change(reason=""):
    if _recorder is not None:
        _recorder.bump_param_version(reason)


def close():
    global _recorder
    if _recorder is not None:
        _recorder.close()
        _recorder = None
