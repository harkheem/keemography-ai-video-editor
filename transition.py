# transition.py

def list_available_transitions():
    return [
        "crossfade",
        "slide_left", "slide_right", "slide_up", "slide_down",
        "zoom_in", "zoom_out",
        "fadein", "fadeout",
    ]


def apply_transition(clip1, clip2, transition_type: str, duration: float):
    """
    Returns a single MoviePy clip representing clip1 with the specified
    transition blended into clip2 at its tail.

    Transitions implemented:
      crossfade     — smooth opacity blend
      slide_left    — clip1 exits left, clip2 enters from right
      slide_right   — clip1 exits right, clip2 enters from left
      slide_up      — clip1 exits upward, clip2 enters from below
      slide_down    — clip1 exits downward, clip2 enters from above
      zoom_in       — clip2 scales from ~45% to 100% as it appears
      zoom_out      — clip1 shrinks away as clip2 fades in
      fadein/out    — clip1 fades to black; clip2 fades up from black
    """
    try:
        from moviepy import CompositeVideoClip, vfx
    except Exception:
        from moviepy.editor import CompositeVideoClip, vfx

    d = max(0.1, float(duration))
    start = max(clip1.duration - d, 0.0)   # when overlap begins in clip1 timeline
    W, H = clip1.size

    # ── compat helpers ────────────────────────────────────────────────────────

    def _with_start(clip, t):
        return clip.with_start(t) if hasattr(clip, "with_start") else clip.set_start(t)

    def _with_duration(clip, t):
        return clip.with_duration(t) if hasattr(clip, "with_duration") else clip.set_duration(t)

    def _fadein(clip, s):
        if hasattr(clip, "fadein"):
            return clip.fadein(s)
        return clip.with_effects([vfx.FadeIn(s)])

    def _fadeout(clip, s):
        if hasattr(clip, "fadeout"):
            return clip.fadeout(s)
        return clip.with_effects([vfx.FadeOut(s)])

    def _set_pos(clip, fn):
        if hasattr(clip, "with_position"):
            return clip.with_position(fn)
        return clip.set_position(fn)

    def _apply_fl(clip, func):
        """Apply per-frame func(get_frame, local_t). local_t is clip-local time.

        MoviePy 2.x renamed Clip.fl → Clip.transform (same signature); without
        this branch the zoom transitions silently degraded to plain fades."""
        if hasattr(clip, "transform"):
            return clip.transform(func)
        if hasattr(clip, "fl"):
            return clip.fl(func)
        return clip   # best-effort: skip effect if neither API exists

    def _smoothstep(x: float) -> float:
        """Cubic Hermite ease-in/out."""
        x = max(0.0, min(1.0, x))
        return x * x * (3.0 - 2.0 * x)

    # ── transitions ───────────────────────────────────────────────────────────

    def _crossfade():
        a = _fadeout(_with_start(clip1, 0), d)
        b = _fadein(_with_start(clip2, start), d)
        return _with_duration(CompositeVideoClip([a, b]), start + d)

    def _slide(dx_exit, dy_exit, dx_enter, dy_enter):
        """
        dx/dy are full-pixel displacements at p=1.
        Position callbacks receive GLOBAL composite time.
        """
        a = _with_start(clip1, 0)
        b = _with_start(clip2, start)

        def pos_a(gt):
            p = _smoothstep(max(0.0, gt - start) / max(d, 0.001))
            return (int(dx_exit * p), int(dy_exit * p))

        def pos_b(gt):
            p = _smoothstep(max(0.0, gt - start) / max(d, 0.001))
            return (int(dx_enter * (1.0 - p)), int(dy_enter * (1.0 - p)))

        return _with_duration(
            CompositeVideoClip([_set_pos(a, pos_a), _set_pos(b, pos_b)], size=(W, H)),
            start + d,
        )

    def _zoom_in():
        """clip2 scales from ~45% → 100% over its first d seconds (local time)."""
        a = _fadeout(_with_start(clip1, 0), d)
        b_raw = _with_start(clip2, start)

        def zoom_frame(get_frame, local_t):
            frame = get_frame(local_t)
            p = _smoothstep(local_t / max(d, 0.001))
            scale = 0.45 + 0.55 * p
            if scale >= 0.99:
                return frame
            import numpy as np
            from PIL import Image
            if frame.dtype != np.uint8:   # PIL rejects int64 frames (e.g. ColorClip)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            h, w = frame.shape[:2]
            nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
            img = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
            result = np.zeros((h, w, frame.shape[2]), dtype=np.uint8)
            ox, oy = (w - nw) // 2, (h - nh) // 2
            result[oy:oy + nh, ox:ox + nw] = np.asarray(img)
            return result

        b = _apply_fl(b_raw, zoom_frame)
        return _with_duration(CompositeVideoClip([a, b]), start + d)

    def _zoom_out():
        """clip1 shrinks 100% → ~45% during its last d seconds (local time)."""
        a_raw = _with_start(clip1, 0)

        def shrink_frame(get_frame, local_t):
            frame = get_frame(local_t)
            if local_t < start:
                return frame
            p = _smoothstep((local_t - start) / max(d, 0.001))
            scale = 1.0 - 0.55 * p
            if scale >= 0.99:
                return frame
            import numpy as np
            from PIL import Image
            if frame.dtype != np.uint8:   # PIL rejects int64 frames (e.g. ColorClip)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            h, w = frame.shape[:2]
            nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
            img = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
            result = np.zeros((h, w, frame.shape[2]), dtype=np.uint8)
            ox, oy = (w - nw) // 2, (h - nh) // 2
            result[oy:oy + nh, ox:ox + nw] = np.asarray(img)
            return result

        a = _apply_fl(a_raw, shrink_frame)
        b = _fadein(_with_start(clip2, start), d)
        return _with_duration(CompositeVideoClip([a, b]), start + d)

    # ── dispatch ──────────────────────────────────────────────────────────────
    t = (transition_type or "crossfade").lower()

    try:
        if t == "slide_left":
            return _slide(-W, 0,  W, 0)
        if t == "slide_right":
            return _slide( W, 0, -W, 0)
        if t == "slide_up":
            return _slide(0, -H, 0,  H)
        if t == "slide_down":
            return _slide(0,  H, 0, -H)
        if t in ("zoom_in", "grow"):
            return _zoom_in()
        if t in ("zoom_out", "shrink"):
            return _zoom_out()
        if t in ("fadein", "fadeout"):
            a = _fadeout(_with_start(clip1, 0), d * 0.7)
            b = _fadein(_with_start(clip2, start), d)
            return _with_duration(CompositeVideoClip([a, b]), start + d)
    except Exception as e:
        print(f"⚠️ Transition '{t}' failed ({repr(e)}), falling back to crossfade")

    return _crossfade()
