# transition.py

def list_available_transitions():
    # Keep lightweight; no imports here so editor bootstrap can run first.
    return [
        "crossfade",
        "slide_left", "slide_right", "slide_up", "slide_down",
        "zoom_in", "zoom_out",
        "rotate",
        "fadein", "fadeout",
        "grow", "shrink",
    ]


def apply_transition(clip1, clip2, transition_type: str, duration: float):
    """
    Returns a SINGLE MoviePy clip that represents clip1 with the transition into clip2
    applied at the end of clip1. Your editor.py then appends clip2 (faded in), so
    our result should end exactly at the end of the overlap.

    If an effect isn't supported on the current platform, we fall back to crossfade.
    """
    # Lazy import so editor.py's bootstrap has time to install deps
    from moviepy.editor import CompositeVideoClip, vfx

    d = max(0.1, float(duration))
    start = max(clip1.duration - d, 0.0)  # when the overlap begins

    # Helper to compose two subclips a/b in the overlap window
    def _compose(a, b):
        # Base timeline: 'a' starts at 0 (it's the tail of clip1); 'b' starts at 'start'
        return CompositeVideoClip([a, b]).set_duration(start + d)

    if transition_type == "crossfade":
        a = clip1.set_start(0).crossfadeout(d)                       # full clip1 w/ tail fading out
        b = clip2.set_start(start).crossfadein(d)                    # head of clip2 fades in
        return _compose(a, b)

    elif transition_type in {"fadein", "fadeout"}:
        a = clip1.set_start(0).fadeout(d)
        b = clip2.set_start(start).fadein(d)
        return _compose(a, b)

    elif transition_type in {"slide_left", "slide_right", "slide_up", "slide_down"}:
        w, h = clip1.size

        def out_pos(t):
            ratio = min(max(t / d, 0.0), 1.0)
            if transition_type.endswith("left"):
                return (-(w * ratio), 0)
            if transition_type.endswith("right"):
                return (w * ratio, 0)
            if transition_type.endswith("up"):
                return (0, -(h * ratio))
            if transition_type.endswith("down"):
                return (0, h * ratio)
            return (0, 0)

        def in_pos(t):
            ratio = 1.0 - min(max(t / d, 0.0), 1.0)
            if transition_type.endswith("left"):
                return (w * ratio, 0)
            if transition_type.endswith("right"):
                return (-w * ratio, 0)
            if transition_type.endswith("up"):
                return (0, h * ratio)
            if transition_type.endswith("down"):
                return (0, -h * ratio)
            return (0, 0)

        a = clip1.subclip(start).set_start(start).set_position(out_pos)
        b = clip2.subclip(0, d).set_start(start).set_position(in_pos)
        return CompositeVideoClip([clip1.set_start(0), a, b]).set_duration(start + d)

    elif transition_type == "zoom_in":
        # clip1 zooms in, clip2 zooms from small -> normal
        a_tail = clip1.subclip(start).fx(vfx.resize, lambda t: 1 + 0.2 * (t / d)).set_start(start)
        b_head = clip2.subclip(0, d).fx(vfx.resize, lambda t: max(0.1, 1 - 0.2 * (t / d))).set_start(start)
        return CompositeVideoClip([clip1.set_start(0), a_tail, b_head]).set_duration(start + d)

    elif transition_type == "zoom_out":
        a_tail = clip1.subclip(start).fx(vfx.resize, lambda t: max(0.1, 1 - 0.2 * (t / d))).set_start(start)
        b_head = clip2.subclip(0, d).fx(vfx.resize, lambda t: 1 + 0.2 * (t / d)).set_start(start)
        return CompositeVideoClip([clip1.set_start(0), a_tail, b_head]).set_duration(start + d)

    elif transition_type == "rotate":
        a_tail = clip1.subclip(start).rotate(lambda t: 15 * (t / d)).set_start(start)
        b_head = clip2.subclip(0, d).rotate(lambda t: -15 * (t / d)).set_start(start)
        return CompositeVideoClip([clip1.set_start(0), a_tail, b_head]).set_duration(start + d)

    elif transition_type in {"grow", "shrink"}:
        # Map to simple scale changes similar to zoom in/out
        if transition_type == "grow":
            a_tail = clip1.subclip(start).fx(vfx.resize, lambda t: 1 + 0.15 * (t / d)).set_start(start)
            b_head = clip2.subclip(0, d).fx(vfx.resize, lambda t: max(0.1, 1 - 0.15 * (t / d))).set_start(start)
        else:  # shrink
            a_tail = clip1.subclip(start).fx(vfx.resize, lambda t: max(0.1, 1 - 0.15 * (t / d))).set_start(start)
            b_head = clip2.subclip(0, d).fx(vfx.resize, lambda t: 1 + 0.15 * (t / d)).set_start(start)
        return CompositeVideoClip([clip1.set_start(0), a_tail, b_head]).set_duration(start + d)

    # Fallback: crossfade
    a = clip1.set_start(0).fadeout(d)
    b = clip2.set_start(start).fadein(d)
    return CompositeVideoClip([a, b]).set_duration(start + d)
