# transition.py
from moviepy.editor import CompositeVideoClip, concatenate_videoclips, vfx

def list_available_transitions():
    return [
        "crossfade", "slide_left", "slide_right", "slide_up", "slide_down",
        "zoom_in", "zoom_out", "rotate", "fadein", "fadeout", "grow", "shrink"
    ]

def apply_transition(clip1, clip2, transition_type, duration):
    """
    Returns a list of clips that together represent:
      [clip1 (trimmed to end-dur), transition_segment (overlap), clip2 (rest)]
    """
    # Ensure both clips have audio/video duration available
    d = min(duration, max(0.1, clip1.duration, clip2.duration))

    # Pre and post segments
    pre = clip1.subclip(0, max(clip1.duration - d, 0.0))
    # Base overlap start time = pre.duration
    t0 = pre.duration

    if transition_type == "crossfade":
        a = clip1.subclip(max(clip1.duration - d, 0.0)).crossfadeout(d).set_start(t0)
        b = clip2.subclip(0, d).crossfadein(d).set_start(t0)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    elif transition_type in {"slide_left", "slide_right", "slide_up", "slide_down"}:
        w, h = clip1.size
        def pos_for(direction):
            if direction == "left":
                return lambda t: (w * (1 - t/d), 0)        # clip1 slides out to left
            if direction == "right":
                return lambda t: (-w * (1 - t/d), 0)
            if direction == "up":
                return lambda t: (0, h * (1 - t/d))
            if direction == "down":
                return lambda t: (0, -h * (1 - t/d))
            return lambda t: (0, 0)

        out_pos = pos_for(transition_type.split("_")[1])
        in_pos  = (lambda t: (0,0))  # clip2 slides in from opposite
        if "left"  in transition_type: in_pos  = lambda t: (-w * (t/d), 0)
        if "right" in transition_type: in_pos  = lambda t: ( w * (t/d), 0)
        if "up"    in transition_type: in_pos  = lambda t: (0, -h * (t/d))
        if "down"  in transition_type: in_pos  = lambda t: (0,  h * (t/d))

        a = clip1.subclip(max(clip1.duration - d, 0.0)).set_start(t0).set_position(out_pos)
        b = clip2.subclip(0, d).set_start(t0).set_position(in_pos)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    elif transition_type == "zoom_in":
        a = clip1.subclip(max(clip1.duration - d, 0.0)).fx(vfx.resize, lambda t: 1 + 0.1 * (t/d)).set_start(t0)
        b = clip2.subclip(0, d).fx(vfx.resize, lambda t: max(0.1, 1 - 0.1 * (t/d))).set_start(t0)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    elif transition_type == "zoom_out":
        a = clip1.subclip(max(clip1.duration - d, 0.0)).fx(vfx.resize, lambda t: max(0.1, 1 - 0.1 * (t/d))).set_start(t0)
        b = clip2.subclip(0, d).fx(vfx.resize, lambda t: 1 + 0.1 * (t/d)).set_start(t0)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    elif transition_type == "rotate":
        a = clip1.subclip(max(clip1.duration - d, 0.0)).rotate(lambda t:  15 * (t/d)).set_start(t0)
        b = clip2.subclip(0, d).rotate(lambda t: -15 * (t/d)).set_start(t0)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    elif transition_type in {"fadein", "fadeout", "grow", "shrink"}:
        # Map simple effects into a crossfade-like overlap to keep API consistent
        a = clip1.subclip(max(clip1.duration - d, 0.0)).fadeout(d).set_start(t0)
        b = clip2.subclip(0, d).fadein(d).set_start(t0)
        overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
        post = clip2.subclip(d)
        return [pre, overlap, post]

    # Fallback to crossfade
    a = clip1.subclip(max(clip1.duration - d, 0.0)).crossfadeout(d).set_start(t0)
    b = clip2.subclip(0, d).crossfadein(d).set_start(t0)
    overlap = CompositeVideoClip([a, b]).set_duration(t0 + d)
    post = clip2.subclip(d)
    return [pre, overlap, post]
