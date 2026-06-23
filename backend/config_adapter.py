"""
config_adapter.py
-----------------
Translates the mobile-app (Eyezer) request payload into the
backend config dict expected by Orchestrator + LedController.

Mobile request shape (JSON from ConfigScreen):
    {
        "participant": { "name": str, "age": int, "sex": str },
        "controlMode": "dual" | "left_to_right" | "right_to_left",
        "schedule": {
            # dual
            "flashes": [{"hex": "#RRGGBB", "duration": seconds}, ...],
            "gap": seconds
              OR
            # sequential
            "rounds": int, "hex": "#RRGGBB", "duration": seconds,
            "innerPause": seconds, "gap": seconds
        },

        # Legacy single-eye payload remains accepted:
        "eye":         "Left" | "Right" | "Both",
        "color":       "Red" | "Green" | "Blue" | "Yellow" | "White" | "All",
        "iterations":  int,     # number of flashes per color
        "duration":    float,   # seconds, flash on-time
        "delay":       float,   # seconds, gap between flashes
        "intensity":   float,   # retained for API compatibility; HIGH/LOW driver ignores it
        "gpio":        { ...optional pin overrides... },
        "common_anode": bool | { "led1": bool, "led2": bool }
    }

Backend config shape (what Orchestrator wants):
    {
        led1_r, led1_g, led1_b, led2_r, led2_g, led2_b: int,
        common_anode | led1_common_anode/led2_common_anode: bool,
        mode: "dual" | "left_to_right" | "right_to_left",
        schedule: {
            flashes: [ {hex, duration_ms, intensity_pct}, ... ],   # dual
              or
            rounds, hex, duration_ms, intensity_pct,
            inner_pause_ms, gap_ms                                 # sequential
        },
        camera: { output_dir: str }
    }
"""

from led_controller import DEFAULT_PINS, DEFAULT_COMMON_ANODE


COLOR_HEX = {
    "Red":    "#FF0000",
    "Green":  "#00FF00",
    "Blue":   "#0000FF",
    "Yellow": "#FFFF00",
    "White":  "#FFFFFF",
}

# Order used when "All" is requested — same set the eyezer UI offers.
ALL_COLORS = ["Red", "Green", "Blue", "Yellow", "White"]


def adapt(payload: dict) -> dict:
    explicit_mode = payload.get("controlMode") or payload.get("control_mode")
    if explicit_mode is not None:
        return _adapt_explicit_control_mode(payload, explicit_mode)

    return _adapt_legacy_mobile_payload(payload)


def _hardware_config(payload: dict) -> dict:
    """Return GPIO and wiring settings shared by both mobile contracts."""
    gpio_in = payload.get("gpio") or {}
    pins = {k: int(gpio_in.get(k, DEFAULT_PINS[k])) for k in DEFAULT_PINS}
    ca_in = payload.get("common_anode")
    if isinstance(ca_in, dict):
        wiring = {
            "led1_common_anode": bool(ca_in.get("led1", DEFAULT_COMMON_ANODE["led1"])),
            "led2_common_anode": bool(ca_in.get("led2", DEFAULT_COMMON_ANODE["led2"])),
        }
    elif ca_in is None:
        wiring = {
            "led1_common_anode": DEFAULT_COMMON_ANODE["led1"],
            "led2_common_anode": DEFAULT_COMMON_ANODE["led2"],
        }
    else:
        wiring = {"common_anode": bool(ca_in)}
    return {**pins, **wiring}


def _seconds_to_ms(value, minimum_ms=50):
    return max(minimum_ms, int(float(value) * 1000))


def _validated_hex(value):
    text = str(value or "").upper()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError(f"invalid RGB hex color: {value!r}")
    try:
        int(text[1:], 16)
    except ValueError as exc:
        raise ValueError(f"invalid RGB hex color: {value!r}") from exc
    return text


def _adapt_explicit_control_mode(payload: dict, mode: str) -> dict:
    if mode not in {"dual", "left_to_right", "right_to_left"}:
        raise ValueError(f"unsupported controlMode: {mode!r}")

    incoming = payload.get("schedule") or {}
    intensity = float(payload.get("intensity", 100.0))

    if mode == "dual":
        incoming_flashes = incoming.get("flashes") or []
        if not incoming_flashes:
            raise ValueError("dual mode requires at least one flash")
        flashes = [
            {
                "hex": _validated_hex(flash.get("hex")),
                "duration_ms": _seconds_to_ms(flash.get("duration", 1.0)),
                "intensity_pct": intensity,
            }
            for flash in incoming_flashes
        ]
        schedule = {
            "flashes": flashes,
            "gap_ms": _seconds_to_ms(incoming.get("gap", 3.0), minimum_ms=3000),
        }
    else:
        schedule = {
            "rounds": max(1, int(incoming.get("rounds", 3))),
            "hex": _validated_hex(incoming.get("hex", "#FFFFFF")),
            "duration_ms": _seconds_to_ms(incoming.get("duration", 1.0)),
            "inner_pause_ms": _seconds_to_ms(incoming.get("innerPause", 1.0)),
            "gap_ms": _seconds_to_ms(incoming.get("gap", 3.0), minimum_ms=3000),
            "intensity_pct": intensity,
        }

    return {
        **_hardware_config(payload),
        "mode": mode,
        "schedule": schedule,
        "camera": {"output_dir": "recordings"},
        "participant": payload.get("participant", {}),
        "pre_roll_s": 1.0,
        "post_roll_s": 3.0,
    }


def _adapt_legacy_mobile_payload(payload: dict) -> dict:
    eye        = payload.get("eye", "Left")
    color      = payload.get("color", "White")
    iterations = max(1, int(payload.get("iterations", 1)))
    duration_s = float(payload.get("duration", 1.0))
    delay_s    = float(payload.get("delay", 1.0))
    intensity  = float(payload.get("intensity", 100.0))

    duration_ms = max(50, int(duration_s * 1000))
    # Initial PLR analysis observes three seconds after each flash. Enforce a
    # matching inter-flash gap so the next flash does not contaminate that window.
    gap_ms      = max(3000, int(delay_s * 1000))

    # ── Mode ─────────────────────────────────────────────────────────────────
    if eye == "Both":
        mode = "dual"
    elif eye == "Right":
        mode = "right_to_left"   # right LED only — sequential w/ rounds=iter
    else:
        mode = "left_to_right"   # default = Left

    # ── Schedule ─────────────────────────────────────────────────────────────
    colors = ALL_COLORS if color == "All" else [color]

    if mode == "dual":
        flashes = []
        for c in colors:
            for _ in range(iterations):
                flashes.append({
                    "hex":           COLOR_HEX[c],
                    "duration_ms":   duration_ms,
                    "intensity_pct": intensity,
                })
        schedule = {"flashes": flashes, "gap_ms": gap_ms}
    else:
        # Sequential modes use a single color per round. When "All" is
        # selected we still loop colors round-by-round.
        # Total rounds = iterations × len(colors); each round uses one color.
        # We achieve this by replicating the schedule with a colors list.
        schedule = {
            "rounds":          iterations * len(colors),
            "colors_cycle":    [COLOR_HEX[c] for c in colors],   # orchestrator may consume
            "hex":             COLOR_HEX[colors[0]],             # backwards-compat single color
            "duration_ms":     duration_ms,
            "intensity_pct":   intensity,
            "inner_pause_ms":  max(50, gap_ms // 2),
            "gap_ms":          gap_ms,
            # Sequential mode in current orchestrator fires both LEDs per round.
            # For single-eye intent we mark which side is active; the other side
            # will be skipped (orchestrator change handles this).
            "active_led":      1 if mode == "left_to_right" else 2,
        }

    return {
        **_hardware_config(payload),
        "mode":     mode,
        "schedule": schedule,
        "camera":   {"output_dir": "recordings"},
        "participant": payload.get("participant", {}),
        "pre_roll_s": 1.0,
        "post_roll_s": 3.0,
    }
