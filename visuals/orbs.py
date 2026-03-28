"""
3D pulsing orb visualizer for Podcast AI.

Run in the main thread via run_visuals().
Signal speaker state from any thread via set_active_speaker() / signal_done().
"""

import math
import pygame

# ---------------------------------------------------------------------------
# Shared state (written from playback thread, read from main/visuals thread)
# ---------------------------------------------------------------------------
_state = {
    "speaker": None,   # "Alex" | "Sam" | None
    "done": False,
}

ALEX_COLOR = (80,  140, 255)   # cool blue
SAM_COLOR  = (255, 110,  50)   # warm orange


def set_active_speaker(name: str | None) -> None:
    _state["speaker"] = name


def signal_done() -> None:
    _state["done"] = True


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
SCREEN_W, SCREEN_H = 900, 500
BG_COLOR            = (8, 8, 18)
ORB_BASE_RADIUS     = 88
ALEX_POS            = (230, 210)
SAM_POS             = (670, 210)

# Indicator bar dimensions (replaces text labels)
BAR_W, BAR_H = 120, 6
BAR_Y         = 330   # below orbs


def _draw_orb(surface: pygame.Surface, cx: int, cy: int,
              base_r: int, color: tuple, pulse: float, t: float) -> None:
    """
    Render a shaded sphere with:
      - smooth pulse on the radius
      - glow rings when active
      - drop shadow
      - specular highlight for 3D depth
    """
    r, g, b = color

    # Animated radius
    beat = abs(math.sin(t * 7.0))
    draw_r = int(base_r * (1.0 + pulse * 0.18 * beat))

    # --- Glow rings ---
    if pulse > 0.05:
        for ring in range(3, 0, -1):
            alpha = int(pulse * 22 / ring)
            glow_r = draw_r + ring * 16
            glow = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            pygame.draw.circle(glow, (r, g, b, alpha), (cx, cy), glow_r)
            surface.blit(glow, (0, 0))

    # --- Drop shadow ---
    shadow = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    pygame.draw.circle(shadow, (0, 0, 0, 90), (cx + 7, cy + 10), draw_r)
    surface.blit(shadow, (0, 0))

    # --- Main sphere body ---
    pygame.draw.circle(surface, (r, g, b), (cx, cy), draw_r)

    # --- Dark shading on the lower-right to fake a light source top-left ---
    shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    pygame.draw.circle(shade, (0, 0, 0, 80),
                       (cx + draw_r // 4, cy + draw_r // 4), draw_r)
    surface.blit(shade, (0, 0))

    # --- Specular highlight: soft gradient cluster top-left ---
    hi_cx = cx - draw_r // 3
    hi_cy = cy - draw_r // 3
    for step in range(6, 0, -1):
        ratio = step / 6
        alpha = int(190 * ratio)
        hi_r  = int(draw_r * 0.38 * ratio)
        hi    = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        pygame.draw.circle(hi, (255, 255, 255, alpha), (hi_cx, hi_cy), hi_r)
        surface.blit(hi, (0, 0))


def _draw_indicator(surface: pygame.Surface, cx: int, color: tuple, pulse: float) -> None:
    """
    Draw a small rounded bar below the orb.
    Glows brightly when the speaker is active, dims when idle.
    """
    r, g, b = color
    alpha = int(40 + pulse * 215)
    bar_surf = pygame.Surface((BAR_W, BAR_H), pygame.SRCALPHA)
    bar_surf.fill((r, g, b, alpha))
    surface.blit(bar_surf, (cx - BAR_W // 2, BAR_Y))

    # Thin bright line on top when active
    if pulse > 0.3:
        line_alpha = int(pulse * 255)
        line_surf = pygame.Surface((BAR_W, 2), pygame.SRCALPHA)
        line_surf.fill((255, 255, 255, line_alpha))
        surface.blit(line_surf, (cx - BAR_W // 2, BAR_Y))

    # Active speaker: add a larger diffuse glow bar behind it
    if pulse > 0.1:
        glow_w = int(BAR_W * (1 + pulse * 0.6))
        glow_surf = pygame.Surface((glow_w, BAR_H * 4), pygame.SRCALPHA)
        glow_surf.fill((r, g, b, int(pulse * 18)))
        surface.blit(glow_surf, (cx - glow_w // 2, BAR_Y - BAR_H))


# ---------------------------------------------------------------------------
# Main loop (must run on the main thread on macOS)
# ---------------------------------------------------------------------------
def run_visuals() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Podcast AI")
    clock = pygame.time.Clock()

    pulse_alex = 0.0
    pulse_sam  = 0.0
    t = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.tick(60) / 1000.0
        t += dt

        speaker = _state["speaker"]

        # Smooth lerp toward target pulse (1.0 = speaking, 0.0 = idle)
        speed = 5.0
        pulse_alex += ((1.0 if speaker == "Alex" else 0.0) - pulse_alex) * speed * dt
        pulse_sam  += ((1.0 if speaker == "Sam"  else 0.0) - pulse_sam)  * speed * dt

        # Background
        screen.fill(BG_COLOR)

        # Orbs
        _draw_orb(screen, *ALEX_POS, ORB_BASE_RADIUS, ALEX_COLOR, pulse_alex, t)
        _draw_orb(screen, *SAM_POS,  ORB_BASE_RADIUS, SAM_COLOR,  pulse_sam,  t + math.pi)

        # Indicator bars (replace text labels)
        _draw_indicator(screen, ALEX_POS[0], ALEX_COLOR, pulse_alex)
        _draw_indicator(screen, SAM_POS[0],  SAM_COLOR,  pulse_sam)

        # Auto-close a moment after playback finishes
        if _state["done"] and speaker is None:
            pygame.time.wait(1800)
            running = False

        pygame.display.flip()

    pygame.quit()
