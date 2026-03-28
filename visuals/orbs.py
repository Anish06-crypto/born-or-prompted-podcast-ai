"""
3D pulsing orb visualizer for Podcast AI.

Run in the main thread via run_visuals().
Signal speaker state from any thread via set_active_speaker() / signal_done().

Colours
-------
  Lyra   — violet / purple  (180, 80, 255)
  Cipher — teal / cyan      (0, 210, 180)
"""

import math
import pygame

# ---------------------------------------------------------------------------
# Shared state (written from playback thread, read from main/visuals thread)
# ---------------------------------------------------------------------------
_state = {
    "speaker": None,   # "Lyra" | "Cipher" | None
    "done":    False,
}

LYRA_COLOR   = (180,  80, 255)   # violet
CIPHER_COLOR = (  0, 210, 180)   # teal


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
LYRA_POS            = (230, 210)
CIPHER_POS          = (670, 210)

BAR_W, BAR_H = 120, 6
BAR_Y         = 330


def _draw_orb(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    base_r: int,
    color: tuple,
    pulse: float,
    t: float,
) -> None:
    r, g, b = color
    beat   = abs(math.sin(t * 7.0))
    draw_r = int(base_r * (1.0 + pulse * 0.18 * beat))

    # Glow rings
    if pulse > 0.05:
        for ring in range(3, 0, -1):
            alpha  = int(pulse * 22 / ring)
            glow_r = draw_r + ring * 16
            glow   = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            pygame.draw.circle(glow, (r, g, b, alpha), (cx, cy), glow_r)
            surface.blit(glow, (0, 0))

    # Drop shadow
    shadow = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    pygame.draw.circle(shadow, (0, 0, 0, 90), (cx + 7, cy + 10), draw_r)
    surface.blit(shadow, (0, 0))

    # Main sphere
    pygame.draw.circle(surface, (r, g, b), (cx, cy), draw_r)

    # Dark lower-right shading (light source top-left)
    shade = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    pygame.draw.circle(shade, (0, 0, 0, 80),
                       (cx + draw_r // 4, cy + draw_r // 4), draw_r)
    surface.blit(shade, (0, 0))

    # Specular highlight cluster top-left
    hi_cx = cx - draw_r // 3
    hi_cy = cy - draw_r // 3
    for step in range(6, 0, -1):
        ratio = step / 6
        alpha = int(190 * ratio)
        hi_r  = int(draw_r * 0.38 * ratio)
        hi    = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        pygame.draw.circle(hi, (255, 255, 255, alpha), (hi_cx, hi_cy), hi_r)
        surface.blit(hi, (0, 0))


def _draw_indicator(
    surface: pygame.Surface,
    cx: int,
    color: tuple,
    pulse: float,
) -> None:
    r, g, b = color
    alpha   = int(40 + pulse * 215)
    bar     = pygame.Surface((BAR_W, BAR_H), pygame.SRCALPHA)
    bar.fill((r, g, b, alpha))
    surface.blit(bar, (cx - BAR_W // 2, BAR_Y))

    if pulse > 0.3:
        line = pygame.Surface((BAR_W, 2), pygame.SRCALPHA)
        line.fill((255, 255, 255, int(pulse * 255)))
        surface.blit(line, (cx - BAR_W // 2, BAR_Y))

    if pulse > 0.1:
        gw   = int(BAR_W * (1 + pulse * 0.6))
        glow = pygame.Surface((gw, BAR_H * 4), pygame.SRCALPHA)
        glow.fill((r, g, b, int(pulse * 18)))
        surface.blit(glow, (cx - gw // 2, BAR_Y - BAR_H))


# ---------------------------------------------------------------------------
# Main loop (must run on the main thread on macOS)
# ---------------------------------------------------------------------------
def run_visuals() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("TechTalk — Lyra & Cipher")
    clock = pygame.time.Clock()

    pulse_lyra   = 0.0
    pulse_cipher = 0.0
    t = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        dt = clock.tick(60) / 1000.0
        t += dt

        speaker = _state["speaker"]
        speed   = 5.0
        pulse_lyra   += ((1.0 if speaker == "Lyra"   else 0.0) - pulse_lyra)   * speed * dt
        pulse_cipher += ((1.0 if speaker == "Cipher" else 0.0) - pulse_cipher) * speed * dt

        screen.fill(BG_COLOR)

        _draw_orb(screen, *LYRA_POS,   ORB_BASE_RADIUS, LYRA_COLOR,   pulse_lyra,   t)
        _draw_orb(screen, *CIPHER_POS, ORB_BASE_RADIUS, CIPHER_COLOR, pulse_cipher, t + math.pi)

        _draw_indicator(screen, LYRA_POS[0],   LYRA_COLOR,   pulse_lyra)
        _draw_indicator(screen, CIPHER_POS[0], CIPHER_COLOR, pulse_cipher)

        if _state["done"] and speaker is None:
            pygame.time.wait(1800)
            running = False

        pygame.display.flip()

    pygame.quit()
