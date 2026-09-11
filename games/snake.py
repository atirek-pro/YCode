"""
Classic Snake Game - Retro Arcade Edition
=========================================
Enhanced Features:
- Dynamic Snake color changing on eating food with multi-color body gradients.
- Speed increases every 5 points with "SPEED UP!" notifications and live Speed HUD.
- 5 Authentic Retro Game Color Palettes (Synthwave Cyberpunk, Pico-8, Game Boy Classic, Vaporwave, NES Arcade).
- Theme toggle with 'T' key anytime during gameplay.
- Particle burst effects on eating food.
- Directional eyes with pupil animation, glowing food, and smooth retro UI.
"""

import sys
import os
import json
import math
import random
import pygame

# ---------------------------------------------------------
# Window & Grid Specifications
# ---------------------------------------------------------
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
BLOCK_SIZE = 20

GRID_WIDTH = WINDOW_WIDTH // BLOCK_SIZE    # 40 cols
GRID_HEIGHT = WINDOW_HEIGHT // BLOCK_SIZE  # 30 rows

BASE_SPEED = 10
SPEED_INCREMENT_INTERVAL = 5  # Increase speed every 5 points
SPEED_INCREMENT_AMOUNT = 2    # +2 FPS every 5 points
MAX_SPEED = 28

# File to store persistent high score in JSON format
SCORE_FILE = "snake_highscore.json"

# ---------------------------------------------------------
# Curated Retro Game Color Palettes
# ---------------------------------------------------------
RETRO_PALETTES = [
    {
        "name": "SYNTHWAVE CYBERPUNK",
        "bg": (12, 15, 29),
        "grid": (22, 28, 50),
        "snake_colors": [
            (0, 245, 212),    # Neon Cyan
            (255, 42, 133),   # Hot Pink
            (155, 93, 229),   # Electric Violet
            (254, 228, 64),   # Laser Yellow
            (0, 187, 249),    # Capri Blue
            (241, 91, 181),   # Neon Rose
        ],
        "food": (255, 0, 110),
        "food_glow": (255, 100, 180),
        "accent": (254, 228, 64),
        "hud_text": (240, 245, 255),
        "overlay": (10, 12, 22, 215),
    },
    {
        "name": "PICO-8 FANTASY",
        "bg": (29, 43, 83),
        "grid": (45, 60, 105),
        "snake_colors": [
            (0, 228, 54),     # Pico Green
            (41, 173, 255),   # Pico Sky Blue
            (255, 163, 0),    # Pico Orange
            (255, 236, 39),   # Pico Yellow
            (255, 119, 168),  # Pico Pink
            (131, 118, 156),  # Pico Lavender
        ],
        "food": (255, 0, 77),
        "food_glow": (255, 120, 150),
        "accent": (255, 236, 39),
        "hud_text": (255, 241, 232),
        "overlay": (15, 20, 45, 215),
    },
    {
        "name": "GAME BOY CLASSIC",
        "bg": (15, 56, 15),
        "grid": (25, 75, 25),
        "snake_colors": [
            (155, 188, 15),   # Lightest Green
            (139, 172, 15),   # Light Olive
            (120, 155, 15),   # Mid Olive
            (170, 200, 20),   # Bright Green
        ],
        "food": (155, 188, 15),
        "food_glow": (180, 210, 25),
        "accent": (155, 188, 15),
        "hud_text": (155, 188, 15),
        "overlay": (8, 30, 8, 225),
    },
    {
        "name": "NEON VAPORWAVE",
        "bg": (26, 9, 51),
        "grid": (45, 20, 80),
        "snake_colors": [
            (255, 113, 206),  # Vapor Pink
            (1, 205, 254),    # Ice Blue
            (5, 255, 161),    # Mint Green
            (185, 103, 255),  # Soft Purple
            (255, 251, 150),  # Pastel Yellow
        ],
        "food": (255, 113, 206),
        "food_glow": (255, 180, 230),
        "accent": (5, 255, 161),
        "hud_text": (255, 250, 255),
        "overlay": (18, 5, 38, 215),
    },
    {
        "name": "RETRO ARCADE 1980",
        "bg": (16, 16, 16),
        "grid": (32, 32, 32),
        "snake_colors": [
            (46, 204, 113),   # Emerald
            (52, 152, 219),   # Peter River
            (155, 89, 182),   # Amethyst
            (241, 196, 15),   # Sun Yellow
            (230, 126, 34),   # Carrot Orange
            (26, 188, 156),   # Turquoise
        ],
        "food": (231, 76, 60),
        "food_glow": (245, 140, 130),
        "accent": (241, 196, 15),
        "hud_text": (236, 240, 241),
        "overlay": (10, 10, 10, 220),
    }
]

# Directions (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

OPPOSITE_DIRECTIONS = {
    UP: DOWN,
    DOWN: UP,
    LEFT: RIGHT,
    RIGHT: LEFT,
}


# ---------------------------------------------------------
# Visual Effects (Particles & Popups)
# ---------------------------------------------------------
class FoodParticle:
    """Spark particles bursting when snake consumes food."""
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 4.5)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.color = color
        self.size = random.uniform(2.5, 5.0)
        self.alpha = 255.0
        self.decay = random.uniform(8.0, 14.0)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.alpha -= self.decay
        self.size = max(0.5, self.size - 0.1)
        return self.alpha > 0

    def draw(self, surface):
        if self.alpha <= 0:
            return
        s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        color_rgba = (*self.color, int(max(0, min(255, self.alpha))))
        pygame.draw.circle(s, color_rgba, (int(self.size), int(self.size)), int(self.size))
        surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))


class FloatingBanner:
    """Popup text notification for Speed Up and Milestones."""
    def __init__(self, text, x, y, color=(255, 255, 255), duration=40):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.duration = duration
        self.lifetime = 0
        self.alpha = 255.0
        self.vy = -1.2

    def update(self):
        self.y += self.vy
        self.lifetime += 1
        if self.lifetime > self.duration * 0.5:
            self.alpha = max(0.0, 255.0 * (1.0 - (self.lifetime - self.duration * 0.5) / (self.duration * 0.5)))
        return self.lifetime < self.duration

    def draw(self, surface, font):
        if self.alpha <= 0:
            return
        text_surf = font.render(self.text, True, self.color)
        text_surf.set_alpha(int(self.alpha))
        rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(text_surf, rect)


# ---------------------------------------------------------
# Main Game Class
# ---------------------------------------------------------
class SnakeGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Retro Arcade Snake")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_score = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_banner = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_title = pygame.font.SysFont("consolas", 50, bold=True)
        self.font_subtitle = pygame.font.SysFont("consolas", 22, bold=True)
        self.font_instruction = pygame.font.SysFont("consolas", 17)

        # Palette configuration
        self.palette_index = 0
        self.palette = RETRO_PALETTES[self.palette_index]

        self.high_score = self.load_high_score()
        self.new_record_notified = False
        self.particles = []
        self.floating_banners = []

        self.reset_game()

    def load_high_score(self):
        """Loads persistent best score from JSON file."""
        if os.path.exists(SCORE_FILE):
            try:
                with open(SCORE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return int(data.get("high_score", 0))
            except Exception:
                return 0
        return 0

    def save_high_score(self):
        """Saves persistent best score to JSON file."""
        try:
            with open(SCORE_FILE, "w", encoding="utf-8") as f:
                json.dump({"high_score": self.high_score}, f, indent=4)
        except Exception:
            pass

    def get_current_speed(self):
        """Calculates game speed: increases every 5 points."""
        speed = BASE_SPEED + (self.score // SPEED_INCREMENT_INTERVAL) * SPEED_INCREMENT_AMOUNT
        return min(MAX_SPEED, speed)

    def cycle_palette(self):
        """Cycles to the next retro palette."""
        self.palette_index = (self.palette_index + 1) % len(RETRO_PALETTES)
        self.palette = RETRO_PALETTES[self.palette_index]
        self.floating_banners.append(
            FloatingBanner(f"THEME: {self.palette['name']}", WINDOW_WIDTH // 2, 80, self.palette["accent"], duration=50)
        )

    def reset_game(self):
        """Resets game state for a new round."""
        start_x = GRID_WIDTH // 4
        start_y = GRID_HEIGHT // 2

        self.snake = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
            (start_x - 3, start_y),
        ]
        self.direction = RIGHT
        self.next_direction = RIGHT
        self.score = 0
        self.color_shift = 0  # Advances color sequence when food is eaten
        self.new_record_notified = False
        self.game_over = False
        self.particles.clear()
        self.floating_banners.clear()
        self.food = self.spawn_food()

    def spawn_food(self):
        """Generates random coordinates for food outside the snake body."""
        all_cells = [
            (x, y)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
            if (x, y) not in self.snake
        ]
        if all_cells:
            return random.choice(all_cells)
        return (0, 0)

    def handle_events(self):
        """Processes keyboard input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.quit_game()

                # Toggle Retro Theme
                if event.key == pygame.K_t:
                    self.cycle_palette()
                    continue

                if self.game_over:
                    if event.key in (pygame.K_r, pygame.K_SPACE):
                        self.reset_game()
                    elif event.key == pygame.K_q:
                        self.quit_game()
                else:
                    # Direction controls (Arrows & WASD)
                    if event.key in (pygame.K_UP, pygame.K_w):
                        if OPPOSITE_DIRECTIONS.get(self.direction) != UP:
                            self.next_direction = UP
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        if OPPOSITE_DIRECTIONS.get(self.direction) != DOWN:
                            self.next_direction = DOWN
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        if OPPOSITE_DIRECTIONS.get(self.direction) != LEFT:
                            self.next_direction = LEFT
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        if OPPOSITE_DIRECTIONS.get(self.direction) != RIGHT:
                            self.next_direction = RIGHT

    def update(self):
        """Updates game state: snake movement, eating food, speed milestones, collisions."""
        # Update FX
        self.particles = [p for p in self.particles if p.update()]
        self.floating_banners = [fb for fb in self.floating_banners if fb.update()]

        if self.game_over:
            return

        self.direction = self.next_direction
        cur_x, cur_y = self.snake[0]
        dx, dy = self.direction
        new_head = (cur_x + dx, cur_y + dy)
        head_x, head_y = new_head

        # Advance Snake with Screen Wrapping (Toroidal Boundary)
        head_x = (cur_x + dx) % GRID_WIDTH
        head_y = (cur_y + dy) % GRID_HEIGHT
        new_head = (head_x, head_y)

        # Self Collision Check (Game only ends when hitting its own body)
        # Note: If new_head is the tail and no food is eaten, the tail will move out of the way
        body_to_check = self.snake[:-1] if new_head != self.food else self.snake
        if new_head in body_to_check:
            self.trigger_game_over()
            return

        # Advance Snake
        self.snake.insert(0, new_head)

        # Food Collision
        if new_head == self.food:
            old_speed = self.get_current_speed()
            self.score += 1
            self.color_shift += 1  # Shift snake color palette on each bite!
            
            # Persistent Best Score Tracking
            if self.score > self.high_score:
                if self.high_score > 0 and not self.new_record_notified:
                    self.floating_banners.append(
                        FloatingBanner("*** NEW BEST SCORE! ***", WINDOW_WIDTH // 2, 110, self.palette["accent"], duration=55)
                    )
                    self.new_record_notified = True
                self.high_score = self.score
                self.save_high_score()

            # Spawn eating particles
            fx = head_x * BLOCK_SIZE + BLOCK_SIZE // 2
            fy = head_y * BLOCK_SIZE + BLOCK_SIZE // 2
            for _ in range(12):
                self.particles.append(FoodParticle(fx, fy, self.palette["food_glow"]))

            # Check for Speed Increase every 5 points
            new_speed = self.get_current_speed()
            if new_speed > old_speed:
                self.floating_banners.append(
                    FloatingBanner(f">> SPEED UP! ({new_speed} FPS) <<", WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 40, self.palette["accent"], duration=45)
                )

            self.food = self.spawn_food()
        else:
            self.snake.pop()

    def trigger_game_over(self):
        self.game_over = True
        if self.score > self.high_score:
            self.high_score = self.score
            self.save_high_score()

    # ---------------------------------------------------------
    # RENDERING
    # ---------------------------------------------------------
    def draw_grid(self):
        """Draws subtle retro grid lines."""
        grid_col = self.palette["grid"]
        for x in range(0, WINDOW_WIDTH, BLOCK_SIZE):
            pygame.draw.line(self.screen, grid_col, (x, 0), (x, WINDOW_HEIGHT), 1)
        for y in range(0, WINDOW_HEIGHT, BLOCK_SIZE):
            pygame.draw.line(self.screen, grid_col, (0, y), (WINDOW_WIDTH, y), 1)

    def draw_snake(self):
        """
        Draws snake body with dynamic color shifting as it eats food.
        Each body segment gets an animated retro gradient color.
        """
        colors = self.palette["snake_colors"]
        num_colors = len(colors)

        for index, (segment_x, segment_y) in enumerate(self.snake):
            rect = pygame.Rect(
                segment_x * BLOCK_SIZE + 1,
                segment_y * BLOCK_SIZE + 1,
                BLOCK_SIZE - 2,
                BLOCK_SIZE - 2,
            )

            # Calculate color based on segment index + food color shift
            color_idx = (index + self.color_shift) % num_colors
            base_color = colors[color_idx]

            if index == 0:
                # Snake Head with bright highlight border
                pygame.draw.rect(self.screen, base_color, rect, border_radius=6)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1, border_radius=6)
                self.draw_eyes(rect)
            else:
                # Snake Body segments with rounded bevel
                pygame.draw.rect(self.screen, base_color, rect, border_radius=4)
                # Soft inner highlight
                highlight_rect = pygame.Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height - 4)
                pygame.draw.rect(self.screen, (255, 255, 255, 50), highlight_rect, 1, border_radius=3)

    def draw_eyes(self, head_rect):
        """Draws stylized directional eyes on the snake head."""
        eye_radius = 3
        pupil_radius = 1
        dx, dy = self.direction

        if dx == 1:  # Facing Right
            eye1 = (head_rect.right - 5, head_rect.top + 5)
            eye2 = (head_rect.right - 5, head_rect.bottom - 6)
        elif dx == -1:  # Facing Left
            eye1 = (head_rect.left + 5, head_rect.top + 5)
            eye2 = (head_rect.left + 5, head_rect.bottom - 6)
        elif dy == -1:  # Facing Up
            eye1 = (head_rect.left + 5, head_rect.top + 5)
            eye2 = (head_rect.right - 6, head_rect.top + 5)
        else:  # Facing Down
            eye1 = (head_rect.left + 5, head_rect.bottom - 6)
            eye2 = (head_rect.right - 6, head_rect.bottom - 6)

        for eye in (eye1, eye2):
            pygame.draw.circle(self.screen, (255, 255, 255), eye, eye_radius)
            pygame.draw.circle(self.screen, (15, 17, 25), eye, pupil_radius)

    def draw_food(self):
        """Draws glowing food item."""
        food_x, food_y = self.food
        center_x = food_x * BLOCK_SIZE + BLOCK_SIZE // 2
        center_y = food_y * BLOCK_SIZE + BLOCK_SIZE // 2
        radius = (BLOCK_SIZE // 2) - 2

        # Pulsing outer glow
        pulse = 1.0 + 0.25 * math.sin(pygame.time.get_ticks() * 0.008)
        glow_radius = int(radius * pulse) + 2
        
        glow_surf = pygame.Surface((glow_radius * 2 + 4, glow_radius * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self.palette["food_glow"], 90), (glow_radius + 2, glow_radius + 2), glow_radius)
        self.screen.blit(glow_surf, (center_x - glow_radius - 2, center_y - glow_radius - 2))

        # Food body
        pygame.draw.circle(self.screen, self.palette["food"], (center_x, center_y), radius)
        # Specular shine
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x - 2, center_y - 2), 2)

    def draw_hud(self):
        """Draws HUD displaying Score, High Score, Speed, and Theme."""
        hud_color = self.palette["hud_text"]
        accent_color = self.palette["accent"]
        current_speed = self.get_current_speed()
        speed_level = (self.score // SPEED_INCREMENT_INTERVAL) + 1

        # Score & High Score
        score_surf = self.font_score.render(f"SCORE: {self.score}", True, hud_color)
        high_surf = self.font_score.render(f"BEST: {self.high_score}", True, accent_color)
        
        # Speed & Level Indicator
        speed_surf = self.font_score.render(f"SPEED: LVL {speed_level} ({current_speed} FPS)", True, accent_color)
        theme_surf = self.font_instruction.render(f"[T] THEME: {self.palette['name']}", True, hud_color)

        self.screen.blit(score_surf, (20, 14))
        self.screen.blit(speed_surf, (180, 14))
        self.screen.blit(high_surf, (WINDOW_WIDTH - high_surf.get_width() - 20, 14))
        self.screen.blit(theme_surf, (20, WINDOW_HEIGHT - 26))

    def draw_game_over(self):
        """Draws retro Game Over modal card."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(self.palette["overlay"])
        self.screen.blit(overlay, (0, 0))

        # Title
        title_surf = self.font_title.render("GAME OVER", True, self.palette["food"])
        title_rect = title_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 80))
        self.screen.blit(title_surf, title_rect)

        # Stats
        speed_level = (self.score // SPEED_INCREMENT_INTERVAL) + 1
        score_surf = self.font_subtitle.render(
            f"Final Score: {self.score}  |  Level reached: {speed_level}  |  Best: {self.high_score}",
            True,
            self.palette["hud_text"],
        )
        score_rect = score_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 - 20))
        self.screen.blit(score_surf, score_rect)

        # Instructions
        restart_surf = self.font_instruction.render(
            "PRESS [R] OR [SPACE] TO PLAY AGAIN", True, self.palette["accent"]
        )
        restart_rect = restart_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 40))
        self.screen.blit(restart_surf, restart_rect)

        quit_surf = self.font_instruction.render(
            "PRESS [Q] OR [ESC] TO QUIT   |   [T] SWITCH THEME", True, self.palette["hud_text"]
        )
        quit_rect = quit_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 75))
        self.screen.blit(quit_surf, quit_rect)

    def draw(self):
        """Renders entire frame."""
        self.screen.fill(self.palette["bg"])
        self.draw_grid()
        self.draw_food()
        self.draw_snake()

        # Render FX
        for p in self.particles:
            p.draw(self.screen)
        for fb in self.floating_banners:
            fb.draw(self.screen, self.font_banner)

        self.draw_hud()

        if self.game_over:
            self.draw_game_over()

        pygame.display.flip()

    def run(self):
        """Main loop driven by dynamic speed."""
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.get_current_speed())

    def quit_game(self):
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = SnakeGame()
    game.run()
