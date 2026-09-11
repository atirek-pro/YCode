"""
Tetris in Pygame - Single File Implementation
=============================================
Features:
- Standard 7-bag piece generation
- Ghost piece preview
- Hold piece functionality (C / Left Shift)
- Next piece queue preview (up to 3 pieces)
- Wall kick rotation system (SRS-inspired)
- DAS (Delayed Auto Shift) & ARR for responsive movement
- Authentic scoring system with level progression & combos
- Multi-block vanish effects (flashing, shrinking, particle bursts, screen shake)
- Floating popup notifications (Single, Double, Triple, TETRIS!, Combos)
- Polished 3D beveled block graphics & neon arcade UI
- Pause (P / ESC) and Game Over screen with Restart (R / Space)
"""

import sys
import math
import random
import pygame

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 720
FPS = 60

# Grid specifications
GRID_COLS = 10
GRID_ROWS = 20
BLOCK_SIZE = 30  # Pixel size of each block in playfield

# Playfield pixel boundaries
PLAYFIELD_W = GRID_COLS * BLOCK_SIZE  # 300px
PLAYFIELD_H = GRID_ROWS * BLOCK_SIZE  # 600px
PLAYFIELD_X = (SCREEN_WIDTH - PLAYFIELD_W) // 2  # 250px
PLAYFIELD_Y = (SCREEN_HEIGHT - PLAYFIELD_H) // 2  # 60px

# Color Palette (Arcade / Cyberpunk style)
BG_COLOR = (15, 17, 26)
BG_PANEL = (24, 28, 42)
BG_PANEL_BORDER = (45, 55, 80)
GRID_BG = (10, 12, 18)
GRID_LINE_COLOR = (25, 30, 45)
BORDER_ACCENT = (0, 220, 255)
GHOST_COLOR = (70, 80, 110)

TEXT_COLOR = (240, 245, 255)
TEXT_MUTED = (140, 155, 185)
TEXT_HIGHLIGHT = (255, 215, 0)

# Tetromino Vibrant Colors: (Main, Highlight, Shadow)
TETROMINO_COLORS = {
    'I': ((0, 220, 235), (140, 245, 255), (0, 150, 170)),    # Cyan
    'J': ((40, 90, 240), (120, 160, 255), (20, 50, 170)),   # Blue
    'L': ((245, 140, 20), (255, 195, 90), (180, 95, 10)),   # Orange
    'O': ((245, 215, 20), (255, 240, 110), (185, 160, 10)), # Yellow
    'S': ((50, 215, 75), (130, 245, 145), (30, 155, 50)),   # Green
    'T': ((175, 45, 230), (220, 120, 255), (120, 25, 165)), # Purple
    'Z': ((235, 45, 65), (255, 125, 140), (170, 25, 40)),   # Red
}

# Tetromino Shapes matrix definition (4x4 coordinate grids for 4 rotations)
SHAPES = {
    'I': [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)]
    ],
    'J': [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)]
    ],
    'L': [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)]
    ],
    'O': [
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)]
    ],
    'S': [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)]
    ],
    'T': [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)]
    ],
    'Z': [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 0), (0, 1), (1, 1), (0, 2)]
    ]
}

# Wall kick offsets to test on rotation (dx, dy)
WALL_KICK_OFFSETS = [
    (0, 0), (-1, 0), (1, 0), (0, -1), (-1, -1), (1, -1), (-2, 0), (2, 0), (0, -2)
]

# -----------------------------------------------------------------------------
# VISUAL EFFECTS (Particles & Floating Text)
# -----------------------------------------------------------------------------
class Particle:
    """Disintegrating block particle with physics and fade."""
    def __init__(self, x, y, color):
        self.x = float(x)
        self.y = float(y)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 6.0)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - random.uniform(1.0, 3.0)
        self.color = color
        self.size = random.uniform(3, 7)
        self.alpha = 255.0
        self.decay = random.uniform(5.0, 10.0)
        self.gravity = 0.25

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.alpha -= self.decay
        self.size = max(0.5, self.size - 0.08)
        return self.alpha > 0

    def draw(self, surface):
        if self.alpha <= 0:
            return
        s = pygame.Surface((int(self.size * 2), int(self.size * 2)), pygame.SRCALPHA)
        color_with_alpha = (*self.color, int(max(0, min(255, self.alpha))))
        pygame.draw.circle(s, color_with_alpha, (int(self.size), int(self.size)), int(self.size))
        surface.blit(s, (int(self.x - self.size), int(self.y - self.size)))


class FloatingText:
    """Floating animated popups for scoring, combos, and Tetris alerts."""
    def __init__(self, text, x, y, color=(255, 255, 255), size=28, duration=50):
        self.text = text
        self.x = float(x)
        self.y = float(y)
        self.color = color
        self.size = size
        self.alpha = 255.0
        self.duration = duration
        self.lifetime = 0
        self.vy = -1.2

    def update(self):
        self.y += self.vy
        self.lifetime += 1
        if self.lifetime > self.duration * 0.4:
            self.alpha = max(0.0, 255.0 * (1.0 - (self.lifetime - self.duration * 0.4) / (self.duration * 0.6)))
        return self.lifetime < self.duration

    def draw(self, surface, font_cache):
        if self.alpha <= 0:
            return
        font = font_cache.get(self.size)
        text_surf = font.render(self.text, True, self.color)
        text_surf.set_alpha(int(self.alpha))
        rect = text_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(text_surf, rect)


# -----------------------------------------------------------------------------
# TETROMINO PIECE LOGIC
# -----------------------------------------------------------------------------
class Piece:
    def __init__(self, shape_key):
        self.shape = shape_key
        self.rotation = 0
        self.x = 3  # Start near top center (10-wide grid)
        self.y = 0
        if shape_key == 'O':
            self.x = 4
        elif shape_key == 'I':
            self.y = -1

    def get_blocks(self, offset_x=0, offset_y=0, rot_offset=0):
        rot = (self.rotation + rot_offset) % len(SHAPES[self.shape])
        coords = []
        for bx, by in SHAPES[self.shape][rot]:
            coords.append((self.x + offset_x + bx, self.y + offset_y + by))
        return coords


# -----------------------------------------------------------------------------
# MAIN GAME ENGINE
# -----------------------------------------------------------------------------
class TetrisGame:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("TETRIS - Cyber Arcade Edition")

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_cache = {
            16: pygame.font.SysFont("Consolas", 16, bold=True),
            20: pygame.font.SysFont("Consolas", 20, bold=True),
            24: pygame.font.SysFont("Consolas", 24, bold=True),
            28: pygame.font.SysFont("Consolas", 28, bold=True),
            36: pygame.font.SysFont("Consolas", 36, bold=True),
            48: pygame.font.SysFont("Consolas", 48, bold=True),
            64: pygame.font.SysFont("Consolas", 64, bold=True),
        }

        self.high_score = 0
        self.reset_game()

    def reset_game(self):
        """Initializes a new game session."""
        self.grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]
        self.bag = []
        self.current_piece = self.get_next_from_bag()
        self.next_queue = [self.get_next_from_bag() for _ in range(3)]
        self.hold_piece = None
        self.can_hold = True

        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.combo = -1
        self.back_to_back = False

        # Drop Timing (frames & ms)
        self.fall_time = 0
        self.lock_delay = 30  # ~0.5s grace period on ground
        self.lock_timer = 0
        self.is_touching_ground = False

        # Line Vanish Animation State
        self.clearing_rows = []
        self.clear_anim_timer = 0
        self.clear_anim_max = 24  # Animation frame count for multi-block vanish
        self.shake_magnitude = 0
        self.shake_offset = (0, 0)

        # FX lists
        self.particles = []
        self.floating_texts = []

        # Key repeat (DAS - Delayed Auto Shift)
        self.key_left_down = False
        self.key_right_down = False
        self.key_down_down = False
        self.das_timer = 0
        self.das_delay = 12  # frames before auto repeat
        self.arr_interval = 2  # frames between repeated shifts

        # State
        self.game_over = False
        self.paused = False

    def get_next_from_bag(self):
        """Standard 7-Bag Randomizer generator."""
        if not self.bag:
            self.bag = ['I', 'J', 'L', 'O', 'S', 'T', 'Z']
            random.shuffle(self.bag)
        return Piece(self.bag.pop())

    def get_gravity_speed(self):
        """Calculates gravity fall interval in milliseconds based on level."""
        # Speed curve from classical tetris formula
        return max(50, int(800 * (0.85 ** (self.level - 1))))

    # -------------------------------------------------------------------------
    # COLLISION & PIECE PLACEMENT
    # -------------------------------------------------------------------------
    def is_valid_position(self, piece, offset_x=0, offset_y=0, rot_offset=0):
        """Checks if a piece can move to a given position and rotation."""
        for bx, by in piece.get_blocks(offset_x, offset_y, rot_offset):
            # Check left, right, bottom boundary
            if bx < 0 or bx >= GRID_COLS or by >= GRID_ROWS:
                return False
            # Check collision with already placed blocks
            if by >= 0 and self.grid[by][bx] is not None:
                return False
        return True

    def get_ghost_y(self):
        """Returns the lowest valid Y offset for the current piece (ghost drop)."""
        dy = 0
        while self.is_valid_position(self.current_piece, offset_y=dy + 1):
            dy += 1
        return self.current_piece.y + dy

    def lock_piece(self):
        """Locks the active piece into the grid and triggers line vanishing checks."""
        for bx, by in self.current_piece.get_blocks():
            if by < 0:
                # Placed above the screen -> Game Over
                self.game_over = True
                if self.score > self.high_score:
                    self.high_score = self.score
                return
            self.grid[by][bx] = self.current_piece.shape

        self.can_hold = True
        self.check_line_clears()

        if not self.clearing_rows:
            self.spawn_next_piece()

    def spawn_next_piece(self):
        """Pulls the next piece from the queue and tests game over."""
        self.current_piece = self.next_queue.pop(0)
        self.next_queue.append(self.get_next_from_bag())
        self.lock_timer = 0
        self.is_touching_ground = False

        if not self.is_valid_position(self.current_piece):
            self.game_over = True
            if self.score > self.high_score:
                self.high_score = self.score

    # -------------------------------------------------------------------------
    # ROTATION & WALL KICKS
    # -------------------------------------------------------------------------
    def try_rotate(self, direction=1):
        """Rotates piece with wall-kick testing."""
        target_rot = (self.current_piece.rotation + direction) % len(SHAPES[self.current_piece.shape])
        for kx, ky in WALL_KICK_OFFSETS:
            if self.is_valid_position(self.current_piece, offset_x=kx, offset_y=ky, rot_offset=direction):
                self.current_piece.x += kx
                self.current_piece.y += ky
                self.current_piece.rotation = target_rot
                # Reset lock delay on successful maneuver
                if self.is_touching_ground:
                    self.lock_timer = 0
                return True
        return False

    # -------------------------------------------------------------------------
    # HOLD MECHANIC
    # -------------------------------------------------------------------------
    def trigger_hold(self):
        """Swaps the current piece into the hold slot."""
        if not self.can_hold:
            return
        
        current_shape = self.current_piece.shape
        if self.hold_piece is None:
            self.hold_piece = current_shape
            self.spawn_next_piece()
        else:
            prev_hold = self.hold_piece
            self.hold_piece = current_shape
            self.current_piece = Piece(prev_hold)
        
        self.can_hold = False
        self.lock_timer = 0
        self.is_touching_ground = False

    # -------------------------------------------------------------------------
    # HARD & SOFT DROPS
    # -------------------------------------------------------------------------
    def hard_drop(self):
        """Instantly drops the piece to the bottom and locks it."""
        dist = 0
        while self.is_valid_position(self.current_piece, offset_y=1):
            self.current_piece.y += 1
            dist += 1
        
        # Hard drop scoring (+2 per cell)
        self.score += dist * 2
        
        # Small screen impact shake
        self.shake_magnitude = 4
        
        # Spawn drop impact dust particles
        for bx, by in self.current_piece.get_blocks():
            px = PLAYFIELD_X + bx * BLOCK_SIZE + BLOCK_SIZE // 2
            py = PLAYFIELD_Y + (by + 1) * BLOCK_SIZE
            for _ in range(4):
                self.particles.append(Particle(px, py, TETROMINO_COLORS[self.current_piece.shape][0]))

        self.lock_piece()

    # -------------------------------------------------------------------------
    # MULTI-BLOCK VANISH & SCORING SYSTEM
    # -------------------------------------------------------------------------
    def check_line_clears(self):
        """Identifies full lines and initiates vanishing animation & particle bursts."""
        self.clearing_rows = []
        for r in range(GRID_ROWS):
            if all(self.grid[r][c] is not None for c in range(GRID_COLS)):
                self.clearing_rows.append(r)

        if self.clearing_rows:
            self.clear_anim_timer = self.clear_anim_max
            num_lines = len(self.clearing_rows)
            
            # Screen shake based on cleared count
            if num_lines == 4:
                self.shake_magnitude = 10  # TETRIS mega shake
            elif num_lines >= 2:
                self.shake_magnitude = 5

            # Spawn exploding vanish particles across clearing lines
            for r in self.clearing_rows:
                for c in range(GRID_COLS):
                    shape = self.grid[r][c]
                    color = TETROMINO_COLORS[shape][0] if shape in TETROMINO_COLORS else (255, 255, 255)
                    bx = PLAYFIELD_X + c * BLOCK_SIZE + BLOCK_SIZE // 2
                    by = PLAYFIELD_Y + r * BLOCK_SIZE + BLOCK_SIZE // 2
                    # 6-8 particles per vanishing tile
                    for _ in range(7):
                        self.particles.append(Particle(bx, by, color))

            # Proper Scoring Mechanics
            # 1 line = 100 * level, 2 = 300 * level, 3 = 500 * level, 4 = 800 * level
            base_scores = {1: 100, 2: 300, 3: 500, 4: 800}
            base_points = base_scores.get(num_lines, num_lines * 200) * self.level
            
            # Back-to-Back Tetris multiplier (1.5x)
            is_tetris = (num_lines == 4)
            if is_tetris and self.back_to_back:
                base_points = int(base_points * 1.5)
            self.back_to_back = is_tetris

            # Combo system
            self.combo += 1
            if self.combo > 0:
                base_points += 50 * self.combo * self.level

            self.score += base_points
            self.lines_cleared += num_lines

            # Level up every 10 lines
            self.level = (self.lines_cleared // 10) + 1

            # Floating Announcement Text
            mid_y = PLAYFIELD_Y + (sum(self.clearing_rows) / len(self.clearing_rows)) * BLOCK_SIZE
            center_x = PLAYFIELD_X + PLAYFIELD_W // 2

            if num_lines == 1:
                self.floating_texts.append(FloatingText("SINGLE!", center_x, mid_y, (100, 220, 255), size=24))
            elif num_lines == 2:
                self.floating_texts.append(FloatingText("DOUBLE!", center_x, mid_y, (100, 255, 150), size=28))
            elif num_lines == 3:
                self.floating_texts.append(FloatingText("TRIPLE!", center_x, mid_y, (255, 200, 50), size=32))
            elif num_lines == 4:
                text = "B2B TETRIS!" if (self.back_to_back and self.combo > 0) else "TETRIS!"
                self.floating_texts.append(FloatingText(text, center_x, mid_y, (255, 60, 100), size=42, duration=70))

            if self.combo > 1:
                self.floating_texts.append(FloatingText(f"{self.combo} COMBO", center_x, mid_y + 35, TEXT_HIGHLIGHT, size=20))
        else:
            self.combo = -1

    def finish_line_clears(self):
        """Collapses the cleared rows down after vanish animation completes."""
        # Create new grid with cleared rows removed
        new_grid = [row for i, row in enumerate(self.grid) if i not in self.clearing_rows]
        # Insert empty rows at the top
        for _ in range(len(self.clearing_rows)):
            new_grid.insert(0, [None for _ in range(GRID_COLS)])
        self.grid = new_grid
        self.clearing_rows = []
        self.spawn_next_piece()

    # -------------------------------------------------------------------------
    # MAIN UPDATE TICK
    # -------------------------------------------------------------------------
    def update(self, dt):
        """Updates game simulation, timers, animations, and inputs."""
        # Screen Shake decay
        if self.shake_magnitude > 0:
            self.shake_offset = (
                random.randint(-self.shake_magnitude, self.shake_magnitude),
                random.randint(-self.shake_magnitude, self.shake_magnitude)
            )
            self.shake_magnitude = max(0, self.shake_magnitude - 1)
        else:
            self.shake_offset = (0, 0)

        # Update visual particles & popups
        self.particles = [p for p in self.particles if p.update()]
        self.floating_texts = [ft for ft in self.floating_texts if ft.update()]

        if self.paused or self.game_over:
            return

        # Vanishing line animation handler
        if self.clearing_rows:
            self.clear_anim_timer -= 1
            if self.clear_anim_timer <= 0:
                self.finish_line_clears()
            return

        # DAS (Delayed Auto Shift) Movement Handling for smooth controls
        if self.key_left_down:
            self.das_timer += 1
            if self.das_timer >= self.das_delay and (self.das_timer - self.das_delay) % self.arr_interval == 0:
                if self.is_valid_position(self.current_piece, offset_x=-1):
                    self.current_piece.x -= 1
        elif self.key_right_down:
            self.das_timer += 1
            if self.das_timer >= self.das_delay and (self.das_timer - self.das_delay) % self.arr_interval == 0:
                if self.is_valid_position(self.current_piece, offset_x=1):
                    self.current_piece.x += 1

        # Gravity & Soft Drop
        self.fall_time += dt
        gravity_interval = self.get_gravity_speed()
        if self.key_down_down:
            gravity_interval = min(40, gravity_interval // 8)  # Fast soft drop

        if self.fall_time >= gravity_interval:
            self.fall_time = 0
            if self.is_valid_position(self.current_piece, offset_y=1):
                self.current_piece.y += 1
                if self.key_down_down:
                    self.score += 1  # Soft drop point
                self.is_touching_ground = False
                self.lock_timer = 0
            else:
                self.is_touching_ground = True

        # Lock Delay Handling
        if not self.is_valid_position(self.current_piece, offset_y=1):
            self.is_touching_ground = True
            self.lock_timer += 1
            if self.lock_timer >= self.lock_delay:
                self.lock_piece()
        else:
            self.is_touching_ground = False
            self.lock_timer = 0

    # -------------------------------------------------------------------------
    # RENDERING HELPERS
    # -------------------------------------------------------------------------
    def draw_block(self, surface, x, y, shape, alpha=255, scale=1.0):
        """Draws a single 3D-beveled arcade tetromino block."""
        if shape not in TETROMINO_COLORS:
            return
        
        main_col, high_col, shadow_col = TETROMINO_COLORS[shape]

        b_size = int(BLOCK_SIZE * scale)
        offset = (BLOCK_SIZE - b_size) // 2
        bx = int(x + offset)
        by = int(y + offset)

        block_surf = pygame.Surface((b_size, b_size), pygame.SRCALPHA)

        # Base fill
        pygame.draw.rect(block_surf, (*main_col, alpha), (0, 0, b_size, b_size))

        # Top & Left bevel highlights
        bevel = max(2, int(b_size * 0.12))
        pygame.draw.polygon(block_surf, (*high_col, alpha), [
            (0, 0), (b_size, 0), (b_size - bevel, bevel), (bevel, bevel), (bevel, b_size - bevel), (0, b_size)
        ])
        # Bottom & Right bevel shadows
        pygame.draw.polygon(block_surf, (*shadow_col, alpha), [
            (b_size, 0), (b_size, b_size), (0, b_size), (bevel, b_size - bevel), (b_size - bevel, b_size - bevel), (b_size - bevel, bevel)
        ])

        # Center tile inner glow
        pygame.draw.rect(block_surf, (*main_col, alpha), (bevel, bevel, b_size - 2 * bevel, b_size - 2 * bevel))
        
        # Subtle outer rim
        pygame.draw.rect(block_surf, (0, 0, 0, int(alpha * 0.4)), (0, 0, b_size, b_size), 1)

        surface.blit(block_surf, (bx, by))

    def draw_ghost_block(self, surface, x, y, shape):
        """Draws translucent outline for landing ghost piece preview."""
        if shape not in TETROMINO_COLORS:
            return
        main_col = TETROMINO_COLORS[shape][0]
        rect = pygame.Rect(x + 1, y + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2)
        
        # Ghost interior with soft glow
        ghost_surf = pygame.Surface((BLOCK_SIZE - 2, BLOCK_SIZE - 2), pygame.SRCALPHA)
        ghost_surf.fill((*main_col, 45))
        surface.blit(ghost_surf, (x + 1, y + 1))
        
        # Ghost outline
        pygame.draw.rect(surface, (*main_col, 130), rect, 2, border_radius=3)

    def draw_panel(self, surface, rect, title=""):
        """Draws a sleek arcade UI panel box."""
        # Main background box
        pygame.draw.rect(surface, BG_PANEL, rect, border_radius=8)
        pygame.draw.rect(surface, BG_PANEL_BORDER, rect, 2, border_radius=8)

        if title:
            title_surf = self.font_cache[16].render(title, True, TEXT_MUTED)
            surface.blit(title_surf, (rect.x + 14, rect.y + 10))

    # -------------------------------------------------------------------------
    # MAIN DRAW ROUTINE
    # -------------------------------------------------------------------------
    def draw(self):
        """Renders the entire game screen."""
        self.screen.fill(BG_COLOR)

        # Apply screen shake offset to playfield
        sx, sy = self.shake_offset

        # 1. Draw Playfield Background
        grid_rect = pygame.Rect(PLAYFIELD_X + sx, PLAYFIELD_Y + sy, PLAYFIELD_W, PLAYFIELD_H)
        pygame.draw.rect(self.screen, GRID_BG, grid_rect)

        # Grid lines
        for c in range(1, GRID_COLS):
            lx = PLAYFIELD_X + sx + c * BLOCK_SIZE
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (lx, PLAYFIELD_Y + sy), (lx, PLAYFIELD_Y + sy + PLAYFIELD_H), 1)
        for r in range(1, GRID_ROWS):
            ly = PLAYFIELD_Y + sy + r * BLOCK_SIZE
            pygame.draw.line(self.screen, GRID_LINE_COLOR, (PLAYFIELD_X + sx, ly), (PLAYFIELD_X + sx + PLAYFIELD_W, ly), 1)

        # 2. Draw Locked Grid Blocks with Multi-Block Vanish effect
        for r in range(GRID_ROWS):
            is_vanishing = r in self.clearing_rows
            if is_vanishing:
                # Flashing and shrinking animation
                progress = 1.0 - (self.clear_anim_timer / self.clear_anim_max)
                flash_alpha = int(255 * (1.0 - progress))
                scale = max(0.1, 1.0 - progress * 0.8)
                flash_col = (255, 255, 255)
            else:
                flash_alpha = 255
                scale = 1.0

            for c in range(GRID_COLS):
                shape = self.grid[r][c]
                if shape:
                    bx = PLAYFIELD_X + sx + c * BLOCK_SIZE
                    by = PLAYFIELD_Y + sy + r * BLOCK_SIZE
                    if is_vanishing:
                        # Draw white flash block then disintegrate
                        s = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
                        s.fill((*flash_col, flash_alpha))
                        self.screen.blit(s, (bx, by))
                        self.draw_block(self.screen, bx, by, shape, alpha=flash_alpha, scale=scale)
                    else:
                        self.draw_block(self.screen, bx, by, shape)

        # 3. Draw Ghost Piece (Drop Projection)
        if not self.clearing_rows and not self.game_over:
            ghost_y = self.get_ghost_y()
            for bx, by in self.current_piece.get_blocks(offset_y=ghost_y - self.current_piece.y):
                if by >= 0:
                    px = PLAYFIELD_X + sx + bx * BLOCK_SIZE
                    py = PLAYFIELD_Y + sy + by * BLOCK_SIZE
                    self.draw_ghost_block(self.screen, px, py, self.current_piece.shape)

        # 4. Draw Active Falling Piece
        if not self.clearing_rows and not self.game_over:
            for bx, by in self.current_piece.get_blocks():
                if by >= 0:
                    px = PLAYFIELD_X + sx + bx * BLOCK_SIZE
                    py = PLAYFIELD_Y + sy + by * BLOCK_SIZE
                    self.draw_block(self.screen, px, py, self.current_piece.shape)

        # Playfield Neon Border
        pygame.draw.rect(self.screen, BORDER_ACCENT, grid_rect, 2, border_radius=4)

        # 5. Draw Left UI Panel: HOLD & CONTROLS
        left_x = PLAYFIELD_X - 180
        hold_rect = pygame.Rect(left_x, PLAYFIELD_Y, 150, 140)
        self.draw_panel(self.screen, hold_rect, "HOLD (C/Shift)")

        if self.hold_piece:
            # Center hold tetromino in the hold box
            hold_shape = self.hold_piece
            blocks = SHAPES[hold_shape][0]
            min_x = min(bx for bx, by in blocks)
            max_x = max(bx for bx, by in blocks)
            min_y = min(by for bx, by in blocks)
            max_y = max(by for bx, by in blocks)
            pw = (max_x - min_x + 1) * 22
            ph = (max_y - min_y + 1) * 22
            ox = hold_rect.centerx - pw // 2 - min_x * 22
            oy = hold_rect.centery + 10 - ph // 2 - min_y * 22
            alpha = 255 if self.can_hold else 90
            for bx, by in blocks:
                self.draw_block(self.screen, ox + bx * 22, oy + by * 22, hold_shape, alpha=alpha, scale=22 / BLOCK_SIZE)

        # Controls panel
        ctrl_rect = pygame.Rect(left_x, PLAYFIELD_Y + 160, 150, 440)
        self.draw_panel(self.screen, ctrl_rect, "CONTROLS")
        controls_text = [
            ("Move", "< > / A D"),
            ("Rotate CW", "^ / W / X"),
            ("Rotate CCW", "Z / Ctrl"),
            ("Soft Drop", "v / S"),
            ("Hard Drop", "SPACE"),
            ("Hold", "C / Shift"),
            ("Pause", "P / ESC"),
            ("Restart", "R"),
        ]
        curr_y = ctrl_rect.y + 38
        for label, key in controls_text:
            l_surf = self.font_cache[16].render(label, True, TEXT_MUTED)
            k_surf = self.font_cache[16].render(key, True, TEXT_COLOR)
            self.screen.blit(l_surf, (ctrl_rect.x + 14, curr_y))
            self.screen.blit(k_surf, (ctrl_rect.x + 14, curr_y + 18))
            curr_y += 48

        # 6. Draw Right UI Panel: NEXT QUEUE & STATS
        right_x = PLAYFIELD_X + PLAYFIELD_W + 30
        
        # Next Pieces Box
        next_rect = pygame.Rect(right_x, PLAYFIELD_Y, 150, 260)
        self.draw_panel(self.screen, next_rect, "NEXT PIECES")
        
        for idx, next_p in enumerate(self.next_queue):
            blocks = SHAPES[next_p.shape][0]
            min_x = min(bx for bx, by in blocks)
            max_x = max(bx for bx, by in blocks)
            min_y = min(by for bx, by in blocks)
            max_y = max(by for bx, by in blocks)
            pw = (max_x - min_x + 1) * 20
            ph = (max_y - min_y + 1) * 20
            slot_y = next_rect.y + 45 + idx * 72
            ox = next_rect.centerx - pw // 2 - min_x * 20
            oy = slot_y + 25 - ph // 2 - min_y * 20
            for bx, by in blocks:
                self.draw_block(self.screen, ox + bx * 20, oy + by * 20, next_p.shape, scale=20 / BLOCK_SIZE)

        # Score & Stats Box
        stats_rect = pygame.Rect(right_x, PLAYFIELD_Y + 280, 150, 320)
        self.draw_panel(self.screen, stats_rect, "STATISTICS")

        stat_items = [
            ("SCORE", str(self.score), TEXT_HIGHLIGHT, 28),
            ("HIGH SCORE", str(self.high_score), (255, 180, 50), 24),
            ("LEVEL", str(self.level), (0, 220, 255), 24),
            ("LINES", str(self.lines_cleared), (120, 240, 140), 24),
        ]
        sy_curr = stats_rect.y + 36
        for title, val, col, font_sz in stat_items:
            t_surf = self.font_cache[16].render(title, True, TEXT_MUTED)
            v_surf = self.font_cache[font_sz].render(val, True, col)
            self.screen.blit(t_surf, (stats_rect.x + 14, sy_curr))
            self.screen.blit(v_surf, (stats_rect.x + 14, sy_curr + 18))
            sy_curr += 68

        # 7. Draw Visual FX Particles & Floating Banners
        for p in self.particles:
            p.draw(self.screen)

        for ft in self.floating_texts:
            ft.draw(self.screen, self.font_cache)

        # 8. Overlays: Pause & Game Over
        if self.paused:
            self.draw_overlay("PAUSED", "Press 'P' or 'ESC' to Resume", (255, 200, 50))
        elif self.game_over:
            self.draw_game_over_overlay()

        pygame.display.flip()

    def draw_overlay(self, title, subtitle, color):
        """Generic dark overlay for pause / modal messages."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 12, 18, 200))
        self.screen.blit(overlay, (0, 0))

        t_surf = self.font_cache[48].render(title, True, color)
        t_rect = t_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 25))
        self.screen.blit(t_surf, t_rect)

        s_surf = self.font_cache[20].render(subtitle, True, TEXT_COLOR)
        s_rect = s_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
        self.screen.blit(s_surf, s_rect)

    def draw_game_over_overlay(self):
        """Arcade Game Over summary card."""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((10, 12, 18, 225))
        self.screen.blit(overlay, (0, 0))

        card_w, card_h = 420, 320
        card_rect = pygame.Rect((SCREEN_WIDTH - card_w) // 2, (SCREEN_HEIGHT - card_h) // 2, card_w, card_h)
        pygame.draw.rect(self.screen, BG_PANEL, card_rect, border_radius=12)
        pygame.draw.rect(self.screen, (255, 50, 75), card_rect, 2, border_radius=12)

        go_surf = self.font_cache[48].render("GAME OVER", True, (255, 60, 80))
        go_rect = go_surf.get_rect(center=(card_rect.centerx, card_rect.y + 45))
        self.screen.blit(go_surf, go_rect)

        # Final Score
        sc_label = self.font_cache[16].render("FINAL SCORE", True, TEXT_MUTED)
        sc_val = self.font_cache[36].render(f"{self.score:,}", True, TEXT_HIGHLIGHT)
        self.screen.blit(sc_label, (card_rect.x + 35, card_rect.y + 90))
        self.screen.blit(sc_val, (card_rect.x + 35, card_rect.y + 110))

        # Details
        det_text = f"Lines: {self.lines_cleared}    Level: {self.level}    High Score: {self.high_score:,}"
        det_surf = self.font_cache[16].render(det_text, True, TEXT_COLOR)
        det_rect = det_surf.get_rect(center=(card_rect.centerx, card_rect.y + 185))
        self.screen.blit(det_surf, det_rect)

        # Restart Prompt
        rst_surf = self.font_cache[20].render("PRESS [R] OR [SPACE] TO PLAY AGAIN", True, (0, 220, 255))
        rst_rect = rst_surf.get_rect(center=(card_rect.centerx, card_rect.y + 250))
        self.screen.blit(rst_surf, rst_rect)

    # -------------------------------------------------------------------------
    # INPUT & EVENT HANDLING
    # -------------------------------------------------------------------------
    def handle_events(self):
        """Processes keyboard inputs, hotkeys, and window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                # Global Exit
                if event.key == pygame.K_q and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    return False

                # Restart Key
                if event.key == pygame.K_r:
                    self.reset_game()
                    continue

                # Game Over controls
                if self.game_over:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self.reset_game()
                    continue

                # Pause toggle
                if event.key in (pygame.K_p, pygame.K_ESCAPE):
                    self.paused = not self.paused
                    continue

                if self.paused or self.clearing_rows:
                    continue

                # In-Game Controls
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    if self.is_valid_position(self.current_piece, offset_x=-1):
                        self.current_piece.x -= 1
                    self.key_left_down = True
                    self.key_right_down = False
                    self.das_timer = 0
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    if self.is_valid_position(self.current_piece, offset_x=1):
                        self.current_piece.x += 1
                    self.key_right_down = True
                    self.key_left_down = False
                    self.das_timer = 0
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.key_down_down = True
                elif event.key in (pygame.K_UP, pygame.K_w, pygame.K_x):
                    self.try_rotate(1)  # Clockwise
                elif event.key in (pygame.K_z, pygame.K_LCTRL, pygame.K_RCTRL):
                    self.try_rotate(-1)  # Counter-Clockwise
                elif event.key == pygame.K_SPACE:
                    self.hard_drop()
                elif event.key in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT):
                    self.trigger_hold()

            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    self.key_left_down = False
                    self.das_timer = 0
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.key_right_down = False
                    self.das_timer = 0
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.key_down_down = False

        return True

    # -------------------------------------------------------------------------
    # RUN LOOP
    # -------------------------------------------------------------------------
    def run(self):
        """Primary game loop."""
        running = True
        while running:
            dt = self.clock.tick(FPS)
            running = self.handle_events()
            self.update(dt)
            self.draw()

        pygame.quit()
        sys.exit()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    game = TetrisGame()
    game.run()
