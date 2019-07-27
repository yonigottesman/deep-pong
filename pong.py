import random
from Direction import Direction
import pygame

from PlayerTracker import PlayerTracker
from State import State
from PlayerHuman import PlayerHuman


class Paddle(pygame.Rect):
    def __init__(self, velocity,player, *args, **kwargs):
        self.velocity = velocity
        self.player = player
        super().__init__(*args, **kwargs)

    def move_paddle(self, board_height, ball):
        direction = self.player.get_move(State(pygame, ball_location=(ball.x, ball.y),
                                               paddle_location=(self.x, self.y)))

        if direction is Direction.UP:
            if self.y - self.velocity > 0:
                self.y -= self.velocity

        if direction is Direction.DOWN:
            if self.y + self.velocity < board_height - self.height:
                self.y += self.velocity


class Ball(pygame.Rect):
    def __init__(self, velocity, *args, **kwargs):
        self.velocity = velocity
        self.angle = 0
        super().__init__(*args, **kwargs)

    def move_ball(self):
        self.x += self.velocity
        self.y += self.angle


class Pong:
    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    HEIGHT = 700
    WIDTH = 500

    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 100

    BALL_WIDTH = 10
    VELOCITY = 10

    COLOUR = (255, 255, 255)

    def __init__(self, player_a, player_b):
        self.paddles = []
        pygame.init()  # Start the pygame instance.
        self.central_line = pygame.Rect(self.WIDTH / 2, 0, 1, self.HEIGHT)
        # Setup the screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        self.init_objects()

        self.score_a = 0
        self.score_b = 0

        self.player_a = player_a
        self.player_b = player_b

    def init_objects(self):
        # Create the player objects.

        self.paddle_a = Paddle(  # The left paddle
            self.VELOCITY,
            player_a,
            0,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.paddle_b = Paddle(  # The right paddle
            self.VELOCITY,
            player_b,
            self.WIDTH - self.PADDLE_WIDTH,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.paddles.append(self.paddle_a)
        self.paddles.append(self.paddle_b)

        self.ball = Ball(
            self.VELOCITY,
            self.WIDTH / 2 - self.BALL_WIDTH / 2,
            self.HEIGHT / 2 - self.BALL_WIDTH / 2,
            self.BALL_WIDTH,
            self.BALL_WIDTH
        )

    def update_losers(self):

        if self.ball.x > self.WIDTH:
            self.score_a = self.score_a + 1
            return True
        if self.ball.x < 0:
            self.score_b = self.score_b + 1
            return True
        return False

    def check_ball_hits_wall(self):
        if self.ball.y > self.HEIGHT - self.BALL_WIDTH or self.ball.y < 0:
            self.ball.angle = -self.ball.angle

    def check_ball_hits_paddle(self):
        for paddle in self.paddles:
            if self.ball.colliderect(paddle):
                self.ball.velocity = -self.ball.velocity
                self.ball.angle = random.randint(-10, 10)
                break

    def game_loop(self):
        while True:
            for event in pygame.event.get():
                # Add some extra ways to exit the game.
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            if self.update_losers():
                self.init_objects()
                continue

            self.check_ball_hits_paddle()
            self.check_ball_hits_wall()

            # Redraw the screen.
            self.screen.fill((0, 0, 0))

            self.paddle_a.move_paddle(self.HEIGHT, self.ball)
            pygame.draw.rect(self.screen, self.COLOUR, self.paddle_a)

            self.paddle_b.move_paddle(self.HEIGHT, self.ball)
            pygame.draw.rect(self.screen, self.COLOUR, self.paddle_b)

            self.ball.move_ball()
            pygame.draw.rect(self.screen, self.COLOUR, self.ball)

            pygame.draw.rect(self.screen, self.COLOUR, self.central_line)

            # Display scores:
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.score_a), 1, self.WHITE)
            self.screen.blit(text, (self.WIDTH / 4, 10))
            text = font.render(str(self.score_b), 1, self.WHITE)
            self.screen.blit(text, (self.WIDTH / 4 * 3, 10))

            pygame.display.flip()
            self.clock.tick(6000)


if __name__ == '__main__':
    # player_a = PlayerHuman(pygame.K_w, pygame.K_s)
    player_a = PlayerTracker()
    # player_b = PlayerHuman(pygame.K_UP, pygame.K_DOWN)
    player_b = PlayerTracker()
    pong = Pong(player_a, player_b)
    pong.game_loop()
