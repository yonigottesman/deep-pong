import random
import sys

import pygame

class Paddle(pygame.Rect):
    def __init__(self, velocity, up_key, down_key, *args, **kwargs):
        self.velocity = velocity
        self.up_key = up_key
        self.down_key = down_key
        super().__init__(*args, **kwargs)

    def move_paddle(self, board_height):
        keys_pressed = pygame.key.get_pressed()

        if keys_pressed[self.up_key]:
            if self.y - self.velocity > 0:
                self.y -= self.velocity

        if keys_pressed[self.down_key]:
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

    def __init__(self):
        pygame.init()  # Start the pygame instance.
        self.central_line = pygame.Rect(self.WIDTH / 2, 0, 1, self.HEIGHT)
        # Setup the screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        self.init_objects()

        self.score_a = 0
        self.score_b = 0

    def init_objects(self):
        # Create the player objects.
        self.paddles = []
        self.balls = []
        self.paddles.append(Paddle(  # The left paddle
            self.VELOCITY,
            pygame.K_w,
            pygame.K_s,
            0,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        ))

        self.paddles.append(Paddle(  # The right paddle
            self.VELOCITY,
            pygame.K_UP,
            pygame.K_DOWN,
            self.WIDTH - self.PADDLE_WIDTH,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        ))

        self.balls.append(Ball(
            self.VELOCITY,
            self.WIDTH / 2 - self.BALL_WIDTH / 2,
            self.HEIGHT / 2 - self.BALL_WIDTH / 2,
            self.BALL_WIDTH,
            self.BALL_WIDTH
        ))


    def update_losers(self):
        for ball in self.balls:
            if ball.x > self.WIDTH:
                self.score_a = self.score_a + 1
                return True
            if ball.x < 0:
                self.score_b = self.score_b + 1
                return True
        return False


    def check_ball_hits_wall(self):
        for ball in self.balls:
            if ball.y > self.HEIGHT - self.BALL_WIDTH or ball.y < 0:
                ball.angle = -ball.angle

    def check_ball_hits_paddle(self):
        for ball in self.balls:
            for paddle in self.paddles:
                if ball.colliderect(paddle):
                    ball.velocity = -ball.velocity
                    ball.angle = random.randint(-10, 10)
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

            for paddle in self.paddles:
                paddle.move_paddle(self.HEIGHT)
                pygame.draw.rect(self.screen, self.COLOUR, paddle)

            for ball in self.balls:
                ball.move_ball()
                pygame.draw.rect(self.screen, self.COLOUR, ball)
            pygame.draw.rect(self.screen, self.COLOUR, self.central_line)

            # Display scores:
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.score_a), 1, self.WHITE)
            self.screen.blit(text, (self.WIDTH/4, 10))
            text = font.render(str(self.score_b), 1, self.WHITE)
            self.screen.blit(text, (self.WIDTH/4*3, 10))

            pygame.display.flip()
            self.clock.tick(60)


if __name__ == '__main__':
    pong = Pong()
    pong.game_loop()