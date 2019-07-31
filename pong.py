import random
from Direction import Direction
import pygame

from PlayerDeep import PlayerDeep
from PlayerTracker import PlayerTracker
from State import State
import numpy as np
from PlayerHuman import PlayerHuman


class Paddle(pygame.Rect):
    def __init__(self, velocity, player, *args, **kwargs):
        self.velocity = velocity
        self.player = player
        super().__init__(*args, **kwargs)

    def move_paddle(self, board_height, ball, image_data, reward=0):
        direction = self.player.get_move(State(pygame, ball_location=(ball.x, ball.y),
                                               paddle_location=(self.x, self.y), reward=reward, image_data=image_data))

        if direction is Direction.UP:
            if self.y - self.velocity > 0:
                self.y -= self.velocity

        if direction is Direction.DOWN:
            if self.y + self.velocity < board_height - self.height:
                self.y += self.velocity


class Ball(pygame.Rect):
    def __init__(self, speed_x, speed_y, *args, **kwargs):
        self.speed_x = speed_x
        self.speed_y = speed_y

        direction = random.randint(0, 3)
        directions = {0: (-1, -1),
                      1: (-1, 1),
                      2: (1, 1),
                      3: (1, -1)}

        self.direction_x, self.direction_y = directions[direction]
        super().__init__(*args, **kwargs)

    def move_ball(self):
        # update the x and y position
        self.x = self.x + self.direction_x * self.speed_x
        self.y = self.y + self.direction_y * self.speed_y


class Pong:
    # Define some colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    HEIGHT = 400
    WIDTH = 400

    PADDLE_WIDTH = 10
    PADDLE_HEIGHT = 60
    PADDLE_SPEED = 8

    BALL_WIDTH = 10
    BALL_X_SPEED = 12
    BALL_Y_SPEED = 8
    VELOCITY = 10

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
            self.PADDLE_SPEED,
            player_a,
            0,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )

        self.paddle_b = Paddle(  # The right paddle
            self.PADDLE_SPEED,
            player_b,
            self.WIDTH - self.PADDLE_WIDTH,
            self.HEIGHT / 2 - self.PADDLE_HEIGHT / 2,
            self.PADDLE_WIDTH,
            self.PADDLE_HEIGHT
        )
        self.paddles.append(self.paddle_a)
        self.paddles.append(self.paddle_b)

        self.ball = Ball(
            self.BALL_X_SPEED,
            self.BALL_Y_SPEED,
            self.WIDTH / 2 - self.BALL_WIDTH / 2,
            random.randint(0, 9) * (self.HEIGHT - self.BALL_WIDTH)/9,
            self.BALL_WIDTH,
            self.BALL_WIDTH
        )

    def winner(self):
        if self.ball.x > self.WIDTH:
            return 'a'
        if self.ball.x < 0:
            return 'b'
        return 'no winner'

    def update_losers(self):

        if self.ball.x > self.WIDTH:
            self.score_a = self.score_a + 1
            return True
        if self.ball.x < 0:
            self.score_b = self.score_b + 1
            return True
        return False

    def check_ball_hits_wall(self):
        if self.ball.y >= self.HEIGHT - self.BALL_WIDTH/2 or self.ball.y <= 0:
            self.ball.direction_y *= -1
            self.ball.y = self.ball.y + self.ball.direction_y * 10

    def check_ball_hits_paddle(self):
        # ball_middle = (self.ball.midbottom - self.BALL_WIDTH/2), (self.ball.midright - self.BALL_WIDTH/2)
        # if self.ball.top + self.BALL_WIDTH/2 >

        for paddle in self.paddles:
            if self.ball.colliderect(paddle):
                self.ball.direction_x *= -1
                self.ball.x = self.ball.x + self.ball.direction_x * self.ball.speed_x
                # print('padle ' + str(paddle.top) + ' ' + str(paddle.bottom) + ' ball: ' + str(self.ball.top) +' '+ str(self.ball.bottom),end=' ')
                # print(str(self.ball.right) +' '+ str(paddle.right))

                break

    def game_loop(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            winner = self.winner()
            if winner == 'a':
                self.score_a += 1
                self.init_objects()

            if winner == 'b':
                self.score_b += 1
                self.init_objects()

            self.check_ball_hits_paddle()
            self.check_ball_hits_wall()

            image_data = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)[:, :, 0]

            # Redraw the screen.
            self.screen.fill((0, 0, 0))

            self.paddle_a.move_paddle(self.HEIGHT, self.ball,
                                      image_data=image_data,
                                      reward=1 if winner == 'a' else (-1 if winner == 'b' else 0))
            pygame.draw.rect(self.screen, self.WHITE, self.paddle_a)

            self.paddle_b.move_paddle(self.HEIGHT, self.ball,
                                      image_data=image_data,
                                      reward=1 if winner == 'b' else (-1 if winner == 'a' else 0))
            pygame.draw.rect(self.screen, self.WHITE, self.paddle_b)

            self.ball.move_ball()
            pygame.draw.rect(self.screen, self.WHITE, self.ball)

            pygame.draw.rect(self.screen, self.WHITE, self.central_line)

            # Display scores:
            font = pygame.font.Font(None, 74)
            text = font.render(str(self.score_a), 1, self.WHITE)
            self.screen.blit(text, (self.WIDTH / 4, 10))
            text = font.render(str(self.score_b), 1, self.WHITE)
            self.screen.blit(text, (self.WIDTH / 4 * 3, 10))

            pygame.display.flip()
            self.clock.tick_busy_loop(60)


if __name__ == '__main__':
    # player_a = PlayerHuman(pygame.K_w, pygame.K_s)
    player_a = PlayerDeep()
    # player_a = PlayerTracker()
    # player_b = PlayerHuman(pygame.K_UP, pygame.K_DOWN)
    player_b = PlayerTracker()
    pong = Pong(player_a, player_b)
    pong.game_loop()
