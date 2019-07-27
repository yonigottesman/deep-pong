class State:
    def __init__(self, pygame, ball_location, paddle_location):
        self.ball_location = ball_location
        self.pygame = pygame
        self.paddle_location = paddle_location