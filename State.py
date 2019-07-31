class State:
    def __init__(self, pygame, ball_location, paddle_location, image_data, reward=0):
        self.ball_location = ball_location
        self.pygame = pygame
        self.paddle_location = paddle_location
        self.reward = reward
        self.image_data=image_data
