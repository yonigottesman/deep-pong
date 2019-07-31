from Direction import Direction


class PlayerHuman:
    def __init__(self, up_key, down_key):
        self.up_key = up_key
        self.down_key = down_key

    def get_move(self, state):
        keys_pressed = state.pygame.key.get_pressed()
        if keys_pressed[self.up_key]:
            return Direction.UP
        if keys_pressed[self.down_key]:
            return Direction.DOWN
