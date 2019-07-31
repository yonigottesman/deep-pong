from Direction import Direction


class PlayerTracker:
    def __init__(self):
        pass

    def get_move(self, state):
        if state.paddle_location[1] < state.ball_location[1]:
            return Direction.DOWN
        if state.paddle_location[1] > state.ball_location[1]:
            return Direction.UP
        else:
            return Direction.STAY