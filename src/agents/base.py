class Agent:
    def __init__(self, x, y, possession, name):
        self.init_x = self.x = x
        self.init_y = self.y = y
        self.init_possession = self.possession = possession
        self.name = name

    def reset(self):
        self.x = self.init_x
        self.y = self.init_y
        self.possession = self.init_possession

    def move(self, action):
        if action == NORTH and self.y > 0:
            self.y -= 1
        elif action == SOUTH and self.y < FIELD_DIM_Y - 1:
            self.y += 1
        elif action == WEST and self.x > 0:
            self.x -= 1
        elif action == EAST and self.x < FIELD_DIM_X - 1:
            self.x += 1

    def set_possession(self, possession):
        self.possession = possession
