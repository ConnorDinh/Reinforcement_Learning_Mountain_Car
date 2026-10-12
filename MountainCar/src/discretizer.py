class Discretizer:
    def __init__(self, position_bins, velocity_bins):
        self.position_bins = position_bins
        self.velocity_bins = velocity_bins
        self.position_min = -1.2
        self.position_max = 0.6
        self.velocity_min = -0.07
        self.velocity_max = 0.07

    def discretize(self, state):
        position, velocity = state
        p_ratio = (position - self.position_min) / (self.position_max - self.position_min)
        v_ratio = (velocity - self.velocity_min) / (self.velocity_max - self.velocity_min)
        p_index = max(0, min(int(p_ratio * self.position_bins), self.position_bins -1))
        v_index = max(0, min(int(v_ratio * self.velocity_bins), self.velocity_bins -1))
        return (p_index, v_index)
