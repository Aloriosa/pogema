from gymnasium.wrappers import TimeLimit


class MultiTimeLimit(TimeLimit):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        self.print_elapsed_steps = self._elapsed_steps
        self.print_max_episode_steps = self.grid_config.max_episode_steps
        if self._elapsed_steps >= self.grid_config.max_episode_steps:#_max_episode_steps:
            truncated = [True] * self.get_num_agents()
        return observation, reward, terminated, truncated, info

    def set_elapsed_steps(self, elapsed_steps):
        if not self.grid_config.persistent:
            raise ValueError("Cannot set elapsed steps for non-persistent environment!")
        assert elapsed_steps >= 0
        self._elapsed_steps = elapsed_steps
