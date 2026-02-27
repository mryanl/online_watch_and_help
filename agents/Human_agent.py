import copy
import threading
from queue import Queue, Full

from utils.utils_graph import check_progress

class HumanResetRequest(Exception):
    """
    Raised by Human_agent.get_action() when the GUI requests an episode reset.
    """
    pass


class Human_agent:
    agent_type = "Human"

    def __init__(self, agent_id: int, char_index: int, **kwargs):
        """
        Args:
            agent_id:   1-indexed agent id used by the VirtualHome env (e.g. 2 for helper).
            char_index: 0-indexed character index used by the arena (e.g. 1 for helper).
        """
        self.agent_id = agent_id
        self.char_index = char_index

        self._action_queue: Queue[str] = Queue(maxsize=1)

        # signals the GUI that a new observation is ready to display.
        self._obs_ready = threading.Event()

        # Latest observation snapshot.
        self._latest_obs: dict | None = None

        # Set by the arena
        self.saver = None

    def reset(self, gt_graph: dict):
        self.init_gt_graph = gt_graph

        try:
            self._action_queue.get_nowait()
        except Exception:
            pass

        self._obs_ready.clear()
        self._latest_obs = None

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """
        Called once per arena step. Publishes the observation to the GUI,
        then BLOCKS until the GUI submits one action string.

        Returns:
            (action_str, agent_info_dict)
        """
        # save the latest observation
        self._latest_obs = copy.deepcopy(obs)
        self._obs_ready.set()

        # blocks until the GUI delivers an action
        action_str = self._action_queue.get()

        self._obs_ready.clear()

        if action_str == "__RESET__":
            raise HumanResetRequest()

        # for logging
        agents_info = {
            "subgoals": [[]],
            "plan": None,
        }
        return action_str, agents_info


    def submit_action(self, action_str: str) -> bool:
        """
        Submit a human action for the current step.

        Returns True if accepted, False if a submission is already pending.
        """
        # If it is held, this step has a pending action.
        try:
            self._action_queue.put_nowait(action_str)
            return True
        except Full:
            return False

    def get_action_history(self) -> list[str | None]:
        """Return the action history for this agent (human = char_index 0 or 1)."""
        if self.saver is None:
            return []
        return self.saver.episode_saved_info["action"][self.char_index]


    def get_progress(self) -> dict:
        actions = self.get_action_history()
        parseable = [a for a in actions if a is not None]
        done_counter, grab_counter, touched_ids = check_progress(parseable)
        return {
            "done": dict(done_counter),
            "holding": dict(grab_counter),
            "touched_ids": list(touched_ids),
        }