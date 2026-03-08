import atexit
import pickle
from collections import Counter
from utils.utils_graph import check_progress
from typing import TYPE_CHECKING
from utils import utils_environment as utils_env2

if TYPE_CHECKING:
    from envs.unity_environment import UnityEnvironment
class Arena(object):
    def __init__(self, env, agents, saver, resume=False):
        self.env: UnityEnvironment = env
        self.agents = agents
        self.saver = saver
        self.resume = resume
        atexit.register(self.env.close)

    def reset_saver(self):
        self.saver.episode_saved_info = {
            "task_id": self.env.task_id,
            "env_id": self.env.env_id,
            "task_name": self.env.task_name,
            "gt_goals": self.env.task_goal[0],
            "goals": self.env.task_goal,
            "init_rooms": {i: self.env.init_rooms[i] for i in range(len(self.agents))},
            "init_unity_graph": self.env.init_graph,
            "goals_finished": [],
            "obs": {i: [] for i in range(len(self.agents))},
            "graph": [self.env.init_unity_graph],
            "action": {i: [] for i in range(len(self.agents))},
            "plan": {i: [] for i in range(len(self.agents))},
            "subgoals": {i: [] for i in range(len(self.agents))},
            "belief": {i: [] for i in range(len(self.agents))},
            "belief_room": {i: [] for i in range(len(self.agents))},
            "belief_graph": {i: [] for i in range(len(self.agents))},
            "hands": [],
            "executed": [],
            "cost": Counter(),
            "io": [],
            "prompt_info": [],
            "human_done": [],
        }

    def reset(self, episode_id, helper_goal_type, seed):
        ob = None
        while ob is None:
            ob = self.env.reset(
                task_id=episode_id,
                helper_goal_type=helper_goal_type,
                seed=seed,
            )
        self.reset_saver()

        for it, agent in enumerate(self.agents):
            agent.seed = seed + it
            agent.saver = self.saver
            match agent.agent_type:
                case "MCTS" | "GnP" | "Human":
                    agent.reset(self.env.full_graph)
                case _:
                    raise ValueError(f"Invalid agent type: {agent.agent_type}")

        return ob

    def get_actions(self, obs):
        actions = dict()
        agents_info = dict()

        for it, agent in enumerate(self.agents):
            match agent.agent_type:
                case "MCTS":
                    actions[it], agents_info[it] = agent.get_action(
                        obs=obs[it],
                        # * `self.env.goal_spec` has additional 'final' 'reward' keys
                        # * than `self.env.task_goal`
                        goal_spec=self.env.goal_spec[it],
                    )

                case "GnP" | "Human":
                    # * agent will collect info from `saver`
                    actions[it], agents_info[it] = agent.get_action(obs=obs[it])
                case _:
                    raise ValueError(f"Invalid agent type: {agent.agent_type}")
            agents_info[it]['agent_type'] = agent.agent_type

        return actions, agents_info

    def step(self):

        steps = self.env.steps
        graph = self.env.get_graph()
        obs = self.env.get_observations()

        actions, agents_info = self.get_actions(obs)
        self.saver.record_pre_step(steps, actions, agents_info, graph)

        try:
            obs, reward, done, env_info = self.env.step(actions)
        except Exception as e:
            if type(e).__name__ not in ("UnityCommunicationException", "UnityEngineException") or not self.resume:
                raise
            self.saver.warning("Unity disconnected — reconnecting with latest graph")
            latest_graph = self.saver.episode_saved_info["graph"][-1]

            grabbed_ids = set()
            for agent_actions in self.saver.episode_saved_info["action"].values():
                parseable = [a for a in agent_actions if a is not None]
                _, _, touched_ids = check_progress(parseable)
                grabbed_ids.update(touched_ids)


            self.env.reconnect(latest_graph, grabbed_ids)
            # Retry the same step once after reconnect
            obs, reward, done, env_info = self.env.step(actions)


        self.saver.record_post_step(steps, env_info, actions)

        return done, env_info["finished"]

    def save_camera_img(self, step):
        for agent_id in range(len(self.agents)):
            for view in self.saver.camera_views:
                obs = self.env.get_observation(
                    agent_id=agent_id,
                    obs_type="image",
                    info=dict(
                        view=view,
                        image_width=self.saver.img_w,
                        image_height=self.saver.img_h,
                    ),
                )
                self.saver.save_camera_img(obs, agent_id, view, step)

    def run(self):
        self.save_camera_img(self.env.steps)

        pbar = self.saver.pbar
        pbar_step = pbar.add_task("step", total=self.env.max_episode_length)

        while True:
            done, success = self.step()
            pbar.update(pbar_step, advance=1)
            self.save_camera_img(self.env.steps)

            if done:
                break

        self.saver.episode_saved_info["success"] = success
        self.saver.episode_saved_info["steps"] = self.env.steps

        pbar.remove_task(pbar_step)

        return success
