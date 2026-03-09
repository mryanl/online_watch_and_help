import copy
import math
import traceback
import threading
from contextlib import nullcontext
import random
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.transform import Rotation as R
from virtualhome.simulation.environment.unity_environment import (
    UnityEnvironment as BaseUnityEnvironment,
)
from virtualhome.simulation.evolving_graph import utils as utils_env
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from utils import utils_environment as utils
from utils import utils_environment as utils_env2
from utils.utils_graph import get_random_goal, bbox_contains


class UnityEnvironment(BaseUnityEnvironment):
    STATEFUL_FURNITURE = {"fridge", "dishwasher", "stove"}
    def __init__(
        self,
        num_agents=2,
        max_episode_length=200,
        env_task_set=None,
        observation_types=None,
        agent_goals=None,
        use_editor=False,
        base_port=8080,
        convert_goal=False,
        port_id=0,
        executable_args={},
        recording_options={
            "recording": False,
            "output_folder": None,
            "file_name_prefix": None,
            "cameras": "PERSON_FROM_BACK",
            "modality": "normal",
        },
        seed=123,
    ):
        if agent_goals is not None:
            self.agent_goals = agent_goals
        else:
            self.agent_goals = ["full" for _ in range(num_agents)]

        self.convert_goal = convert_goal
        self.task_goal, self.goal_spec = {0: {}, 1: {}}, {0: {}, 1: {}}
        self.initial_task_goal = self.task_goal
        self.env_task_set = env_task_set
        self.agent_object_touched = []
        super(UnityEnvironment, self).__init__(
            num_agents=num_agents,
            max_episode_length=max_episode_length,
            observation_types=observation_types,
            use_editor=use_editor,
            base_port=base_port,
            port_id=port_id,
            executable_args=executable_args,
            recording_options=recording_options,
            seed=seed,
        )
        self.full_graph = None
        self._comm_lock = threading.Lock()
        # self._comm_lock = nullcontext()

    def get_graph(self):
        with self._comm_lock:
            graph = super(UnityEnvironment, self).get_graph()
        objects_seen = self.agent_object_touched
        for node in graph["nodes"]:
            if (
                "TOUCHED" not in [st.upper() for st in node["states"]]
                and node["id"] in objects_seen
            ):
                node["states"].append("TOUCHED")
        return graph

    def reward(self):
        reward = 0.0
        done = True
        # print(self.goal_spec)
        if self.convert_goal:
            satisfied, unsatisfied = utils.check_progress2(
                self.get_graph(), self.goal_spec[0]
            )

        else:
            satisfied, unsatisfied = utils.check_progress(
                self.get_graph(), self.goal_spec[0]
            )
        for key, value in satisfied.items():
            if self.convert_goal:
                resp = self.goal_spec[0][key]
                preds_needed, mandatory, reward_per_pred = (
                    resp["count"],
                    resp["final"],
                    resp["reward"],
                )
            else:
                preds_needed, mandatory, reward_per_pred = self.goal_spec[0][key]
            # How many predicates achieved
            value_pred = min(len(value), preds_needed)
            reward += value_pred * reward_per_pred
            if self.convert_goal:
                if mandatory and unsatisfied[key] > 0:
                    done = False
            else:
                if mandatory and unsatisfied[key] > 0:
                    done = False

        self.prev_reward = reward
        return reward, done, {"satisfied_goals": satisfied}

    def get_goal2(self, task_spec, agent_goal):
        if agent_goal == "full":
            # pred = [x for x, y in task_spec.items() if y['count'] > 0 and x.split('_')[0] in ['on', 'inside']]
            # object_grab = [pr.split('_')[1] for pr in pred]
            # predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
            res_dict = {
                goal_k: copy.deepcopy(goal_c)
                for goal_k, goal_c in task_spec.items()
                if goal_c["count"] > 0
            }
            for goal_k, goal_dict in res_dict.items():
                goal_dict.update({"final": True, "reward": 2})
            # res_dict.update(predicates_grab)
            return res_dict
        elif agent_goal == "grab":
            candidates = [
                x.split("_")[1]
                for x, y in task_spec.items()
                if y > 0 and x.split("_")[0] in ["on", "inside"]
            ]
            object_grab = self.rnd.choice(candidates)
            # print('GOAL', candidates, object_grab)
            return {
                "holds_" + object_grab + "_" + "1": {
                    "count": 1,
                    "final": True,
                    "reward": 10,
                    "grab_obj_ids": object_grab,
                    "container_ids": [1],
                },
                "close_" + object_grab + "_" + "1": {
                    "count": 1,
                    "final": False,
                    "reward": 0.1,
                    "grab_obj_ids": object_grab,
                    "container_ids": [1],
                },
            }
        elif agent_goal == "put":
            pred = self.rnd.choice(
                [
                    (x, y)
                    for x, y in task_spec.items()
                    if y["count"] > 0 and x.split("_")[0] in ["on", "inside"]
                ]
            )
            object_grab = [pred[0].split("_")[1]]
            ctid = pred[1]["container_ids"]
            return {
                pred: {
                    "count": 1,
                    "final": True,
                    "reward": 60,
                    "grab_obj_ids": object_grab,
                    "container_ids": ctid,
                },
                "holds_" + object_grab + "_" + "1": {
                    "count": 1,
                    "final": False,
                    "reward": 2,
                    "grab_obj_ids": object_grab,
                    "container_ids": [1],
                },
                "close_" + object_grab + "_" + "1": {
                    "count": 1,
                    "final": False,
                    "reward": 0.05,
                    "grab_obj_ids": object_grab,
                    "container_ids": [1],
                },
            }
        else:
            raise NotImplementedError

    def get_goal(self, task_spec, agent_goal):
        if agent_goal == "full":
            pred = [
                x
                for x, y in task_spec.items()
                if y > 0 and x.split("_")[0] in ["on", "inside"]
            ]
            # object_grab = [pr.split('_')[1] for pr in pred]
            # predicates_grab = {'holds_{}_1'.format(obj_gr): [1, False, 2] for obj_gr in object_grab}
            res_dict = {
                goal_k: [goal_c, True, 2] for goal_k, goal_c in task_spec.items()
            }
            # res_dict.update(predicates_grab)
            return res_dict
        elif agent_goal == "grab":
            candidates = [
                x.split("_")[1]
                for x, y in task_spec.items()
                if y > 0 and x.split("_")[0] in ["on", "inside"]
            ]
            object_grab = self.rnd.choice(candidates)
            # print('GOAL', candidates, object_grab)
            return {
                "holds_" + object_grab + "_" + "1": [1, True, 10],
                "close_" + object_grab + "_" + "1": [1, False, 0.1],
            }
        elif agent_goal == "put":
            pred = self.rnd.choice(
                [
                    x
                    for x, y in task_spec.items()
                    if y > 0 and x.split("_")[0] in ["on", "inside"]
                ]
            )
            object_grab = pred.split("_")[1]
            return {
                pred: [1, True, 60],
                "holds_" + object_grab + "_" + "1": [1, False, 2],
                "close_" + object_grab + "_" + "1": [1, False, 0.05],
            }
        else:
            raise NotImplementedError

    def rescale_objects_to_place(self, updated_graph):
        new_graph = copy.deepcopy(updated_graph)
        objects_change = [
            "pudding",
            "chips",
            "condimentbottle",
            "condimentshaker",
            "salmon",
            "plate",
        ]
        for node in new_graph["nodes"]:
            if node["class_name"].lower().replace("_", "") in objects_change:
                node["obj_transform"]["scale"] = [
                    x * 0.4 for x in node["obj_transform"]["scale"]
                ]
        return new_graph

    def reset(
        self, environment_graph=None, task_id=None, helper_goal_type=None, seed=None
    ):
        # Make sure that characters are out of graph, and ids are ok
        # ipdb.set_trace()
        if task_id is None:
            task_id = self.rnd.choice(list(range(len(self.env_task_set))))
        env_task = self.env_task_set[task_id]

        self.agent_object_touched = []

        self.task_id = env_task["task_id"]
        self.init_graph = copy.deepcopy(env_task["init_graph"])
        self.init_rooms = env_task["init_rooms"]
        self.task_goal = env_task["task_goal"]
        self.initial_task_goal = copy.deepcopy(self.task_goal)
        if helper_goal_type == "gt":
            self.task_goal[1] = self.task_goal[0]
        elif helper_goal_type == "random":
            from utils.utils_graph import EG

            while True:
                try:
                    goal = get_random_goal(env_task["env_id"], seed)

                    # * check if object exists
                    goal_spec = utils_env2.convert_goal(goal, self.init_graph)

                    # * check if no multiple locations for the same object
                    eg = EG(env_task["init_graph"])
                    for subgoal_name, subgoal in goal_spec.items():
                        for obj_id in subgoal["grab_obj_ids"]:
                            obj = eg[obj_id]
                            room, ctnr, srfc = obj.get_location()

                    break
                except Exception:
                    traceback.print_exc()
                    seed += 100
            self.task_goal[1] = goal
        elif helper_goal_type == "unknown":
            self.task_goal[1] = dict()
        else:
            raise ValueError(f"{helper_goal_type = }")

        if self.convert_goal:
            self.task_goal = {
                agent_id: utils_env2.convert_goal(task_goal, self.init_graph)
                for agent_id, task_goal in self.task_goal.items()
            }
        # ipdb.set_trace()
        # TODO: remove
        self.task_name = env_task["task_name"]

        old_env_id = self.env_id
        self.env_id = env_task["env_id"]
        # print(
        #     "Resetting... Envid: {}. Taskid: {}. Index: {}".format(
        #         self.env_id, self.task_id, task_id
        #     )
        # )

        # TODO: in the future we may want different goals
        if self.convert_goal:
            self.goal_spec = {
                agent_id: self.get_goal2(
                    self.task_goal[agent_id], self.agent_goals[agent_id]
                )
                for agent_id in range(self.num_agents)
            }

        else:
            self.goal_spec = {
                agent_id: self.get_goal(
                    self.task_goal[agent_id], self.agent_goals[agent_id]
                )
                for agent_id in range(self.num_agents)
            }

        if False:  # old_env_id == self.env_id:
            print("Fast reset")
            self.comm.fast_reset()
        else:
            self.comm.reset(self.env_id)

        s, g = self.comm.environment_graph()
        edge_ids = set(
            [edge["to_id"] for edge in g["edges"]]
            + [edge["from_id"] for edge in g["edges"]]
        )
        node_ids = set([node["id"] for node in g["nodes"]])
        if len(edge_ids - node_ids) > 0:
            raise AssertionError

        if self.env_id not in self.max_ids.keys():
            max_id = max([node["id"] for node in g["nodes"]])
            self.max_ids[self.env_id] = max_id

        max_id = self.max_ids[self.env_id]


        # ipdb.set_trace()
        if environment_graph is not None:
            updated_graph = environment_graph
            s, g = self.comm.environment_graph()
            updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
            if self.env_id == 6:
                # TODO: this is because trashcan causes problems when removed, need to check more later
                cids = [node["id"] for node in updated_graph["nodes"]]
                nodes_trash = [node for node in g["nodes"] if node["id"] == 360]
                edges_trash = [
                    edge
                    for edge in g["edges"]
                    if (edge["from_id"] == 360 and edge["to_id"] in cids)
                    or (edge["to_id"] == 360 and edge["from_id"] in cids)
                ]
                updated_graph["nodes"] += nodes_trash
                updated_graph["edges"] += edges_trash
            updated_graph = self.rescale_objects_to_place(updated_graph)
            success, m = self.comm.expand_scene(updated_graph)
        else:
            updated_graph = self.init_graph
            s, g = self.comm.environment_graph()
            updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
            if self.env_id == 6:
                # TODO: this is because tashcaan causes problems, check more later
                cids = [node["id"] for node in updated_graph["nodes"]]
                nodes_trash = [node for node in g["nodes"] if node["id"] == 360]
                edges_trash = [
                    edge
                    for edge in g["edges"]
                    if (edge["from_id"] == 360 and edge["to_id"] in cids)
                    or (edge["to_id"] == 360 and edge["from_id"] in cids)
                ]
                updated_graph["nodes"] += nodes_trash
                updated_graph["edges"] += edges_trash
            try:
                updated_graph = self.rescale_objects_to_place(updated_graph)
                success, m = self.comm.expand_scene(updated_graph)
            except:
                print("Failure starting graph")
                content_dict = {"env_id": self.env_id, "graph": updated_graph}
                import pickle as pkl

                with open("debug_wah.pkl", "wb") as f:
                    pkl.dump(content_dict, f)
                raise AssertionError

        if not success:
            print("Error expanding scene")
            raise AssertionError

        self.num_static_cameras = self.offset_cameras = self.comm.camera_count()[1]
        if self.init_rooms[0] not in ["kitchen", "bedroom", "livingroom", "bathroom"]:
            rooms = self.rnd.sample(["kitchen", "bedroom", "livingroom", "bathroom"], 2)
        else:
            rooms = list(self.init_rooms)

        for i in range(self.num_agents):
            if i in self.agent_info:
                # rooms[i] = 'kitchen'
                self.comm.add_character(self.agent_info[i], initial_room=rooms[i])
            else:
                self.comm.add_character()

        self.changed_graph = True
        graph = self.get_graph()
        self.init_unity_graph = graph
        self.rooms = [
            (node["class_name"], node["id"])
            for node in graph["nodes"]
            if node["category"] == "Rooms"
        ]
        self.id2node = {node["id"]: node for node in graph["nodes"]}

        obs = self.get_observations()
        self.steps = 0
        self.prev_reward = 0.0
        return obs

    def step(self, action_dict):
        script_list, script_dict = utils.convert_action(action_dict)
        failed_execution = False
        if len(script_list[0]) > 0:
            # print(script_list)
            if self.recording_options["recording"]:
                with self._comm_lock:
                    success, message = self.comm.render_script(
                        script_list,
                        recording=True,
                        skip_animation=False,
                        camera_mode=self.recording_options["cameras"],
                        file_name_prefix="task_{}".format(self.task_id),
                        image_synthesis=self.recording_optios["modality"],
                    )
            else:
                if "touch" in script_list[0]:
                    objid = int(action_dict[0].split("(")[1].strip()[:-1])
                    self.agent_object_touched.append(objid)
                    success, message = True, {}
                else:
                    with self._comm_lock:
                        # print(colored(script_list, "yellow"))
                        success, message = self.comm.render_script(
                            script_list,
                            recording=False,
                            image_synthesis=[],
                            skip_animation=True,
                        )
            if not success:
                # ipdb.set_trace()
                # print("NO SUCCESS")
                # print(message, script_list)
                failed_execution = True
            else:
                self.changed_graph = True
        else:
            message = {"0": {"message": "Empty action"}}

        # Obtain reward
        reward, done, info = self.reward()

        graph = self.get_graph()
        self.steps += 1

        obs = self.get_observations()

        info["finished"] = done
        info["graph"] = graph
        info["executed_script"] = script_dict
        info["failed_exec"] = failed_execution
        info["message"] = {
            int(k): v["message"]
            for k, v in message.items()
            if v["message"] != "Success"
        }
        if self.steps == self.max_episode_length:
            done = True
        return obs, reward, done, info

    def get_angle(self, rot):
        rot = R.from_quat(rot)
        euler = rot.as_euler("xzy")
        # dchange = np.sin(euler[1])*np.cos(euler[0]), np.cos(euler[1])*np.sin(euler[0])
        # dchange = np.sin(euler[1]+euler[0]), np.cos(euler[1]+euler[0])
        x = np.cos(euler[2]) * np.cos(euler[1])
        y = np.sin(euler[2]) * np.cos(euler[1])
        z = np.sin(euler[1])
        dchange = y, x
        return np.arctan2(x, y) * 180 / math.pi

    def get_observation(self, agent_id, obs_type, info={}):
        if obs_type == "partial":
            # agent 0 has id (0 + 1)
            curr_graph = self.get_graph()
            curr_graph = utils.clean_house_obj(curr_graph)
            curr_graph = utils.inside_not_trans(curr_graph)
            self.full_graph = copy.deepcopy(curr_graph)
            obs = utils_env.get_visible_nodes(curr_graph, agent_id=(agent_id + 1))
            return obs

        elif obs_type == "full":
            curr_graph = self.get_graph()
            curr_graph = utils.clean_house_obj(curr_graph)
            curr_graph = utils.inside_not_trans(curr_graph)
            self.full_graph = copy.deepcopy(curr_graph)
            return curr_graph

        elif obs_type == "cone":
            curr_graph = self.get_graph()
            curr_graph = utils.clean_house_obj(curr_graph)
            curr_graph = utils.inside_not_trans(curr_graph)
            self.full_graph = copy.deepcopy(curr_graph)
            obs = utils_env.get_visible_nodes(curr_graph, agent_id=(agent_id + 1))

            # TODO: implement a real coen here, with unity
            # s, obs_cone = self.comm.get_visible_objects(camera_index)
            agent_node = [node for node in obs["nodes"] if node["id"] == agent_id + 1][
                0
            ]
            position, rotation = (
                agent_node["obj_transform"]["position"],
                agent_node["obj_transform"]["rotation"],
            )
            rotation_char = self.get_angle(rotation)
            rotation_all = [
                (
                    node["id"],
                    180.0
                    / math.pi
                    * np.arctan2(
                        node["obj_transform"]["position"][2] - position[2],
                        node["obj_transform"]["position"][0] - position[0],
                    ),
                )
                for node in obs["nodes"]
            ]
            rot = [
                rot_id for rot_id in rotation_all if abs(rot_id[1] - rotation_char) < 20
            ]
            rotation_ids = [r[0] for r in rot]
            room_doors = [
                node["id"]
                for node in obs["nodes"]
                if node["category"] in ["Rooms", "Doors"]
            ]
            rotation_ids = set(rotation_ids + room_doors + [agent_id + 1])
            all_ids = [node["id"] for node in obs["nodes"]]
            missing_ids = list(set(all_ids) - set(rotation_ids))
            # print("Removed:")
            # print([node['class_name'] for node in obs['nodes'] if node['id'] in missing_ids])
            new_obs = {
                "nodes": [node for node in obs["nodes"] if node["id"] in rotation_ids],
                "edges": [
                    edge
                    for edge in obs["edges"]
                    if edge["from_id"] in rotation_ids and edge["to_id"] in rotation_ids
                ],
            }
            return new_obs

        elif obs_type == "visible":
            # Only objects in the field of view of the agent
            raise NotImplementedError

        elif obs_type == "image":
            with self._comm_lock:
                return super().get_observation(agent_id, obs_type, info)
            camera_ids = [
                self.offset_cameras
                + agent_id * self.num_camera_per_agent
                + self.CAMERA_NUM
            ]
            if "image_width" in info:
                image_width = info["image_width"]
                image_height = info["image_height"]
            else:
                image_width, image_height = (
                    self.default_image_width,
                    self.default_image_height,
                )
            if "obs_type" in info:
                curr_obs_type = info["obs_type"]
            else:
                curr_obs_type = self.default_obs_type
            with self._comm_lock:
                s, images = self.comm.camera_image(
                    camera_ids,
                    mode=curr_obs_type,
                    image_width=image_width,
                    image_height=image_height,
                )
            if not s:
                raise AssertionError
            return images[0]
        else:
            raise NotImplementedError

        return updated_graph

    def reconnect(self, latest_graph, grabbed_classes=None):
        """
        Called after a UnityCommunicationException to restart the Unity backend
        and restore it to the state captured in latest_graph.
        All task/goal/episode state in Python memory is preserved as-is.
        """
        self.port_number += 1
        self.comm = UnityCommunication(port=str(self.port_number), **self.executable_args)
        self.comm.reset(self.env_id)

        char_ids = {
            n["id"] for n in latest_graph["nodes"]
            if n["class_name"] == "character"
        }

        clean_graph = {
            "nodes": [n for n in latest_graph["nodes"] if n["id"] not in char_ids],
            "edges": [
                e for e in latest_graph["edges"]
                if e["from_id"] not in char_ids and e["to_id"] not in char_ids and e["relation_type"] not in ("HOLDS_RH", "HOLDS_LH")
            ],
        }


        clean_graph, on_intents = self.fix_grabbed_object_positions(clean_graph, grabbed_classes)

        max_id = self.max_ids[self.env_id]
        updated_graph = utils.separate_new_ids_graph(clean_graph, max_id)
        to_reopen = self.force_furniture_closed(updated_graph)

        success, m = self.comm.expand_scene(updated_graph)
        if not success:
            raise AssertionError(f"reconnect: expand_scene failed: {m}")

        # add characters with cameras backs
        self.num_static_cameras = self.offset_cameras = self.comm.camera_count()[1]

        for i in range(self.num_agents):
            char_id = i + 1
            # find which room the character was in
            room_ids = {
                e["to_id"] for e in latest_graph["edges"]
                if e["from_id"] == char_id and e["relation_type"] == "INSIDE"
            }
            room_name = next(
                (n["class_name"] for n in latest_graph["nodes"] if n["id"] in room_ids),
                self.init_rooms[i]
            )
            if i in self.agent_info:
                self.comm.add_character(self.agent_info[i], initial_room=room_name)
            else:
                self.comm.add_character(initial_room=room_name)

        # move each character to their saved position
        for i in range(self.num_agents):
            char_id = i + 1
            char_node = next(
                (n for n in latest_graph["nodes"] if n["id"] == char_id), None
            )
            if char_node is not None:
                pos = char_node["obj_transform"]["position"]
                self.comm.move_character(i, pos)

        self.reopen_furniture(to_reopen)
        _, new_graph = self.comm.environment_graph()
        self.restore_held_objects(latest_graph, new_graph)
        self.restore_on_objects(on_intents, new_graph, char_idx=0)

        self.changed_graph = True
        self.update_goal(graph=new_graph)
        pass

    def fix_grabbed_object_positions(self, clean_graph, grabbed_classes):
        """
        For objects whose class name is in grabbed_classes (currently grabbed by an
        agent since the last reconnect):
          - INSIDE: randomize position within the container's bounding box so
                    Unity accepts the placement via expand_scene.
          - ON: collect a placement intent to be replayed after expand_scene via
                restore_on_objects (grab + put), since position-only fixes are
                unreliable for surface placements. Only goal-class objects are restored.
        Returns (clean_graph, on_intents).
        """
        if not grabbed_classes:
            return clean_graph, defaultdict(list)

        id2node = {n["id"]: n for n in clean_graph["nodes"]}

        # Find the placement edge (INSIDE or ON) for each grabbed object
        placement_edges = {}
        for edge in clean_graph["edges"]:
            from_node = id2node.get(edge["from_id"], {})
            if (edge["relation_type"] in ("INSIDE", "ON")
                and from_node.get("class_name") in grabbed_classes
                and id2node.get(edge["to_id"], {}).get("category") not in
                    {"Rooms", "Walls", "Floor", "Ceiling", "Doors", "Windows"}):
                placement_edges[edge["from_id"]] = (
                    edge["relation_type"], edge["to_id"]
                )

        on_intents = defaultdict(list)
        for obj_id, (rel, container_id) in placement_edges.items():
            obj_node = id2node.get(obj_id)
            container_node = id2node.get(container_id)
            if obj_node is None or container_node is None:
                continue

            obj_class = obj_node["class_name"]
            container_class = container_node["class_name"]

            if rel == "INSIDE":

                obj_pos = obj_node["obj_transform"]["position"]

                if not bbox_contains(container_node, obj_pos):
                    bb = container_node.get("bounding_box")
                    if bb:
                        cx, cy, cz = bb["center"]
                        sx, sy, sz = bb["size"]
                        new_pos = [
                            random.uniform(cx - sx / 2, cx + sx / 2),
                            random.uniform(cy - sy / 2, cy + sy / 2),
                            random.uniform(cz - sz / 2, cz + sz / 2),
                        ]
                    else:
                        new_pos = list(container_node["obj_transform"]["position"])
                    print(
                        f"fix_grabbed_object_positions: moved {obj_class} ({obj_id}) "
                        f"INSIDE {container_class} ({container_id}) "
                        f"from {[round(v, 3) for v in obj_pos]} "
                        f"to {[round(v, 3) for v in new_pos]}"
                    )
                    obj_node["obj_transform"]["position"] = new_pos
                else:
                    print(
                        f"fix_grabbed_object_positions: {obj_class} ({obj_id}) "
                        f"INSIDE {container_class} ({container_id}) already within bbox, no change"
                    )
            elif rel == "ON":
                container_pos = tuple(container_node["obj_transform"]["position"])
                obj_pos = obj_node["obj_transform"]["position"]
                on_intents[(container_class, container_pos)].append((obj_class, obj_pos))
                print(
                    f"fix_grabbed_object_positions: will restore {obj_class} "
                    f"ON {container_class} @ {[round(v, 3) for v in container_pos]}"
                )

        return clean_graph, on_intents

    def restore_held_objects(self, latest_graph, new_graph):
        """
        After reconnection, restore held objects for each agent by grabbing the
        closest object of the same class to the agent's position in the new graph.
        Uses class name + proximity instead of object ID since IDs may have changed.
        """
        id2node = {n["id"]: n for n in latest_graph["nodes"]}

        for i in range(self.num_agents):
            char_id = i + 1

            char_node = next(
                (n for n in latest_graph["nodes"] if n["id"] == char_id), None
            )
            if char_node is None:
                continue
            char_pos = char_node["obj_transform"]["position"]

            held_ids = [
                e["to_id"] for e in latest_graph["edges"]
                if e["from_id"] == char_id
                and e["relation_type"] in ("HOLDS_RH", "HOLDS_LH")
            ]

            held_classes = [
                id2node[oid]["class_name"]
                for oid in held_ids
                if oid in id2node
            ]

            already_grabbed = set()

            for obj_class in held_classes:
                candidates = [
                    n for n in new_graph["nodes"]
                    if n["class_name"] == obj_class
                    and n["id"] not in already_grabbed
                ]
                if not candidates:
                    continue

                def dist(n):
                    p = n["obj_transform"]["position"]
                    return sum((a - b) ** 2 for a, b in zip(char_pos, p))

                closest = min(candidates, key=dist)
                already_grabbed.add(closest["id"])

                script = [f"<char{i}> [grab] <{obj_class}> ({closest['id']})"]
                result = self.comm.render_script(
                    script,
                    recording=False,
                    image_synthesis=[],
                    skip_animation=True,
                )
                print(f"grab restore char{i} {obj_class} ({closest['id']}): {result}")

    def restore_on_objects(self, on_intents, new_graph, char_idx=0):
        def pos_dist_sq(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b))
        holds_relations = {"HOLDS_RH", "HOLDS_LH"}
        id2node = {n["id"]: n for n in new_graph["nodes"]}
        on_edges = defaultdict(Counter)
        obj_ids_with_on = set()
        for edge in new_graph["edges"]:
            if edge["relation_type"] == "ON":
                obj_node = id2node.get(edge["from_id"])
                if obj_node:
                    on_edges[edge["to_id"]][obj_node["class_name"]] += 1
                    obj_ids_with_on.add(edge["from_id"])
            if edge["relation_type"] in holds_relations:
                obj_ids_with_on.add(edge["to_id"])

        for (container_class, saved_container_pos), items in on_intents.items():
            container_candidates = [
                n for n in new_graph["nodes"]
                if n["class_name"] == container_class
                and pos_dist_sq(n["obj_transform"]["position"], saved_container_pos) <= 1.0
            ]
            if not container_candidates:
                print(f"restore_on_objects: no {container_class} within 1m of {saved_container_pos}, skipping")
                continue
            container_node = min(container_candidates, key=lambda n: pos_dist_sq(n["obj_transform"]["position"], saved_container_pos))
            container_id = container_node["id"]
            container_pos = container_node["obj_transform"]["position"]

            needed = Counter(obj_class for obj_class, _ in items)
            already = on_edges[container_id]
            for obj_class, count_needed in needed.items():
                deficit = count_needed - already[obj_class]
                if deficit <= 0:
                    print(f"restore_on_objects: {obj_class} x{count_needed} already ON {container_class} ({container_id}), skipping")
                    continue

                # 3. Find the `deficit` closest unplaced objects within 1 metre.
                obj_candidates = [
                    n for n in new_graph["nodes"]
                    if n["class_name"] == obj_class
                    and n["id"] not in obj_ids_with_on
                    and pos_dist_sq(n["obj_transform"]["position"], container_pos) <= 1.0
                ]
                obj_candidates.sort(key=lambda n: pos_dist_sq(n["obj_transform"]["position"], container_pos))

                if not obj_candidates:
                    print(f"restore_on_objects: no {obj_class} within 1m of {container_class} ({container_id}), skipping")
                    continue

                for chosen in obj_candidates[:deficit]:

                    char_tag = f"<char{char_idx}>"
                    walk_script  = f"{char_tag} [walk] <{container_class}> ({container_id})"
                    grab_script  = f"{char_tag} [grab] <{obj_class}> ({chosen['id']})"
                    place_script = f"{char_tag} [put] <{obj_class}> ({chosen['id']}) <{container_class}> ({container_id})"
                    for script_line in [walk_script, grab_script, place_script]:
                        ok, msg = self.comm.render_script(
                            [script_line],
                            recording=False,
                            image_synthesis=[],
                            skip_animation=True,
                        )
                        print(f"restore_on_objects: {script_line!r} -> ok={ok}, msg={msg}")
                        if not ok:
                            print(f"restore_on_objects: step failed, aborting this object")
                            break

                    obj_ids_with_on.add(chosen['id'])

    def update_goal(self, graph=None):
        if graph is None:
            graph = self.get_graph()
        if self.convert_goal:
            self.task_goal = {
                agent_id: utils_env2.convert_goal(task_goal, graph)
                for agent_id, task_goal in self.initial_task_goal.items()
            }
            self.goal_spec = {
                agent_id: self.get_goal2(
                    self.task_goal[agent_id], self.agent_goals[agent_id]
                )
                for agent_id in range(self.num_agents)
            }

        else:
            self.goal_spec = {
                agent_id: self.get_goal(
                    self.task_goal[agent_id], self.agent_goals[agent_id]
                )
                for agent_id in range(self.num_agents)
            }

    def force_furniture_closed(self, updated_graph):
        """
        Mutates updated_graph in-place: forces all STATEFUL_FURNITURE nodes that
        are OPEN to CLOSED before passing the graph to expand_scene.
        Returns a list of (id, class_name) for nodes that need to be re-opened
        after expand_scene via reopen_furniture().
        """
        to_reopen = []
        for node in updated_graph["nodes"]:
            if node["class_name"].lower() in self.STATEFUL_FURNITURE:
                if "OPEN" in [s.upper() for s in node["states"]]:
                    to_reopen.append((node["id"], node["class_name"]))
                    node["states"] = [s for s in node["states"] if s.upper() != "OPEN"]
                    node["states"].append("CLOSED")
        return to_reopen

    def reopen_furniture(self, to_reopen):
        """
        After expand_scene and characters have been added, re-open furniture that
        was open before reconnect by issuing [open] scripts via char0.
        to_reopen is the list returned by force_furniture_closed().
        """
        for node_id, class_name in to_reopen:
            script = f"<char0> [open] <{class_name}> ({node_id})"
            s, msg = self.comm.render_script([script], recording=False, skip_animation=True)
            if not s:
                print(f"reconnect: failed to re-open {class_name} ({node_id}): {msg}")
            else:
                print(f"reconnect: re-opened {class_name} ({node_id})")