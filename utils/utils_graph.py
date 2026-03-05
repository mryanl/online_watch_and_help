import random
import re
from collections import Counter
from dataclasses import dataclass

from virtualhome.simulation.evolving_graph.environment import (
    Bounds,
    EnvironmentGraph,
    GraphEdge,
    GraphNode,
    Property,
    Relation,
    State,
)

from utils import utils_environment as utils_env
from utils.utils_logging import format_row, get_my_logger
from utils.utils_environment import check_progress2

TASK_TO_OBJECT_NAMES = dict(
    prepare_food=["apple", "cupcake", "pudding", "salmon"],
    put_dishwasher=["cutleryfork", "plate", "waterglass", "wineglass"],
    put_fridge=["apple", "cupcake", "pudding", "salmon"],
    setup_table=["cutleryfork", "plate", "waterglass", "wineglass"],
    watch_tv=["chips", "condimentbottle", "remotecontrol"],
)
TASK_TO_TARGET_NAMES = dict(
    prepare_food=["kitchentable", "stove"],
    put_dishwasher=["dishwasher"],
    put_fridge=["fridge"],
    setup_table=["coffeetable", "kitchentable"],
    watch_tv=["coffeetable"],
)

TASK_NAMES = sorted(list(TASK_TO_OBJECT_NAMES.keys()))
OBJECT_NAMES = sorted(
    list(set([obj for objs in TASK_TO_OBJECT_NAMES.values() for obj in objs]))
)

TARGET_NAME_TO_PREP = dict(
    coffeetable="on",
    dishwasher="inside",
    fridge="inside",
    kitchentable="on",
    stove="inside",
)
TARGET_NAMES = sorted(list(TARGET_NAME_TO_PREP.keys()))

ENV_ID_TO_TARGET_NAME_TO_ID = {
    0: dict(
        coffeetable=372,
        dishwasher=None,  # ! wrong target
        fridge=306,
        kitchentable=231,
        stove=312,  # ! wrong target
    ),
    1: dict(
        coffeetable=[221, 290][0],  # ! wrong target
        dishwasher=152,
        fridge=149,
        kitchentable=123,
        stove=150,
    ),
    2: dict(
        coffeetable=215,
        dishwasher=166,
        fridge=163,
        kitchentable=136,
        stove=164,
    ),
    3: dict(
        coffeetable=268,
        dishwasher=104,
        fridge=103,
        kitchentable=72,
        stove=105,
    ),
    4: dict(
        coffeetable=None,  # ! wrong target
        dishwasher=154,
        fridge=155,
        kitchentable=136,
        stove=152,
    ),
    5: dict(
        coffeetable=111,
        dishwasher=None,  # ! wrong target
        fridge=247,
        kitchentable=194,
        stove=242,
    ),
    6: dict(
        coffeetable=[247, 327][0],
        dishwasher=159,
        fridge=[165, 166][1],
        kitchentable=130,
        stove=157,
    ),
}


def item(it):
    assert len(it) == 1
    if isinstance(it, list) or isinstance(it, tuple):
        return it[0]
    if (
        isinstance(it, set)
        or isinstance(it, type({}.keys()))
        or isinstance(it, type({}.values()))
        or isinstance(it, type({}.items()))
    ):
        return next(iter(it))
    if isinstance(it, dict):
        return list(it.items())[0]
    elif isinstance(it, Counter):
        return it.most_common(1)[0]
    else:
        raise ValueError(f"unsupported type: {type(it)}")


def fix_graph(env_task_set):
    for env in env_task_set:
        init_gr = env["init_graph"]
        gbg_can = [
            node["id"]
            for node in init_gr["nodes"]
            if node["class_name"] in ["garbagecan", "clothespile"]
        ]
        init_gr["nodes"] = [
            node for node in init_gr["nodes"] if node["id"] not in gbg_can
        ]
        init_gr["edges"] = [
            edge
            for edge in init_gr["edges"]
            if edge["from_id"] not in gbg_can and edge["to_id"] not in gbg_can
        ]
        for node in init_gr["nodes"]:
            if node["class_name"] == "cutleryfork":
                node["obj_transform"]["position"][1] += 0.1
    return env_task_set


def get_random_goal(env_id, seed):
    goal = dict()

    rng = random.Random(seed)

    task_name = rng.choice(TASK_NAMES)

    target_names = TASK_TO_TARGET_NAMES[task_name]
    target_name = rng.choice(target_names)
    target_id = ENV_ID_TO_TARGET_NAME_TO_ID[env_id][target_name]
    prep = TARGET_NAME_TO_PREP[target_name]

    object_names = TASK_TO_OBJECT_NAMES[task_name]
    subgoal_count = rng.randint(1, 3)
    for _ in range(subgoal_count):
        object_name = rng.choice(object_names)
        count = rng.randint(1, 6)
        goal[f"{prep}_{object_name}_{target_id}"] = count

    return goal


def parse_action(s):
    if s is None:
        return [None]

    # Define a regular expression pattern to match the parts of the string
    pattern = r"\[(.*?)\] <(.*?)> \((\d+)\) <(.*?)> \((\d+)\)"
    match = re.match(pattern, s)

    if match:
        # Extract the parts and convert the number to an integer
        action = match.group(1)
        device = match.group(2)
        number = match.group(3)
        device1 = match.group(4)
        number1 = match.group(5)
        return [action, device, number, device1, number1]

    else:
        # If the pattern does not match, return an empty list or raise an error
        pattern1 = r"\[(.*?)\] <(.*?)> \((\d+)\)"
        match = re.match(pattern1, s)
        if match:
            action = match.group(1)
            device = match.group(2)
            number = match.group(3)
            return [action, device, number]


def dedup_list(l):
    if len(l) == 0:
        return l
    result = [l[0]]
    for action in l[1:]:
        if action != result[-1]:
            result.append(action)
    return result


def subgoal_string_to_tuple(s):
    if s is None:
        return None
    res = []
    for i, part in enumerate(s.split("_")):
        if i != 0:
            part = int(part)
        res.append(part)
    return tuple(res)


def check_progress(actions):
    """
    Input:  history actions
    Output: done counter (put), ongoing counter (grab)
    """
    done_counter = Counter()
    grab_counter = Counter()
    touched_obj_ids = set()
    for action in actions:
        if action is None:
            continue
        parsed = parse_action(action)
        match parsed[0]:
            case "grab":
                obj = parsed[1]
                grab_counter[obj] += 1

                touched_obj_ids.add(int(parsed[2]))

            case "putin" | "putback":
                obj = parsed[1]
                done_counter[obj] += 1

                grab_counter[obj] -= 1
                if grab_counter[obj] == 0:
                    grab_counter.pop(obj)
    return done_counter, grab_counter, touched_obj_ids

def bbox_contains(node, pos):
    """Return True if world position pos is inside node's bounding box."""
    bb = node.get("bounding_box")
    if not bb:
        return False
    cx, cy, cz = bb["center"]
    sx, sy, sz = bb["size"]
    return (
        abs(pos[0] - cx) <= sx / 2 and
        abs(pos[1] - cy) <= sy / 2 and
        abs(pos[2] - cz) <= sz / 2
    )

class GN(GraphNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indeg = []
        self.outdeg = []

    @staticmethod
    def from_dict(d):
        kwargs = {"category": None, "prefab_name": None, "bounding_box": None}
        for k in kwargs.keys():
            if k in d:
                if k == "bounding_box":
                    kwargs[k] = Bounds(**d[k]) if d[k] is not None else d[k]
                else:
                    kwargs[k] = d[k]

        return GN(
            d["id"],
            d["class_name"],
            {
                s if isinstance(s, Property) else Property[s.upper()]
                for s in d["properties"]
            },
            {State[s.upper()] for s in d["states"]},
            **kwargs,
        )

    def __str__(self):
        basic = f"{self.id:3d}|{self.class_name}"
        # if Property.CONTAINERS in self.properties:
        if len(self.states) > 0:
            states = [s.name for s in self.states]
            basic += f"|{'+'.join(states)}"
        return basic

    def __repr__(self):
        return self.__str__()

    def filter_deg(self, relation, deg_type):
        return [
            self.graph[node_id]
            for node_id, rel in getattr(self, deg_type)
            if rel == relation
        ]

    def contained_by(self):
        return self.filter_deg(Relation.INSIDE, "outdeg")

    def contains(self):
        return self.filter_deg(Relation.INSIDE, "indeg")

    def supported_by(self):
        return self.filter_deg(Relation.ON, "outdeg")

    def supports(self):
        return self.filter_deg(Relation.ON, "indeg")

    def holds(self):
        lh = self.filter_deg(Relation.HOLDS_LH, "outdeg")
        rh = self.filter_deg(Relation.HOLDS_RH, "outdeg")
        return lh + rh

    def held_by(self):
        lh = self.filter_deg(Relation.HOLDS_LH, "indeg")
        rh = self.filter_deg(Relation.HOLDS_RH, "indeg")
        return lh + rh

    def close(self):
        idg = self.filter_deg(Relation.CLOSE, "indeg")
        odg = self.filter_deg(Relation.CLOSE, "outdeg")
        assert {n.id for n in idg} == {n.id for n in odg}, (idg, odg)
        return idg

    @staticmethod
    def _assure_unique(elems, fix):
        elems = set(elems)
        match len(elems):
            case 0:
                return None
            case 1:
                return elems.pop()
            case _:
                if fix:
                    return elems
                else:
                    raise AssertionError(elems)

    def get_room(self, fix=False):
        rooms = []
        for obj in self.contained_by():
            if obj.category == "Rooms":
                rooms.append(obj)
            else:
                ctnrs = obj.contained_by()
                assert all(ctnr.category == "Rooms" for ctnr in ctnrs)
                rooms.extend(ctnrs)
        return self._assure_unique(rooms, fix)

    def get_ctnr(self, fix=False):
        ctnrs = []
        for obj in self.contained_by():
            if obj.category == "Rooms":
                continue
            else:
                ctnrs.append(obj)
        return self._assure_unique(ctnrs, fix)

    def get_srfc(self, fix=False):
        srfc = self.supported_by()
        srfc = [x for x in srfc if x.class_name not in ("floor", "rug")]
        return self._assure_unique(srfc, fix)

    def get_location(self, fix=False):
        room = self.get_room(fix)
        ctnr = self.get_ctnr(fix)
        srfc = self.get_srfc(fix)
        return room, ctnr, srfc

    @staticmethod
    def res(n):
        return (None, None) if n is None else (n.id, n.class_name)

    def __format__(self, format_spec):
        return format(str(self), format_spec)

    @property
    def is_grabbable(self) -> bool:
        return Property.GRABBABLE in self.properties

    @property
    def is_container(self) -> bool:
        return Property.CONTAINERS in self.properties

    @property
    def is_surface(self) -> bool:
        return Property.SURFACES in self.properties

    @property
    def is_open(self) -> bool:
        return State.OPEN in self.states

    @property
    def is_closed(self) -> bool:
        return State.CLOSED in self.states

    def walk_target(self) -> "GN | None":
        """
        Return the node the agent should walk towards to interact with this object.
        """
        if self.is_grabbable:
            ctnr = self.get_ctnr(fix=True)
            srfc = self.get_srfc(fix=True)
            if isinstance(ctnr, set):
                ctnr = next(iter(ctnr))
            if isinstance(srfc, set):
                srfc = next(iter(srfc))
            parent = ctnr or srfc
            if parent is not None and (parent.is_container or parent.is_surface):
                return parent
            # No parent
            return self
        if self.is_container or self.is_surface:
            return self
        return None

    @property
    def instance_num(self) -> int:
        """1-based index among all nodes with the same class_name in the graph."""
        same = self.graph._class_name_map.get(self.class_name, [])
        for i, n in enumerate(same, start=1):
            if n.id == self.id:
                return i
        return 1

    def to_gui_dict(self, agent: "GN", held_ids: set[int]) -> dict:
        """
        Serialise this node into the dict shape the GUI frontend expects.
        `agent` is the human character node.
        `held_ids` is the set of object ids currently held by the agent.

        walktowards is always included for every item and furniture so the user
        can navigate toward any object.
        Grab is offered whenever the agent holds fewer than 2 items.
        """
        is_close   = self in agent.close()
        is_grabbed = self.id in held_ids
        walk_tgt   = self.walk_target()
        can_grab_more = len(held_ids) < 2

        actions: list[str] = []

        if self.is_container or self.is_surface:
            # Furniture path — but if this node is also grabbable and has a parent
            # (e.g. a plate that is also a surface, sitting inside a fridge),
            # redirect the walk action to the parent rather than self.
            if self.is_grabbable and walk_tgt is not None and walk_tgt.id != self.id:
                parent_is_close = walk_tgt in agent.close()
                actions.append("walktowards_parent")
                if parent_is_close:
                    if walk_tgt.is_container and not walk_tgt.is_open:
                        pass
                    elif can_grab_more:
                        actions.append("grab")
            else:
                actions.append("walktowards")
                if is_close:
                    if self.is_container:
                        actions.append("close" if self.is_open else "open")
                        if held_ids and self.is_open:
                            actions.append("putin")
                    if self.is_surface and not self.is_container and held_ids:
                        actions.append("putback")
                    if self.is_grabbable and can_grab_more:
                        actions.append("grab")

        elif self.is_grabbable:
            # Item
            if walk_tgt is not None and walk_tgt.id != self.id:
                parent_is_close = walk_tgt in agent.close()
                # walk to the parent container/surface
                actions.append("walktowards_parent")
                if parent_is_close:
                    if walk_tgt.is_container and not walk_tgt.is_open:
                        pass
                    elif can_grab_more:
                        actions.append("grab")
            else:
                # Item is directly in the room
                actions.append("walktowards")
                if is_close and can_grab_more:
                    actions.append("grab")

        d: dict = {
            "id":           self.id,
            "class_name":   self.class_name,
            "instance_num": self.instance_num,
            "is_close":     is_close,
            "is_open":      self.is_open,
            "is_grabbed":   is_grabbed,
            "is_grabbable": self.is_grabbable,
            "is_container": self.is_container,
            "is_surface":   self.is_surface,
            "actions":      actions,
        }
        # items inside containers or on surfaces (for furniture gui display)
        if self.is_container or self.is_surface:
            contents: list[dict] = []
            if self.is_container:
                for child in self.contains():
                    contents.append({
                        "id":           child.id,
                        "class_name":   child.class_name,
                        "instance_num": child.instance_num,
                        "is_grabbable": child.is_grabbable,
                        "relation":     "inside",
                    })
            if self.is_surface and not self.is_container:
                for child in self.supports():
                    contents.append({
                        "id":           child.id,
                        "class_name":   child.class_name,
                        "instance_num": child.instance_num,
                        "is_grabbable": child.is_grabbable,
                        "relation":     "on",
                    })
            d["contents"] = contents
        # items inside containers or on surfaces (for item gui display)
        if self.is_grabbable and walk_tgt is not None and walk_tgt.id != self.id:
            d["walk_to_id"]    = walk_tgt.id
            d["walk_to_class"] = walk_tgt.class_name
            d["walk_to_instance_num"] = walk_tgt.instance_num
            if walk_tgt.is_container:
                d["container_id"]       = walk_tgt.id
                d["container_class"]    = walk_tgt.class_name
                d["container_instance_num"] = walk_tgt.instance_num
                d["container_is_open"]  = walk_tgt.is_open
                d["container_is_close"] = walk_tgt in agent.close()
        else:
            d["walk_to_id"]    = None
            d["walk_to_class"] = None
            d["walk_to_instance_num"] = None

        return d


class GE(GraphEdge):
    def __str__(self):
        return str((self.from_node, self.relation, self.to_node))

    def __repr__(self):
        return self.__str__()


@dataclass
class Subgoal:
    name: str
    cnt: int
    obj_nodes: list[GN]
    tgt_nodes: list[GN]

    def __post_init__(self):
        self.prep = self.name.split("_")[0]

    def __str__(self):
        return f"{self.tgt_nodes} <== {self.cnt} of {self.obj_nodes}"

    def __repr__(self):
        return self.__str__()

    @property
    def natlang(self):
        obj_name = self.obj_nodes[0].class_name
        tgt = self.tgt_nodes[0]
        tgt_label = f"{tgt.class_name}.{tgt.instance_num}"
        return f"Put {self.cnt} {obj_name} {self.prep} the {tgt_label}"


class Goal:
    def __init__(self, goal, eg):
        # * convert `graph_dict` to `eg`
        if isinstance(eg, dict):
            eg = EG(eg)
        self.eg = eg

        # * convert `goal` to `goal_spec`
        if isinstance(list(goal.values())[0], int):
            goal = utils_env.convert_goal(goal, eg._dictionary)
        self.spec = goal

        self.subgoals = []
        for subgoal_name, subgoal in self.spec.items():
            name = subgoal_name
            cnt = subgoal["count"]
            obj_ids = subgoal["grab_obj_ids"]
            tgt_ids = subgoal["container_ids"]
            obj_nodes = [eg[obj_id] for obj_id in obj_ids]
            tgt_nodes = [eg[tgt_id] for tgt_id in tgt_ids]
            self.subgoals.append(Subgoal(name, cnt, obj_nodes, tgt_nodes))

    def __str__(self):
        return "\n".join([str(subgoal) for subgoal in self.subgoals])

    def __repr__(self):
        return self.__str__()

    @property
    def natlang(self):
        return "\n".join([f"[{i}] {sg.natlang}" for i, sg in enumerate(self.subgoals)])

    def check_progress(self):
        sat, unsat = utils_env.check_progress2(self.eg._dictionary, self.spec)
        return sat, unsat


class EG(EnvironmentGraph):
    def __init__(self, dictionary):
        super().__init__(dictionary)

        self.lg = get_my_logger()
        self.lg_res = get_my_logger(name="shunchi_res_py")
        self._dictionary = dictionary

    def _from_dictionary(self, d):
        nodes = [GN.from_dict(n) for n in d["nodes"]]
        for n in nodes:
            self._node_map[n.id] = n
            self._class_name_map.setdefault(n.class_name, []).append(n)
            if n.id > self._max_node_id:
                self._max_node_id = n.id

        for ed in d["edges"]:
            from_id = ed["from_id"]
            relation = Relation[ed["relation_type"].upper()]
            to_id = ed["to_id"]

            es = self._edge_map.setdefault((from_id, relation), {})
            es[to_id] = self._node_map[to_id]

            self._node_map[from_id].outdeg.append([to_id, relation])
            self._node_map[to_id].indeg.append([from_id, relation])

        for node_id in self._node_map.keys():
            self._node_map[node_id].graph = self

    def __getitem__(self, node_id):
        return self._node_map[node_id]

    def filter_nodes(self, filter):
        return [node for node in self._node_map.values() if filter(node)]

    def has_class(self, class_name):
        return len(list(self.get_nodes_by_attr("class_name", class_name))) > 0

    def inspect(self):
        n_nodes = len(self.get_nodes())
        n_edges = len(self.get_from_pairs())
        agents = self.get_agents()
        agent_ids = [a.id for a in agents]
        return f"#nodes = {n_nodes}, #edges = {n_edges}, agents = {agent_ids}"

    def get_rooms(self):
        rooms = self.get_nodes_by_attr("category", "Rooms")
        return list(rooms)

    def get_containers(self):
        containers = self.filter_nodes(lambda x: Property.CONTAINERS in x.properties)
        return list(containers)

    def get_surfaces(self):
        surfaces = self.filter_nodes(lambda x: Property.SURFACES in x.properties)
        return list(surfaces)

    def get_agents(self):
        agents = self.get_nodes_by_attr("class_name", "character")
        return list(agents)

    def goal_table(self, goal, title):
        assert self.lg is not None
        # raise NotImplementedError("node.get_location() bug")
        log_content = format(title, "-^120")
        self.lg.info(log_content)
        self.lg_res.debug(log_content)
        log_content = format_row(["Object", "Room", "Container", "Surface"], "<30")
        self.lg.debug(log_content)
        self.lg_res.debug(log_content)

        ret = dict()

        for subgoal_name, subgoal_spec in goal.items():
            log_content = format(subgoal_name, ".^30")
            self.lg.debug(log_content)
            self.lg_res.debug(log_content)
            obj_ids = subgoal_spec["grab_obj_ids"]
            tgt_ids = subgoal_spec["container_ids"]

            for obj_id in obj_ids:
                obj = self[obj_id]
                if obj is None:
                    log_content = format_row([obj_id] + ["Unknown"] * 3, "<30")
                else:
                    room, ctnr, srfc = obj.get_location()
                    ret[GN.res(obj)] = dict(
                        room=GN.res(room),
                        ctnr=GN.res(ctnr),
                        srfc=GN.res(srfc),
                    )
                    log_content = format_row([obj, room, ctnr, srfc], "<30")
                    self.lg_res.debug(log_content)
                self.lg.debug(log_content)

            for tgt_id in tgt_ids:
                tgt = self[tgt_id]
                if tgt is None:
                    log_content = format_row([tgt_id] + ["Unknown"] * 3, "<30")
                else:
                    room, ctnr, srfc = tgt.get_location()
                    ret[GN.res(tgt)] = dict(
                        room=GN.res(room),
                        ctnr=GN.res(ctnr),
                        srfc=GN.res(srfc),
                    )
                    log_content = format_row([tgt, room, ctnr, srfc], "<30")
                    self.lg_res.debug(log_content)
                self.lg.debug(log_content)

        # self.lg.info(format(" Describe Done ", "=^120") + "\n")

        return ret

    def actions_to_natlang(self, actions, init_room=None, name="Human"):
        actions = dedup_list(actions)
        lines = []
        if init_room is not None:
            lines.append(f"{name} is in the {init_room}")
        for action in actions:
            if action is None:
                continue
            parsed = parse_action(action)
            predicate = parsed[0]
            match predicate:
                case "walk" | "walktowards" | "run":
                    line = f"{name} walks to the {parsed[1]}"
                case "walkforward":
                    line = f"{name} walks forward"
                case "turnleft":
                    line = f"{name} turns left"
                case "turnright":
                    line = f"{name} turns right"
                case "putback":
                    prep = "on"
                    line = f"{name} puts the {parsed[1]} {prep} the {parsed[3]}"
                case "putin":
                    prep = "in"
                    line = f"{name} puts the {parsed[1]} {prep} the {parsed[3]}"
                case "grab":
                    line = f"{name} grabs the {parsed[1]}"
                case "open":
                    line = f"{name} opens the {parsed[1]}"
                    # note = self.describe_related_objects(
                    #     self[int(parsed[2])], "contains"
                    # )
                    # line += f" ({note})"
                case "close":
                    line = f"{name} closes the {parsed[1]}"
                case "switchon":
                    line = f"{name} switches on the {parsed[1]}"
                case "switchoff":
                    line = f"{name} switches off the {parsed[1]}"
                case "sit":
                    line = f"{name} sits on the {parsed[1]}"
                case "standup":
                    line = f"{name} stands up"
                case "drink":
                    line = f"{name} drinks from the {parsed[1]}"
                case "touch":
                    line = f"{name} touches the {parsed[1]}"
                case "lookat":
                    line = f"{name} looks at the {parsed[1]}"
                case _:
                    raise ValueError(parsed)
            lines.append(line)
        # return "\n".join([f"[{i}] {line}" for i, line in enumerate(lines)])
        return lines

    def describe_related_objects(self, node, predicate):
        """given a container or surface, describe the objects inside or on it"""

        name = node.class_name
        objs = Counter([n.class_name for n in getattr(node, predicate)()])
        objs = [f"{cnt} {obj}" for obj, cnt in objs.items() if obj in OBJECT_NAMES]
        if len(objs) > 0:
            return f"the {name} {predicate} {', '.join(objs)}"
        else:
            return f"the {name} {predicate} nothing"

    def env_state(self, ctnr_ids, srfc_ids):
        tree = dict()  # room => ctnr|srfc => obj

        rooms = self.get_rooms()
        for room in rooms:
            tree[room.class_name] = []

        for mid_id in ctnr_ids | srfc_ids:
            if mid_id in ctnr_ids:
                predicate = "contains"
            elif mid_id in srfc_ids:
                predicate = "supports"
            else:
                raise ValueError(mid_id)

            mid = self[mid_id]
            room = mid.get_room()
            objs_str = self.describe_related_objects(mid, predicate)
            tree[room.class_name].append([mid.class_name, objs_str])

        room_list = [r.class_name for r in rooms]
        story = f"The apartment has {len(rooms)} rooms: {', '.join(room_list)}."
        for room_name, room_info in tree.items():
            room_info = sorted(room_info)
            # * skip empty rooms
            if len(room_info) == 0:
                continue
            story += "\n"
            # * describe counter(ctnr|srfc)
            mid_names = Counter([mid_name for mid_name, _ in room_info])
            mid_names = [f"{cnt} {mid_name}" for mid_name, cnt in mid_names.items()]
            story += f"\nThe {room_name} has {', '.join(mid_names)}."
            # * describe each ctnr|srfc
            for _, objs_str in room_info:
                if "nothing" in objs_str:
                    continue
                story += f"\n- {objs_str.capitalize()}."
        return story

    def agent_state_natlang(self, agent_id, name):
        agent = self[agent_id]
        close_objs = agent.close()
        in_hands = agent.holds()
        room_name = agent.get_room().class_name
        lines = [f"{name} is in the {room_name}."]
        if len(close_objs) > 0:
            close_objs = Counter([o.class_name for o in close_objs])
            lines.append(
                f"{name} is close to {', '.join(f'{cnt} {obj}' for obj, cnt in close_objs.items())}."
            )
        if len(in_hands) > 0:
            in_hands = Counter([o.class_name for o in in_hands])
            lines.append(
                f"{name} is holding {', '.join(f'{cnt} {obj}' for obj, cnt in in_hands.items())}."
            )
        else:
            lines.append(f"{name} is holding nothing.")
        return "\n".join(lines)

    def goal_rooms(self, goal):
        obj_rooms = []
        tgt_rooms = []

        goal = Goal(goal, self)
        obj_rooms = [n.get_room() for sg in goal.subgoals for n in sg.obj_nodes]
        tgt_rooms = [n.get_room() for sg in goal.subgoals for n in sg.tgt_nodes]

        obj_rooms = [room for room, cnt in Counter(obj_rooms).most_common()]
        tgt_rooms = [room for room, cnt in Counter(tgt_rooms).most_common()]

        return obj_rooms, tgt_rooms

    def gui_state(self, agent_id: int, arena=None) -> dict:
        """
        Return a GUI state dict for the human agent.

        Shape:
          current_room:  {id, class_name}
          rooms:         [{id, class_name}]
          held_objects:  [{id, class_name, instance_num}]
          items:         [GN.to_gui_dict, ...]   — grabbable objects in current room
          furniture:     [GN.to_gui_dict, ...]   — containers/surfaces in current room
          helper:        {last_action, held_objects, agent_type}  — helper info
        """
        agent = self._node_map.get(agent_id)
        assert agent is not None, f"Agent with id {agent_id} not found in graph"

        # rooms
        rooms_list = [
            {"id": r.id, "class_name": r.class_name}
            for r in self.get_rooms()
        ]

        # current room
        current_room_node = agent.get_room(fix=True)
        current_room = (
            {"id": current_room_node.id, "class_name": current_room_node.class_name}
            if current_room_node else {}
        )

        # held objects
        held_lh = agent.filter_deg(Relation.HOLDS_LH, "outdeg")
        held_rh = agent.filter_deg(Relation.HOLDS_RH, "outdeg")
        held_ids_lh = {n.id for n in held_lh}
        held_ids_rh = {n.id for n in held_rh}
        held_nodes = held_lh + held_rh
        held_ids   = {n.id for n in held_nodes}
        held_objects = [
            {
                "id": n.id,
                "class_name": n.class_name,
                "instance_num": n.instance_num,
                "hand": "LH" if n.id in held_ids_lh else "RH",
            }
            for n in held_nodes
        ]

        # all objects inside the current room
        if current_room_node is None:
            return {"current_room": current_room, "rooms": rooms_list,
                    "held_objects": held_objects, "items": [], "furniture": []}

        room_obj_nodes = [
            self[from_id]
            for from_id, rel in current_room_node.indeg
            if rel == Relation.INSIDE and from_id in self._node_map
        ]

        # Collect all grabbable items in this room
        # 1. directly in room
        # 2. inside a container in this room,
        # 3. on a surface in this room
        seen_item_ids: set[int] = set()
        items: list[dict] = []

        for node in room_obj_nodes:
            if node.category == "Rooms":
                continue

            if node.is_grabbable and node.id not in held_ids and node.id not in seen_item_ids:
                seen_item_ids.add(node.id)
                d = node.to_gui_dict(agent, held_ids)
                items.append(d)

            if node.is_container:
                for child in node.contains():
                    if child.is_grabbable and child.id not in seen_item_ids and child.id not in held_ids:
                        seen_item_ids.add(child.id)
                        d = child.to_gui_dict(agent, held_ids)
                        items.append(d)

            if node.is_surface and not node.is_container:
                for child in node.supports():
                    if child.is_grabbable and child.id not in seen_item_ids and child.id not in held_ids:
                        seen_item_ids.add(child.id)
                        d = child.to_gui_dict(agent, held_ids)
                        items.append(d)

        items.sort(key=lambda o: (not o["is_close"], o["class_name"], o["instance_num"]))

        # collect furniture
        furniture: list[dict] = []
        for node in room_obj_nodes:
            if not (node.is_container or node.is_surface):
                continue
            d = node.to_gui_dict(agent, held_ids)
            furniture.append(d)
        furniture.sort(key=lambda o: (not o["is_close"], o["class_name"], o["instance_num"]))

        # Helper agent info
        helper: dict = {"last_action": None, "held_objects": [], "agent_type": "unknown"}
        if arena is not None:
            try:
                ai_agent = next(
                    (a for a in arena.agents if a.agent_type != "Human"), None
                )
                if ai_agent is not None:
                    helper["agent_type"] = ai_agent.agent_type
                    # last action from episode history
                    try:
                        action_hist = arena.saver.episode_saved_info.get(
                            "action", {}
                        ).get(ai_agent.char_index, [])
                        nonempty = [a for a in action_hist if a is not None]
                        helper["last_action"] = nonempty[-1] if nonempty else None
                    except Exception:
                        pass
                    # held objects from the already-built graph
                    try:
                        ai_node = self._node_map.get(ai_agent.agent_id)
                        if ai_node is not None:
                            ai_held_lh = {n.id for n in ai_node.filter_deg(Relation.HOLDS_LH, "outdeg")}
                            ai_held_rh = {n.id for n in ai_node.filter_deg(Relation.HOLDS_RH, "outdeg")}
                            helper["held_objects"] = [
                                {
                                    "id": n.id,
                                    "class_name": n.class_name,
                                    "instance_num": n.instance_num,
                                    "hand": "LH" if n.id in ai_held_lh else "RH",
                                }
                                for n in ai_node.holds()
                            ]
                    except Exception:
                        pass
            except Exception:
                pass
        return {
            "current_room": current_room,
            "rooms":        rooms_list,
            "held_objects": held_objects,
            "items":        items,
            "furniture":    furniture,
            "helper":       helper,
        }

    def goal_tree(self, goal, title):
        assert self.lg is not None
        self.lg.info(format(title, "-^120"))
        self.lg.info(format("objects", ".^30"))
        obj_nodes = []
        for subgoal_name, subgoal_spec in goal.items():
            obj_ids = subgoal_spec["grab_obj_ids"]
            for obj_id in obj_ids:
                obj_node = self[obj_id]
                obj_nodes.append(obj_node)

                self.lg.info(obj_node)
                in_objs = obj_node.contained_by()
                on_objs = obj_node.supported_by()
                if len(in_objs) > 0:
                    self.lg.debug(f"    <in> {in_objs} </in>")
                if len(on_objs) > 0:
                    self.lg.debug(f"    <on> {on_objs} </on>")

        self.lg.info(obj_nodes)
        # self.lg.info(format(" Describe Done ", "=^120") + "\n")

    def room_tree(self, title):
        raise NotImplementedError("Not implemented")
        assert self.lg is not None
        exclude_classes = ["floor", "wall", "ceiling", "window"]
        # * describe by room
        rooms = self.get_rooms()
        for room in rooms:
            self.lg.debug(format(str(room), ".^30"))
            xs = room.contains(self)
            for x in xs:
                if x.class_name in exclude_classes:
                    continue

                in_objs = [_ for _ in x.contains(self) if _ in obj_nodes]
                on_objs = [_ for _ in x.surface(self) if _ in obj_nodes]
                if len(in_objs) == len(on_objs) == 0:
                    continue
                self.lg.debug(x)
                if len(in_objs) > 0:
                    self.lg.debug(f"    <in> {in_objs} </in>")
                if len(on_objs) > 0:
                    self.lg.debug(f"    <on> {on_objs} </on>")

        self.lg.debug("Done Rooms")

        # * describe by agent
        agents = self.get_agents()
        for agent in agents:
            self.lg.debug(format(str(agent), ".^30"))


def fix_multiple_location(env_task_set, verbose=False, drop_env=False):
    """
    Only fix multiple location of goal objects,
    by simply keeping the first one.
    """

    def keep_first(edges, obj_id, relation_type, error_nodes):
        return [
            edge
            for edge in edges
            if not all(
                [
                    edge["from_id"] == obj_id,
                    edge["to_id"] in [x.id for x in error_nodes][1:],
                    edge["relation_type"] == relation_type,
                ]
            )
        ]

    bug_tasks = set()

    for ith_task, env_task in enumerate(env_task_set):
        eg = EG(env_task["init_graph"])
        edges = env_task["init_graph"]["edges"]
        human_goal = utils_env.convert_goal(
            env_task["task_goal"][0], env_task["init_graph"]
        )
        for subgoal_name, subgoal in human_goal.items():
            for obj_id in subgoal["grab_obj_ids"]:
                obj = eg[obj_id]
                room, ctnr, srfc = obj.get_location(fix=True)
                if isinstance(room, set):
                    if verbose:
                        print(f"ROOM [{ith_task:02d}] {obj:<20} {room}")
                    env_task["init_graph"]["edges"] = keep_first(
                        edges, obj_id, "INSIDE", room
                    )
                    bug_tasks.add(ith_task)
                if isinstance(ctnr, set):
                    if verbose:
                        print(f"CTNR [{ith_task:02d}] {obj:<20} {ctnr}")
                    env_task["init_graph"]["edges"] = keep_first(
                        edges, obj_id, "INSIDE", ctnr
                    )
                    bug_tasks.add(ith_task)
                if isinstance(srfc, set):
                    if verbose:
                        print(f"SRFC [{ith_task:02d}] {obj:<20} {srfc}")
                    env_task["init_graph"]["edges"] = keep_first(
                        edges, obj_id, "ON", srfc
                    )
                    bug_tasks.add(ith_task)

    if drop_env:
        if verbose:
            print(f"Dropping tasks: {bug_tasks}")
        env_task_set = [
            env_task for i, env_task in enumerate(env_task_set) if i not in bug_tasks
        ]

    if verbose:
        print(Counter([x["env_id"] for x in env_task_set]).most_common())
        print(Counter([x["task_name"] for x in env_task_set]).most_common())

    return env_task_set