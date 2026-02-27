"""
API
  GET  /api/state   current graph snapshot (rooms, objects, agent state)
  GET  /api/task    task metadata and goal progress
  GET  /api/image   current camera images (base64 JPEG)
  POST /api/action  submit a human action (walk, grab, put, open, close…)
  GET  /api/ready   wait until the next observation is ready after an action
  POST /api/reset   request an episode reset (handled by arena thread)
"""

import base64
import io
import threading
from typing import TYPE_CHECKING
from time import sleep
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from utils.utils_graph import EG, Goal

if TYPE_CHECKING:
    from agents.Human_agent import Human_agent
    from envs.arena import Arena
    from envs.unity_environment import UnityEnvironment

ZERO_ARG_ACTIONS = {
    "walkforward",
    "turnleft",
    "turnright",
    "standup",
}

ONE_ARG_ACTIONS = {
    "walk",
    "run",
    "walktowards",
    "sit",
    "grab",
    "open",
    "close",
    "switchon",
    "switchoff",
    "drink",
    "touch",
    "lookat",
}

TWO_ARG_ACTIONS = {
    "put",
    "putback",
    "putin",
}

def _build_action_script(
    action: str,
    obj_id: int | None,
    obj_class: str | None,
    target_id: int | None,
    target_class: str | None,
    char_index: int,
) -> str | None:
    """
    Translate a GUI action request into a VirtualHome script string.

    Format: [action] <class_name> (id)
            [action] <obj_class> (obj_id) <target_class> (target_id)
    """
    if action in ONE_ARG_ACTIONS:
        if obj_id is None or obj_class is None:
            return None
        return f"[{action}] <{obj_class}> ({obj_id})"

    elif action in TWO_ARG_ACTIONS:
        if None in (obj_id, obj_class, target_id, target_class):
            return None
        return f"[{action}] <{obj_class}> ({obj_id}) <{target_class}> ({target_id})"

    return None


def _encode_image(np_img) -> str:
    """Convert an H×W×3 uint8 numpy array (RGB or BGR) to base64 JPEG string."""
    img = Image.fromarray(np_img.astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class HumanServer:
    def __init__(self, arena: "Arena", human_agent: "Human_agent", env: "UnityEnvironment"):
        self.arena = arena
        self.human_agent = human_agent
        self.env = env

        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self._register_routes()


    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template("index.html")


        @app.route("/api/state", methods=["GET"])
        def api_state():
            """
            Returns the current environment graph structure.

            Response:
            {
              "current_room": {"id": int, "class_name": str},
              "rooms": [{"id": int, "class_name": str}, ...],
              "held_objects": [{"id": int, "class_name": str, "instance_num": int}],
              "visible_objects": [{
                "id": int, "class_name": str, "instance_num": int,
                "is_close": bool, "is_open": bool, "is_grabbed": bool,
                "is_grabbable": bool, "is_container": bool, "is_surface": bool,
                "actions": [str, ...]          ← valid action names for this object
              }, ...]
            }
            """
            graph = self.env.get_graph()
            state = EG(graph).gui_state(self.human_agent.agent_id)
            return jsonify(state)

        @app.route("/api/task", methods=["GET"])
        def api_task():
            """
            Returns task metadata and goal progress.

            Response:
            {
              "task_name": str,
              "task_id": int,
              "env_id": int,
              "step": int,
              "max_steps": int,
              "goals": {
                "<predicate_key>": {
                  "count_needed": int,
                  "count_done": int,
                  "label": str       ← human-readable description
                }, ...
              },
              "progress": {
                "done":    {"<obj_class>": int, ...},
                "holding": {"<obj_class>": int, ...}
              },
              "episode_done": bool
            }
            """
            progress = self.human_agent.get_progress()

            task_info = {
            "task_name": self.env.task_name,
            "task_id": self.env.task_id,
            "env_id": self.env.env_id,
            "gt_goals": self.env.task_goal[0],
            "step": self.env.steps,
        }


            goals_out: dict = {}
            try:
                graph = self.env.get_graph()
                goal  = Goal(self.env.task_goal[0], graph)
                sat, unsat = goal.check_progress()
                for sg in goal.subgoals:
                    needed = sg.cnt
                    done   = min(len(sat[sg.name]), needed)
                    goals_out[sg.name] = {
                        "count_needed": needed,
                        "count_done":   done,
                        "label":        sg.natlang,
                    }
            except Exception as e:
                self.app.logger.error(f"/api/task goals failed: {e}", exc_info=True)

            return jsonify({
                **task_info,
                "max_steps": getattr(self.env, "max_episode_length", 0),
                "goals": goals_out,
                "progress": progress,
                "episode_done": False,
            })

        @app.route("/api/image", methods=["GET"])
        def api_image():
            """
            Returns camera images for the human agent's viewpoint.

            Response:
            {
              "images": ["<base64-jpeg>", ...]
            }
            """
            width = int(request.args.get("width", 400))
            height = int(request.args.get("height", 225))

            images_b64: list[str] = []
            try:
                obs = self.env.get_observation(
                    agent_id=self.human_agent.char_index,
                    obs_type="image",
                    info=dict(
                        view="third_behind",
                        image_width=width,
                        image_height=height,
                    ),
                )
                if obs is not None:
                    images_b64.append(_encode_image(obs))
            except Exception as e:
                self.app.logger.warning(f"/api/image failed: {e}")

            return jsonify({"images": images_b64})


        @app.route("/api/action", methods=["POST"])
        def api_action():
            """
            Submit one human action for the current step.

            Request body (JSON):
            {
              "action":       str,        ← "walk" | "grab" | "putback" | "putin" |
                                             "open" | "close" | "switchon" | "switchoff"
              "obj_id":       int,
              "obj_class":    str,
              "target_id":    int | null, ← required for putback / putin
              "target_class": str | null
            }

            Response (success):
            { "accepted": true, "action_str": "<char1> [grab] <apple> (42)" }

            Response (duplicate / already submitted this step):
            { "accepted": false, "reason": "already_submitted" }

            Response (invalid action params):
            { "accepted": false, "reason": "invalid_params", "detail": "..." }
            """
            data = request.get_json(force=True, silent=True) or {}

            action = data.get("action", "")
            obj_id = data.get("obj_id")
            obj_class = data.get("obj_class")
            target_id = data.get("target_id")
            target_class = data.get("target_class")

            script = _build_action_script(
                action=action,
                obj_id=obj_id,
                obj_class=obj_class,
                target_id=target_id,
                target_class=target_class,
                char_index=self.human_agent.char_index,
            )

            if script is None:
                return jsonify({
                    "accepted": False,
                    "reason": "invalid_params",
                    "detail": (
                        f"Cannot build script for action='{action}' "
                        f"obj_id={obj_id} obj_class={obj_class} "
                        f"target_id={target_id} target_class={target_class}"
                    ),
                }), 400

            accepted = self.human_agent.submit_action(script)

            if not accepted:
                return jsonify({
                    "accepted": False,
                    "reason": "already_submitted",
                }), 409

            return jsonify({"accepted": True, "action_str": script})

        @app.route("/api/ready", methods=["GET"])
        def api_ready():
            """
            Blocks until the arena has completed the current step and published
            a new observation.

            The frontend calls this after submitting an action and waits here
            until the arena loop has consumed that action, advanced the
            simulation, and made the next state available.

            Query params:
            timeout  (float, default 10.0)  — max seconds to wait

            Response when step is done:
            { "ready": true }

            Response on timeout (frontend should retry or show error):
            { "ready": false, "reason": "timeout" }
            """
            timeout = float(request.args.get("timeout", 10.0))
            became_ready = self.human_agent._obs_ready.wait(timeout=timeout)
            if became_ready:
                return jsonify({"ready": True})
            return jsonify({"ready": False, "reason": "timeout"}), 202

        @app.route("/api/reset", methods=["POST"])
        def api_reset():
            """
            Request a task reset. Injects '__RESET__' that the
            arena loop detects and converts into an episode reset.
            Response: { "accepted": true }   or   { "accepted": false, "reason": "..." }
            """
            accepted = self.human_agent.submit_action("__RESET__")
            if not accepted:
                return jsonify({"accepted": False, "reason": "already_submitted"}), 409
            return jsonify({"accepted": True})

    def run(self, host: str = "0.0.0.0", port: int = 5005, debug: bool = False):
        """Start the Flask dev server. Call from a daemon thread."""
        self.app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)