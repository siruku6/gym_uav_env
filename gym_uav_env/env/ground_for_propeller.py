import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from env.drone_with_propeller import Drone, MachineState, PositionCoordinates
from gymnasium.envs.registration import EnvSpec

PropellerAngularVelocity = Tuple[float, float, float, float, float]


# 16次元の状態空間
class State(NamedTuple):
    u: float
    rot_v_x: float
    rot_v_y: float
    rot_v_z: float
    rot_x: float
    rot_y: float
    rot_z: float
    v_x: float
    v_y: float
    v_z: float
    pos_x: float
    pos_y: float
    pos_z: float
    goal_x: float
    goal_y: float
    goal_z: float
    distance_from_goal: float


class FlightRecorder:
    def __init__(self, goal: Tuple[float, float, float]) -> None:
        self.history: list[dict] = []
        self.goal = goal

    def record_step(
        self,
        obs: State,
        action: List[float],
        re: float,
        done: bool,
        info: Dict[str, Optional[str]],
    ) -> None:
        u = obs[0]
        rot_v_x = obs[1]
        rot_v_y = obs[2]
        rot_v_z = obs[3]
        rot_x = obs[4]
        rot_y = obs[5]
        rot_z = obs[6]
        v_x = obs[7]
        v_y = obs[8]
        v_z = obs[9]
        pos_x = obs[10]
        pos_y = obs[11]
        pos_z = obs[12]
        goal_x = obs[13]
        goal_y = obs[14]
        goal_z = obs[15]
        distance_from_goal = obs[16]
        self.history.append(
            {
                "u": u,
                "propeller_ang_v_base": action[0],
                "propeller_ang_v_1": action[1],
                "propeller_ang_v_2": action[2],
                "propeller_ang_v_3": action[3],
                "propeller_ang_v_4": action[4],
                "rot_v_x": rot_v_x,
                "rot_v_y": rot_v_y,
                "rot_v_z": rot_v_z,
                "rot_x": rot_x,
                "rot_y": rot_y,
                "rot_z": rot_z,
                "v_x": v_x,
                "v_y": v_y,
                "v_z": v_z,
                "pos_x": pos_x,
                "pos_y": pos_y,
                "pos_z": pos_z,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_z": goal_z,
                "distance_from_goal": distance_from_goal,
                "reward": re,
                "done": done,
                "info": info,
            }
        )

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


class PropellerDroneFlight(gymnasium.Env):
    # この環境ではrenderのモードとしてrgb_arrayのみを用意していることを宣言しておく
    # GymのWrapperなどから参照される可能性がある
    metadata = {"render_modes": ["rgb_array"]}
    # https://gymnasium.farama.org/api/env/#gymnasium.Env.render

    def __init__(
        self,
        render_mode: str,
        drone: Optional[Drone] = None,
        weight_approach_reward: float = 5.0,
        weight_force_penalty: float = 0.01,
        gravity_enabled: bool = True,
        debug_mode: bool = False,
        complement_force: bool = False,
        max_episode_steps: int = 400,
        recorder: Optional[FlightRecorder] = None,
        specified_goal: Optional[tuple[float, float, float]] = None,
    ) -> None:
        super(PropellerDroneFlight, self).__init__()

        print("[INFO] v0.1.0: added the state denoting the distance from a goal")

        # 環境の登録の代替
        # https://developers.agirobots.com/jp/openai-gym-custom-env/
        self.spec = EnvSpec(id="PropellerDroneFlight")

        self.render_mode = render_mode
        # if render_mode == "rgb_array":
        #     self.fig = plt.figure(figsize=(4, 4))
        # self.ax = self.fig.add_subplot(111, projection='3d')

        self.drone: Drone = drone
        self.gravity_enabled: bool = gravity_enabled
        self.steps: int = None
        self._max_episode_steps: int = max_episode_steps  # 200
        self.specified_goal: Optional[tuple[float, float, float]] = specified_goal
        print("[INFO] A specified goal position:", self.specified_goal)
        self.goal: Tuple[float, float, float] = specified_goal
        self.traveling_distance: float = 0.0

        if self.specified_goal is not None:
            traveling_distance: float = self._calc_distance_from_goal(0.0, 0.0, 0.0)
            print("[INFO] traveling_distance:", traveling_distance)

        self.state: State = None
        self.profit: float = None

        # ------------------------------------------------------------------------
        # 行動空間
        # ------------------------------------------------------------------------
        MIN_ROTATION_V: int = -10  # 最大出力
        MAX_ROTATION_V: int = 10  # 反対方向に出せる最大出力
        PROPELLER_NUM: int = 4  # プロペラ4枚
        self.action_space = gymnasium.spaces.Box(
            low=np.array(
                [
                    MIN_ROTATION_V,
                    MIN_ROTATION_V,
                    MIN_ROTATION_V,
                    MIN_ROTATION_V,
                    MIN_ROTATION_V,
                ]
            ),
            high=np.array(
                [
                    MAX_ROTATION_V,
                    MAX_ROTATION_V,
                    MAX_ROTATION_V,
                    MAX_ROTATION_V,
                    MAX_ROTATION_V,
                ]
            ),
            shape=(PROPELLER_NUM + 1,),
        )

        # ------------------------------------------------------------------------
        # 状態空間
        # ------------------------------------------------------------------------
        POSITIVE_MAX: float = math.inf  # どこまでも行ける
        NEGATIVE_MAX: float = -math.inf  # 同上
        DIRECTIONS: int = 3  # x, y, z の3方向
        self.observation_space = gymnasium.spaces.Box(
            low=NEGATIVE_MAX,
            high=POSITIVE_MAX,
            # 推力(1), 回転速度 / 角度 / 位置座標 / 速度 4種類の x, y, z(12) + ゴールの x, y, z 座標(3) + ゴールからの距離(1)
            shape=(1 + DIRECTIONS * 4 + len(self.goal) + 1,),
        )
        # self.observation_space = gymnasium.spaces.Discrete(12)

        # ------------------------------------------------------------------------
        # 報酬関連パラメータ
        # ------------------------------------------------------------------------
        # 即時報酬の値は-10から1の間とした
        self.reward_range = (-10, 1)

        # 報酬計算時の重み
        self.weight_approach_reward: float = weight_approach_reward
        self.weight_force_penalty: float = weight_force_penalty

        self.debug_mode = debug_mode

        # 重力に逆らえないドローンを補助する力
        self.complement_force: bool = complement_force

        self.multiply_force: float = 1.0

        self.recorder: Optional[FlightRecorder] = recorder

    def reset(self, seed: int = 0) -> Tuple[State, dict]:
        if seed:
            """
            seed の固定を試みる
            https://qiita.com/north_redwing/items/1e153139125d37829d2d
            """
            np.random.seed(seed=seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.steps = 0

        # bug
        # if self.drone is None:
        initial_position: PositionCoordinates = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.drone = Drone(initial_position, gravity_enabled=self.gravity_enabled)

        # 重力に逆らえないドローンを補助する力
        if self.complement_force:
            self.multiply_force = self.drone.mass * self.drone.g

        if self.specified_goal is None:
            self.goal = tuple(np.random.randint(10, size=3))
        elif self.debug_mode:
            self.goal = (5.0, 5.0, 5.0)
        elif not self.debug_mode:
            while True:
                self.goal = tuple(np.random.randint(10, size=3))
                # [0, 0, 0] の位置にゴールを設定しない
                if self.goal != (0, 0, 0):
                    break

        self.traveling_distance = self._calc_distance_from_goal(0.0, 0.0, 0.0)

        self.state = State(
            u=0.0,
            rot_v_x=0.0,
            rot_v_y=0.0,
            rot_v_z=0.0,
            rot_x=0.0,
            rot_y=0.0,
            rot_z=0.0,
            v_x=0.0,
            v_y=0.0,
            v_z=0.0,
            pos_x=0.0,
            pos_y=0.0,
            pos_z=0.0,
            goal_x=self.goal[0],
            goal_y=self.goal[1],
            goal_z=self.goal[2],
            distance_from_goal=self._calc_distance_from_goal(0.0, 0.0, 0.0),
        )

        self.profit = 0.0

        info: dict = {}
        return self.state, info

    def _detect_incident(self, next_state: State) -> bool:
        incident: bool = False
        speed: float = self.drone.calc_integrated_speed()
        if (speed > 2.0) and (next_state.pos_z < 0.0):
            # 秒速 2.0 m 以上で地面に接触したら事故
            incident = True

        return incident
        # return False

    def _calc_distance_from_goal(self, x: float, y: float, z: float) -> float:
        """
        入力した位置座標とゴールとの間の距離を計算して返す
        """
        squared_sum_distance: float = (
            (self.goal[0] - x) ** 2  # x 成分
            + (self.goal[1] - y) ** 2  # y 成分
            + (self.goal[2] - z) ** 2  # z 成分
        )
        return math.sqrt(squared_sum_distance)

    def _detect_event(self, next_state: State) -> Optional[str]:
        event: Optional[str] = None

        # dist_from_goal: float = self._calc_distance_from_goal(
        #     next_state.pos_x, next_state.pos_y, next_state.pos_z
        # )
        if next_state.distance_from_goal < 0.5:
            event = "goal"
        elif self._detect_incident(next_state):
            event = "incident"
        elif self.steps == self._max_episode_steps:  # 200 step で終了
            event = "timeout"

        return event

    def _calc_distance_approach(self, state: State, next_state: State) -> float:
        """
        前ステップから今回ステップまでの間に、ゴールにどの程度近づいたかを計算して返却
        """
        distance_approach: float = (
            # 前ステップにおける、ゴールまでの距離
            self._calc_distance_from_goal(state.pos_x, state.pos_y, state.pos_z)
            # 今ステップ行動後の、ゴールまでの距離
            - self._calc_distance_from_goal(
                next_state.pos_x, next_state.pos_y, next_state.pos_z
            )
        )
        return distance_approach

    def _calc_integrated_angular_v(self, action: PropellerAngularVelocity) -> float:
        """
        1ステップの間にかけた力 (x, y, z) をベクトルの大きさとして統合した値を返す
        """
        integrated_force: float = math.sqrt(
            action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
        )
        return integrated_force

    def aviary_reward(self, next_state: State) -> float:
        # 報酬設計は utiasDSL/gym-pybullet-drones を参考にした
        # https://github.com/utiasDSL/gym-pybullet-drones/blob/5404871f32697b7c568d9e2520368e81d46f0ab3/gym_pybullet_drones/envs/HoverAviary.py#L68-L79
        distance_from_goal: float = self._calc_distance_from_goal(
            next_state.pos_x, next_state.pos_y, next_state.pos_z
        )
        reward_approach: float = max(
            0.0,
            (0.7 - (distance_from_goal / self.traveling_distance) ** 4),
        )
        return reward_approach

    def attitude_reward(self, next_state: State) -> float:
        """
        ドローンの姿勢に対する報酬を計算
        ドローンの姿勢の変化量（角速度）が小さいほど高い報酬を与える
        """
        # ドローンの姿勢の変化量（角速度）を計算
        attitude_change: float = (
            abs(next_state.rot_v_x) + abs(next_state.rot_v_y) + abs(next_state.rot_v_z)
        )
        # 姿勢の変化量が小さいほど高い報酬を与える
        reward_attitude_multiplier: float = max(1.0, 3.0 - attitude_change)
        return reward_attitude_multiplier

    def _calc_reward(
        self,
        event: Optional[str],
        state: State,
        action: PropellerAngularVelocity,
        next_state: State,
    ) -> float:
        """
        報酬を計算
        """
        assert event in [
            "incident",
            "goal",
            "timeout",
            None,
        ], f"inputted event is invalid: {event}"

        reward_approach: float = (
            self._calc_distance_approach(state, next_state)
            * self.weight_approach_reward
        )
        # reward_approach += self.aviary_reward(next_state)
        if reward_approach > 0.0:
            reward_approach *= self.attitude_reward(next_state)

        penalty_force_consumption: float = (
            self._calc_integrated_angular_v(action) * self.weight_force_penalty
        )
        reward: float = reward_approach - penalty_force_consumption
        reward_dic: Dict[Any, float] = {
            "incident": -100.0,
            "goal": 50.0,
            "timeout": reward,
            None: reward,
        }
        return reward_dic[event]

    def step(
        self, action: List[float]
    ) -> Tuple[State, float, Optional[bool], Optional[bool], Dict[str, Optional[str]]]:
        """
        現在の状態と行動から次の状態に遷移 (状態は self.drone クラスで管理されている)
        """

        ang_v_4_dim: tuple[float, float, float, float] = (
            action[0] * action[1] * self.multiply_force,
            action[0] * action[2] * self.multiply_force,
            action[0] * action[3] * self.multiply_force,
            action[0] * action[4] * self.multiply_force,
        )
        machine_state: MachineState = self.drone.flight(ang_v_4_dim)

        # TODO: 事故検出の前に、地面潜り込み補正をしているので、事故が見つからない
        # NOTE: 地面の下に潜り込むことはない
        if machine_state["position_z"] < 0.0:
            machine_state["position_z"] = 0.0
            if machine_state["velocity_z"] < 0.0:
                machine_state["velocity_z"] = 0.0

        next_state: State = State(
            u=machine_state["u"],
            rot_v_x=machine_state["rotation_velocity_x"],
            rot_v_y=machine_state["rotation_velocity_y"],
            rot_v_z=machine_state["rotation_velocity_z"],
            rot_x=machine_state["machine_rotation_x"],
            rot_y=machine_state["machine_rotation_y"],
            rot_z=machine_state["machine_rotation_z"],
            v_x=machine_state["velocity_x"],
            v_y=machine_state["velocity_y"],
            v_z=machine_state["velocity_z"],
            pos_x=machine_state["position_x"],
            pos_y=machine_state["position_y"],
            pos_z=machine_state["position_z"],
            goal_x=self.goal[0],
            goal_y=self.goal[1],
            goal_z=self.goal[2],
            # ゴールまでの距離
            distance_from_goal=self._calc_distance_from_goal(
                machine_state["position_x"],
                machine_state["position_y"],
                machine_state["position_z"],
            ),
        )

        event: Optional[str] = self._detect_event(next_state)
        done: bool = True if event is not None else False
        reward: float = self._calc_reward(event, self.state, ang_v_4_dim, next_state)
        info: Dict[str, Optional[str]] = {"event": event}

        # 経験の記録
        if self.recorder is not None:
            self.recorder.record_step(
                obs=self.state,
                action=action,
                re=reward,
                done=done,
                info=info,
            )

        # 行動後の後処理
        # if next_state[2] < 0.0:  # State[2] は z 軸方向の位置座標
        #     next_state[2] = 0.0
        self.state = next_state
        self.steps += 1

        # if self.debug_mode:
        #     print(f"Step[{self.steps}] reward:", reward)

        self.profit += reward
        # if done:
        #     print("profit___{}".format(self.profit), event)

        truncated: Optional[bool] = None

        return self.state, reward, done, truncated, info

    def _generate_numpy_array(self) -> np.ndarray:
        def to_numpy(fig: plt.Figure) -> np.ndarray:
            # Figure をレンダリングする。
            fig.canvas.draw()

            # 画像をバイト列で取得する。
            # data = fig.canvas.tostring_argb()  # 色がおかしくなる
            buffer = fig.canvas.buffer_rgba()

            # この時点では、画像の各ピクセルが1次元の配列として格納されている。
            one_line_img: np.ndarray = np.frombuffer(buffer, dtype=np.uint8)

            # 画像の大きさを取得する。
            w, h = fig.canvas.get_width_height()
            c = len(one_line_img) // (w * h)  # channel 数

            # numpy 配列に変換する
            img = one_line_img.reshape(h, w, c)

            return img

        fig: plt.Figure = plt.figure(figsize=(4, 4))
        df_pos = pd.DataFrame(self.drone.internal_record.position_coordinate_history)

        ax = fig.add_subplot(111, projection="3d")
        # OPTIMIZE: コメントインした方が描画は早くなる
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])

        x_candidates = pd.concat([df_pos["x"], pd.Series([self.goal[0]])])
        y_candidates = pd.concat([df_pos["y"], pd.Series([self.goal[1]])])

        x = np.linspace(x_candidates.min(), x_candidates.max(), 10)
        y = np.linspace(y_candidates.min(), y_candidates.max(), 10)
        ground_X, ground_Y = np.meshgrid(x, y)

        # 高さ0.0の平面を定義
        ground_Z = np.zeros_like(ground_X)

        # OPTIMIZE: コメントアウトした方が描画は早くなる
        ax.set_title(
            f"Flight trajectory step: {self.steps} / {self._max_episode_steps}"
        )

        trajectory_Z = np.full_like(df_pos["z"], 0.0)
        ax.plot_surface(ground_X, ground_Y, ground_Z, color="gray", alpha=0.5)
        ax.scatter(df_pos["x"], df_pos["y"], df_pos["z"], c="blue", label="trajectory")
        ax.scatter(
            df_pos["x"],
            df_pos["y"],
            trajectory_Z,
            c="darkblue",
            alpha=0.1,
            label="trajectory on ground",
        )
        ax.scatter(
            [self.goal[0]], [self.goal[1]], [self.goal[2]], c="orange", label="goal"
        )
        ax.scatter(
            [self.goal[0]],
            [self.goal[1]],
            [0.0],
            c="orange",
            alpha=0.5,
            label="goal on ground",
        )

        # OPTIMIZE: コメントアウトした方が描画は早くなる
        # ax.legend()

        numpy_img: np.ndarray = to_numpy(fig)
        plt.close()
        return numpy_img

    def render(self, mode: str = "rgb_array") -> None:
        if mode == "rgb_array":
            return self._generate_numpy_array()
        # elif mode == 'human':
        #     ...
        else:
            # super(MyEnv, self).render(mode=mode)
            raise ValueError(f"The valuable mode must be `{mode}`!")

    # def seed(self, seed: int = 0) -> None:
    #     """
    #     seed の固定を試みる
    #     https://qiita.com/north_redwing/items/1e153139125d37829d2d
    #     """
    #     np.random.seed(seed=seed)
    #     torch.manual_seed(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    def close(self) -> None:
        plt.close()
