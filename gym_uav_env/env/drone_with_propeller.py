# TODO: 空気抵抗の計算
# TODO: Action を10回に1回にする
import math
from typing import List, Tuple, TypedDict

import numpy as np


class PositionCoordinates(TypedDict):
    """
    機体の位置座標を表す辞書
    """

    x: float
    y: float
    z: float


class Velocity(TypedDict):
    """
    機体の速度を表す辞書
    """

    x: float
    y: float
    z: float


class MachineState(TypedDict):
    """
    機体の状態を表す辞書
    """

    u: float
    rotation_velocity_x: float
    rotation_velocity_y: float
    rotation_velocity_z: float
    machine_rotation_x: float
    machine_rotation_y: float
    machine_rotation_z: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    position_x: float
    position_y: float
    position_z: float


# class Force(TypedDict):
#     """
#     3次元空間に生じる力を表す辞書
#     """

#     x: float
#     y: float
#     z: float


class InternalRecord:
    def __init__(self) -> None:
        self.position_coordinate_history: List[PositionCoordinates] = []
        self.velocity_history: List[Velocity] = []

    def get_latest_2_positions(self) -> List[PositionCoordinates]:
        return self.position_coordinate_history[-2:]


class Drone:
    g: float = 9.8
    k_air_resistance: float = 0.4

    def __init__(
        self,
        position_coordinates: PositionCoordinates,  # 機体の位置 (m)
        mass: float = 4.0,
        time_unit: float = 0.01,  # これ以上小さくしても動作に対して影響ない
        debug_mode: bool = False,
        gravity_enabled: bool = True,
    ) -> None:
        # 単位時間 (second)
        self.t: float = time_unit

        # 機体重量 (kg)
        self.mass: float = mass

        # 機体サイズ (m, 未対応)
        # self.size: Dict[str, float] = {"length": 0.5, "width": 0.5}

        # 進行速度 (m/s)
        velocity: Velocity = {"x": 0.0, "y": 0.0, "z": 0.0}

        self.state: MachineState = self._init_machine_state()

        self.debug_mode = debug_mode
        self.internal_record: InternalRecord = InternalRecord()
        self.internal_record.position_coordinate_history.append(position_coordinates)
        self.internal_record.velocity_history.append(velocity)

        self.gravity_enabled: bool = gravity_enabled
        if self.gravity_enabled is False:
            Drone.g = 0.0
            print("Gravity is disabled. :", Drone.g)
        else:
            Drone.g = 9.8

        # ------------------------------------------------------------
        # プロペラや機体の回転を考慮する際の定数係数
        # ------------------------------------------------------------
        # ロー: プロペラの回転にかかる係数で、プロペラの形状によって値が変わる
        #    摩耗したり、破損すれば、当然大きく値が変わってくるし、
        #    (望ましくはないが、)プロペラによって値が異なることもありうる
        self.PROPELLER_COEFFICIENT: float = 0.5
        # l: 機体の x, y 軸方向の回転(roll, pitch)に影響を与える係数
        self.MACHINE_ROTATION_COEFFCIENT: float = 0.5
        # kappa: プロペラが回転する際に空気を掻くことにより、ヨー yaw 角に及ぼす影響度合いを表す係数
        self.PROPELLER_SCOOP_COEFFICIENT: float = 0.5

        if self.debug_mode:
            print("Acceleration rate of gravity:", Drone.g)

    def _init_machine_state(self) -> MachineState:
        u = 0
        # rotation_velocity_x: float = 1.0
        rotation_velocity_x: float = 0.0
        machine_rotation_x: float = np.pi * 0.0
        # rotation_velocity_y: float = 1.0
        rotation_velocity_y: float = 0.0
        machine_rotation_y: float = np.pi * 0.0
        rotation_velocity_z: float = 0.0
        machine_rotation_z: float = np.pi * 0.0

        velocity_x = 0
        position_x = 0
        velocity_y = 0
        position_y = 0
        velocity_z = 0
        position_z = 0

        initial_state: MachineState = {
            "u": u,
            "rotation_velocity_x": rotation_velocity_x,
            "rotation_velocity_y": rotation_velocity_y,
            "rotation_velocity_z": rotation_velocity_z,
            "machine_rotation_x": machine_rotation_x,
            "machine_rotation_y": machine_rotation_y,
            "machine_rotation_z": machine_rotation_z,
            "velocity_x": velocity_x,
            "position_x": position_x,
            "velocity_y": velocity_y,
            "position_y": position_y,
            "velocity_z": velocity_z,
            "position_z": position_z,
        }
        return initial_state

    def calc_thrust(self, omegas: np.ndarray) -> float:
        """
        推力を計算する
        4本のプロペラの回転する角速度から、ドローンにかかる推力を計算

        Parameters
        ------
        omegas: np.ndarray ([float, float, float, float])
            ドローンのプロペラ1番から4番まで、それぞれの角速度
        """
        thrust: float = (
            self.PROPELLER_COEFFICIENT * (omegas[0] * omegas[0])
            + self.PROPELLER_COEFFICIENT * (omegas[1] * omegas[1])
            + self.PROPELLER_COEFFICIENT * (omegas[2] * omegas[2])
            + self.PROPELLER_COEFFICIENT * (omegas[3] * omegas[3])
        )
        return thrust

    def calc_rotation_velocity_x(
        self, current_rotation_velocity_x: float, omegas: np.ndarray
    ) -> float:
        """
        4本のプロペラの回転する角速度から、ドローン機体の x 軸方向の角度 roll （φ） を計算

        Parameters
        ------
        omegas: np.ndarray ([float, float, float, float])
            ドローンのプロペラ1番から4番まで、それぞれの角速度
        """
        torque_phi: float = (
            -self.MACHINE_ROTATION_COEFFCIENT
            * self.PROPELLER_COEFFICIENT
            * omegas[1]
            * omegas[1]
            + self.MACHINE_ROTATION_COEFFCIENT
            * self.PROPELLER_COEFFICIENT
            * omegas[3]
            * omegas[3]
        )
        rotation_velocity_x: float = torque_phi * self.t + current_rotation_velocity_x
        return rotation_velocity_x

    def calc_rotation_velocity_y(
        self, current_rotation_velocity_y: float, omegas: np.ndarray
    ) -> float:
        """
        4本のプロペラの回転する角速度から、ドローン機体の y 軸方向の角度 pitch （θ） を計算

        Parameters
        ------
        omegas: np.ndarray ([float, float, float, float])
            ドローンのプロペラ1番から4番まで、それぞれの角速度
        """
        torque_theta: float = (
            -self.MACHINE_ROTATION_COEFFCIENT
            * self.PROPELLER_COEFFICIENT
            * omegas[0]
            * omegas[0]
            + self.MACHINE_ROTATION_COEFFCIENT
            * self.PROPELLER_COEFFICIENT
            * omegas[2]
            * omegas[2]
        )
        rotation_velocity_y: float = torque_theta * self.t + current_rotation_velocity_y
        return rotation_velocity_y

    def calc_rotation_velocity_z(
        self, current_rotation_velocity_z: float, omegas: np.ndarray
    ) -> float:
        """
        4本のプロペラの回転する角速度から、ドローン機体の z 軸方向の角度 yaw （ψ） を計算

        Parameters
        ------
        omegas: np.ndarray ([float, float, float, float])
            ドローンのプロペラ1番から4番まで、それぞれの角速度
        """
        torque_psi: float = (
            self.PROPELLER_SCOOP_COEFFICIENT * omegas[0] * omegas[0]
            - self.PROPELLER_SCOOP_COEFFICIENT * omegas[1] * omegas[1]
            + self.PROPELLER_SCOOP_COEFFICIENT * omegas[2] * omegas[2]
            - self.PROPELLER_SCOOP_COEFFICIENT * omegas[3] * omegas[3]
        )
        rotation_velocity_z: float = torque_psi * self.t + current_rotation_velocity_z
        return rotation_velocity_z

    def calc_next_velocity_posi_x(
        self,
        velocity_x: float,
        position_x: float,
        u: float,
        machine_rotation_x: float,
        machine_rotation_y: float,
        machine_rotation_z: float,
    ) -> Tuple[float, float]:
        """
        x 軸方向の速度と位置を計算
        """
        next_velocity_x: float = (self.t / self.mass) * u * (
            np.cos(machine_rotation_x)
            * np.sin(machine_rotation_y)
            * np.cos(machine_rotation_z)
            - np.sin(machine_rotation_x) * np.sin(machine_rotation_z)
        ) + velocity_x

        next_position_x: float = (velocity_x * self.t) + position_x
        return next_velocity_x, next_position_x

    def calc_next_velocity_posi_y(
        self,
        velocity_y: float,
        position_y: float,
        u: float,
        machine_rotation_x: float,
        machine_rotation_y: float,
        machine_rotation_z: float,
    ) -> Tuple[float, float]:
        """
        y 軸方向の速度と位置を計算
        """
        next_velocity_y: float = (self.t / self.mass) * u * (
            np.cos(machine_rotation_x)
            * np.sin(machine_rotation_y)
            * np.sin(machine_rotation_z)
            - np.sin(machine_rotation_x) * np.cos(machine_rotation_z)
        ) + velocity_y

        next_position_y: float = (velocity_y * self.t) + position_y
        return next_velocity_y, next_position_y

    def calc_next_velocity_posi_z(
        self,
        velocity_z: float,
        position_z: float,
        u: float,
        machine_rotation_x: float,
        machine_rotation_y: float,
    ) -> Tuple[float, float]:
        """
        z 軸方向の速度と位置を計算
        """
        next_velocity_z = (
            (self.t / self.mass)
            * u
            * (np.cos(machine_rotation_y) * np.cos(machine_rotation_x))
            - (Drone.g * self.t)
            + velocity_z
        )

        next_position_z = (velocity_z * self.t) + position_z
        # /*
        #  * 境界条件：地面から下には落ちない
        #  */
        if next_position_z < 0:
            next_position_z = 0
            next_velocity_z = 0
        return next_velocity_z, next_position_z

    def flight(self, propeller_angular_velocities: np.ndarray) -> MachineState:
        # ------------------------------------------------------------
        # 機体状態の取得
        # ------------------------------------------------------------
        u: float = self.calc_thrust(propeller_angular_velocities)

        rotation_velocity_x = self.state["rotation_velocity_x"]
        rotation_velocity_y = self.state["rotation_velocity_y"]
        rotation_velocity_z = self.state["rotation_velocity_z"]
        machine_rotation_x = self.state["machine_rotation_x"]
        machine_rotation_y = self.state["machine_rotation_y"]
        machine_rotation_z = self.state["machine_rotation_z"]
        velocity_x = self.state["velocity_x"]
        velocity_y = self.state["velocity_y"]
        velocity_z = self.state["velocity_z"]
        position_x = self.state["position_x"]
        position_y = self.state["position_y"]
        position_z = self.state["position_z"]

        # ------------------------------------------------------------
        # 機体の回転関連の計算
        # ------------------------------------------------------------
        rotation_velocity_x = self.calc_rotation_velocity_x(
            rotation_velocity_x, propeller_angular_velocities
        )
        machine_rotation_x = ((rotation_velocity_x * self.t) + machine_rotation_x) % (
            2 * np.pi
        )
        rotation_velocity_y = self.calc_rotation_velocity_y(
            rotation_velocity_y, propeller_angular_velocities
        )
        machine_rotation_y = ((rotation_velocity_y * self.t) + machine_rotation_y) % (
            2 * np.pi
        )
        rotation_velocity_z = self.calc_rotation_velocity_z(
            rotation_velocity_z, propeller_angular_velocities
        )
        machine_rotation_z = ((rotation_velocity_z * self.t) + machine_rotation_z) % (
            2 * np.pi
        )

        # ------------------------------------------------------------
        # 機体の速度と位置の計算
        # ------------------------------------------------------------
        velocity_x, position_x = self.calc_next_velocity_posi_x(
            velocity_x,
            position_x,
            u,
            machine_rotation_x,
            machine_rotation_y,
            machine_rotation_z,
        )
        velocity_y, position_y = self.calc_next_velocity_posi_y(
            velocity_y,
            position_y,
            u,
            machine_rotation_x,
            machine_rotation_y,
            machine_rotation_z,
        )
        velocity_z, position_z = self.calc_next_velocity_posi_z(
            velocity_z, position_z, u, machine_rotation_x, machine_rotation_y
        )

        # ------------------------------------------------------------
        # 機体の位置と速度の記録
        # ------------------------------------------------------------
        self.internal_record.position_coordinate_history.append(
            {"x": position_x, "y": position_y, "z": position_z}
        )
        self.internal_record.velocity_history.append(
            {"x": velocity_x, "y": velocity_y, "z": velocity_z}
        )

        next_machine_state: MachineState = {
            "u": u,  # TODO: 状態としてほとんど価値はないが、他の処理との整合性のために残しておく
            "rotation_velocity_x": rotation_velocity_x,
            "rotation_velocity_y": rotation_velocity_y,
            "rotation_velocity_z": rotation_velocity_z,
            "machine_rotation_x": machine_rotation_x,
            "machine_rotation_y": machine_rotation_y,
            "machine_rotation_z": machine_rotation_z,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "velocity_z": velocity_z,
            "position_x": position_x,
            "position_y": position_y,
            "position_z": position_z,
        }
        self.state = next_machine_state
        return next_machine_state

    def calc_integrated_speed(self) -> float:
        velocity = self.internal_record.velocity_history[-1]
        return math.sqrt(velocity["x"] ** 2 + velocity["y"] ** 2 + velocity["z"] ** 2)


# if __name__ == "__main__":
#     drone_instance: Drone = Drone({"x": 0.0, "y": 0.0, "z": 0.0})
