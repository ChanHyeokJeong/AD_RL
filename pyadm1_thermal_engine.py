from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import scipy.integrate

from config import RLConfig


ADM_STATE_NAMES = [
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_IC",
    "S_IN",
    "S_I",
    "X_xc",
    "X_ch",
    "X_pr",
    "X_li",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "X_I",
    "S_cation",
    "S_anion",
    "S_H_ion",
    "S_va_ion",
    "S_bu_ion",
    "S_pro_ion",
    "S_ac_ion",
    "S_hco3_ion",
    "S_co2",
    "S_nh3",
    "S_nh4_ion",
    "S_gas_h2",
    "S_gas_ch4",
    "S_gas_co2",
]

INFLUENT_STATE_NAMES = [
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_IC",
    "S_IN",
    "S_I",
    "X_xc",
    "X_ch",
    "X_pr",
    "X_li",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "X_I",
    "S_cation",
    "S_anion",
]


@dataclass
class EngineStepResult:
    time_d: float
    state_vector: np.ndarray
    diagnostics: dict[str, float]


class PyADM1ThermalEngine:
    """Adapter that calls the original monolithic PyADM1_thermal ODE by interval.

    The source file is intentionally not imported as a Python module because the
    bottom of PyADM1_thermal.py runs a whole simulation at import time. Instead,
    this engine executes only the definition block before "## time array
    definition", then drives ADM1_ODE, DAESolve, and thermal_terms one interval
    at a time from the Gym plant.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.source_path = Path(config.full_pyadm1_source_path)
        self.input_dir = Path(config.full_pyadm1_engine_input_dir)
        self.solver_method = config.full_pyadm1_solver_method
        self.rtol = float(config.full_pyadm1_rtol)
        self.atol = float(config.full_pyadm1_atol)
        self.ns: dict[str, Any] = {}
        self.time_grid: np.ndarray = np.array([], dtype=float)
        self.state_vector = np.zeros(len(ADM_STATE_NAMES) + 2, dtype=float)
        self.last_diagnostics: dict[str, float] = {}
        self.current_q_ch4_heater_m3_d = 0.0
        self.current_time_d = 0.0

    @property
    def state_names(self) -> list[str]:
        return ADM_STATE_NAMES + ["T_reactor", "T_adapt"]

    @property
    def adm_state_names(self) -> list[str]:
        return ADM_STATE_NAMES.copy()

    @property
    def namespace_view(self) -> MappingProxyType:
        return MappingProxyType(self.ns)

    def reset(self, start_day: float = 0.0) -> EngineStepResult:
        self._prepare_engine_inputs()
        self._load_source_prefix()
        self.time_grid = self.ns["influent_state"]["time"].to_numpy(dtype=float)
        self.state_vector = np.asarray(self.ns["state_zero"], dtype=float).copy()
        self.current_time_d = float(start_day)
        self.current_q_ch4_heater_m3_d = float(self.ns.get("q_ch4_heater_base", 0.0))
        self._install_heater_override()
        self._set_influent_for_time(self.current_time_d)
        self._assign_state_to_namespace(self.state_vector)
        self._postprocess_algebraic(self.current_time_d)
        return EngineStepResult(
            time_d=self.current_time_d,
            state_vector=self.state_vector.copy(),
            diagnostics=self.last_diagnostics.copy(),
        )

    def influent_at(self, time_d: float) -> tuple[float, float]:
        idx = self._row_index(time_d)
        row = self.ns["influent_state"].iloc[idx]
        return float(row["T_in"]), float(row["Q"])

    def open_loop_ch4_flow(self, target_C: float, time_d: float) -> float:
        t_in_C, q_flow = self.influent_at(time_d)
        return float(
            self.ns["calculate_open_loop_ch4_flow"](
                target_C + 273.15,
                t_in_C + 273.15,
                q_flow,
            )
        )

    def adm_state_values(self) -> np.ndarray:
        return self.state_vector[: len(ADM_STATE_NAMES)].astype(float).copy()

    def reactor_temperature_C(self) -> float:
        return float(self.state_vector[-2] - 273.15)

    def adapted_temperature_C(self) -> float:
        return float(self.state_vector[-1] - 273.15)

    def methanogenesis_shock_factor(self) -> float:
        return float(self.last_diagnostics.get("methanogenesis_shock_factor", 1.0))

    def methanogenesis_temp_mismatch_K(self) -> float:
        return float(self.last_diagnostics.get("methanogenesis_temp_mismatch_K", 0.0))

    def step(
        self,
        t0_d: float,
        t1_d: float,
        q_ch4_heater_m3_d: float,
    ) -> EngineStepResult:
        self.current_q_ch4_heater_m3_d = float(q_ch4_heater_m3_d)
        self._install_heater_override()
        self._set_influent_for_time(t0_d)
        self.ns["state_zero"] = self.state_vector.astype(float).tolist()

        result = scipy.integrate.solve_ivp(
            self.ns["ADM1_ODE"],
            (float(t0_d), float(t1_d)),
            self.state_vector.astype(float),
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol,
        )
        if not result.success:
            raise RuntimeError(f"PyADM1 solve_ivp failed: {result.message}")

        self.state_vector = np.asarray(result.y[:, -1], dtype=float)
        self.current_time_d = float(t1_d)
        self._assign_state_to_namespace(self.state_vector)
        self._postprocess_algebraic(float(t1_d))
        return EngineStepResult(
            time_d=self.current_time_d,
            state_vector=self.state_vector.copy(),
            diagnostics=self.last_diagnostics.copy(),
        )

    def _prepare_engine_inputs(self) -> None:
        if not self.source_path.exists():
            raise FileNotFoundError(f"Missing PyADM1 thermal source: {self.source_path}")
        self.input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config.influent_path, self.input_dir / "digester_influent.csv")
        shutil.copy2(self.config.initial_state_path, self.input_dir / "digester_initial.csv")

    def _load_source_prefix(self) -> None:
        source = self.source_path.read_text(encoding="utf-8")
        marker = "## time array definition"
        if marker not in source:
            raise ValueError(f"Could not find source split marker {marker!r}")
        prefix = source.split(marker, 1)[0]
        fake_file = self.input_dir / "PyADM1_thermal.py"
        ns: dict[str, Any] = {"__file__": str(fake_file), "__name__": "pyadm1_thermal_engine_source"}
        exec(compile(prefix, str(self.source_path), "exec"), ns)
        ns["use_methanogenesis_temp_shock"] = bool(
            self.config.use_methanogenesis_temp_shock
        )
        self.ns = ns

    def _install_heater_override(self) -> None:
        # ADDED: use the Gym/PI heater MV instead of the source file's
        # open-loop step schedule.
        # Reason: RL action changes T_setpoint; the PI controller computes
        # q_ch4_heater, and ADM1_ODE reads it through thermal_terms().
        # Role: keep original thermal_terms and ADM1_ODE intact while replacing
        # only the manipulated variable provider.
        # Reference: user-requested methane flow as MV inside Gym step.
        self.ns["q_ch4_heater_at"] = lambda _time_d: float(self.current_q_ch4_heater_m3_d)

    def _set_influent_for_time(self, time_d: float) -> None:
        idx = self._row_index(time_d)
        self.ns["setInfluent"](idx)
        self.ns["state_input"] = [
            float(self.ns[f"{name}_in"]) for name in INFLUENT_STATE_NAMES
        ] + [float(self.ns["T_in"])]

    def _row_index(self, time_d: float) -> int:
        if self.time_grid.size == 0:
            return 0
        max_time = float(self.time_grid[-1])
        wrapped = float(time_d if time_d <= max_time else time_d % max_time)
        idx = int(np.searchsorted(self.time_grid, wrapped, side="right") - 1)
        return int(np.clip(idx, 0, len(self.time_grid) - 1))

    def _assign_state_to_namespace(self, state: np.ndarray) -> None:
        for name, value in zip(ADM_STATE_NAMES, state[: len(ADM_STATE_NAMES)]):
            self.ns[name] = float(value)
        self.ns["T_reactor"] = float(state[-2])
        self.ns["T_adapt"] = float(state[-1])
        self.ns["T_op"] = float(state[-2])
        self.ns["T_ad"] = float(state[-2])

    def _state_from_namespace(self) -> np.ndarray:
        values = [float(self.ns[name]) for name in ADM_STATE_NAMES]
        values.append(float(self.ns["T_reactor"]))
        values.append(float(self.ns.get("T_adapt", self.ns["T_reactor"])))
        return np.asarray(values, dtype=float)

    def _postprocess_algebraic(self, time_d: float) -> None:
        self.ns["T_op"] = float(self.ns["T_reactor"])
        self.ns["T_ad"] = float(self.ns["T_reactor"])
        self.ns["DAESolve"]()

        t_reactor = float(self.ns["T_reactor"])
        p_gas_h2 = float(self.ns["S_gas_h2"]) * self.ns["R"] * t_reactor / 16.0
        p_gas_ch4 = float(self.ns["S_gas_ch4"]) * self.ns["R"] * t_reactor / 64.0
        p_gas_co2 = float(self.ns["S_gas_co2"]) * self.ns["R"] * t_reactor
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + float(self.ns["p_gas_h2o"])
        q_gas = max(0.0, float(self.ns["k_p"]) * (p_gas - float(self.ns["p_atm"])))
        q_ch4 = max(0.0, q_gas * p_gas_ch4 / p_gas) if p_gas > 0.0 else 0.0

        self.ns["p_gas_h2"] = p_gas_h2
        self.ns["p_gas_ch4"] = p_gas_ch4
        self.ns["p_gas_co2"] = p_gas_co2
        self.ns["p_gas"] = p_gas
        self.ns["q_gas"] = q_gas
        self.ns["q_ch4"] = q_ch4
        self.ns["S_nh4_ion"] = float(self.ns["S_IN"]) - float(self.ns["S_nh3"])
        self.ns["S_co2"] = float(self.ns["S_IC"]) - float(self.ns["S_hco3_ion"])

        heat = self.ns["thermal_terms"](
            float(self.ns["T_reactor"]),
            float(self.ns["T_in"]),
            float(time_d),
            float(self.ns["q_ad"]),
            q_ch4,
        )
        t_adapt = float(self.ns.get("T_adapt", t_reactor))
        shock_factor = float(
            self.ns["methanogenesis_temp_shock_factor"](t_reactor, t_adapt)
        )
        self.state_vector = self._state_from_namespace()
        self.ns["state_zero"] = self.state_vector.astype(float).tolist()
        self.last_diagnostics = {
            "time_d": float(time_d),
            "T_reactor_C": float(self.ns["T_reactor"] - 273.15),
            "T_adapt_C": float(t_adapt - 273.15),
            "methanogenesis_shock_factor": shock_factor,
            "methanogenesis_temp_mismatch_K": float(t_reactor - t_adapt),
            "T_in_C": float(self.ns["T_in"] - 273.15),
            "Q_m3_d": float(self.ns["q_ad"]),
            "pH": float(self.ns["pH"]),
            "q_gas_m3_d": q_gas,
            "q_ch4_prod_m3_d": q_ch4,
            "q_ch4_heater_m3_d": float(heat["q_ch4_heater"]),
            "feed_heat_MJ_d": float(heat["feed_heat"]),
            "wall_heat_loss_MJ_d": float(heat["wall_heat_loss"]),
            "heater_heat_MJ_d": float(heat["heater_heat"]),
        }
