"""Server pi0 (openpi) ho tro DemoDiffusion: khu nhieu tu buoc trung gian s* thay vi tu nhieu thuan.

Chay trong venv cua openpi (commit Arena dung):
    $WS/openpi/.venv/bin/python demodiffusion/openpi_server/serve_demodiffusion.py --port 8000

Giao thuc: nhu server openpi chuan (client openpi_client), them 2 khoa tuy chon trong observation:
    "retargeted_actions": (H, 8) hanh dong retarget tu demo nguoi, CUNG khong gian voi dau ra cua policy
                          (7 vi tri khop tuyet doi + gripper 0..1) - server tu doi sang delta + chuan hoa
    "time":               s* / S trong [0, 1]; 1 = pi0 goc (nhieu thuan), 0 = tra lai quy dao retarget
Khoi tao (DemoDiffusion, eq. 2):  x_{s*} = s* * eps + (1 - s*) * a_retarget  (a_retarget da chuan hoa, pad 32 chieu)
roi tich phan flow cua pi0 tu t = s* ve 0 voi buoc dt = -1/num_steps (20 buoc cho ca qua trinh nhu bai bao).
Khong sua ma nguon openpi: Pi0.sample_actions duoc va luc chay de nhan them start_time.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import textwrap

import jax
import jax.numpy as jnp
import numpy as np


def patch_pi0_start_time():
    """Them tham so start_time (mac dinh 1.0) vao Pi0.sample_actions - giu nguyen moi thu khac."""
    from openpi.models import pi0 as pi0_mod

    src = textwrap.dedent(inspect.getsource(pi0_mod.Pi0.sample_actions))
    assert "(noise, 1.0)" in src, "openpi doi ma sample_actions - can cap nhat ban va"
    src = src.replace("noise: at.Float[at.Array, \"b ah ad\"] | None = None,",
                      "noise: at.Float[at.Array, \"b ah ad\"] | None = None,\n        start_time: float = 1.0,", 1)
    assert "start_time: float = 1.0" in src, "khong chen duoc tham so start_time"
    src = src.replace("(noise, 1.0)", "(noise, jnp.asarray(start_time, dtype=jnp.float32))")
    src = src.replace("@override\n", "")
    ns = dict(vars(pi0_mod))
    exec(compile(src, pi0_mod.__file__, "exec"), ns)  # noqa: S102
    pi0_mod.Pi0.sample_actions = ns["sample_actions"]
    logging.info("Da va Pi0.sample_actions: them start_time")


class DemoDiffusionPolicy:
    """Boc openpi Policy: neu co 'retargeted_actions' + 'time' < 1 thi khu nhieu tu s*."""

    def __init__(self, policy, num_steps: int = 20, seed: int = 0):
        self._p = policy
        self._num_steps = num_steps
        self._rng = jax.random.key(seed)

    @property
    def metadata(self):
        return self._p.metadata

    def infer(self, obs: dict) -> dict:
        obs = dict(obs)
        retarget = obs.pop("retargeted_actions", None)
        s = float(np.clip(obs.pop("time", 1.0), 0.0, 1.0))
        p = self._p
        inputs = dict(obs)
        if retarget is not None:
            inputs["actions"] = np.asarray(retarget, np.float32)
        inputs = p._input_transform(inputs)  # noqa: SLF001  (DroidInputs -> DeltaActions -> Normalize -> Pad)
        a_norm = inputs.pop("actions", None)
        batch = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, rng_sample, rng_noise = jax.random.split(self._rng, 3)
        model = p._model  # noqa: SLF001
        shape = (1, model.action_horizon, model.action_dim)
        noise = jax.random.normal(rng_noise, shape)
        if retarget is not None and s < 1.0:
            a = jnp.asarray(a_norm)[np.newaxis, ...]
            assert a.shape == shape, f"retargeted_actions sau bien doi {a.shape} != {shape}"
            x_init = s * noise + (1.0 - s) * a
            t0 = s
        else:
            x_init, t0 = noise, 1.0
        from openpi.models import model as _model

        observation = _model.Observation.from_dict(batch)
        actions = p._sample_actions(rng_sample, observation, noise=x_init, start_time=t0, num_steps=self._num_steps)  # noqa: SLF001
        outputs = {"state": batch["state"][0], "actions": np.asarray(actions[0])}
        outputs = p._output_transform(outputs)  # noqa: SLF001  (Unnormalize -> AbsoluteActions -> DroidOutputs)
        outputs["demodiffusion"] = {"time": t0, "denoise_steps": int(round(t0 * self._num_steps))}
        return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi0_droid_jointpos_polaris")
    ap.add_argument("--checkpoint", default="gs://openpi-assets-simeval/pi0_droid_jointpos")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-steps", type=int, default=20, help="so buoc khu nhieu cho ca qua trinh 1 -> 0 (bai bao: 20)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    patch_pi0_start_time()
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    base = policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
    policy = DemoDiffusionPolicy(base, num_steps=args.num_steps)
    logging.info("Server DemoDiffusion: %s, %d buoc khu nhieu, cong %d", args.config, args.num_steps, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=base.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
