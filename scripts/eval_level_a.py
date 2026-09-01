#!/usr/bin/env python3
"""Level A+ — independent MuJoCo adjudication of the honest Coulomb transport
claim (post honest-rebuild).

The gate (h) claim — the physics-aware NCA's margin-contracted transport steps
hold the object on hard cells while the blind baseline's ~52 mm steps slip it
out — is adjudicated by MuJoCo 3.11's rigid-body contact solver, independent of
the analytic Coulomb model (soma/physics.py). The exact closed-loop EEF motion
is replayed against a two-finger grip whose pads apply the commanded grip force
F_n = F_max·(1−g); MuJoCo decides held vs slipped.

Model — the pseudo-force / co-moving-frame grip. The hand is accelerating, so we
work in its co-moving frame: two FIXED world-geom pads squeeze the object and
the hand's motion enters as an inertial pseudo-force — the object feels
effective gravity g_eff = g − Ḧ. The hand's world acceleration is reconstructed
from the sim's first-order plant (state += plant_f·(target−state)):
  s(t) = target + (state − target)·(1 − plant_f)^(t/τ)
  a(t) = Ḧ(t) = −k²·(target − state)·(1 − plant_f)^(t/τ),  k = −ln(1−plant_f)/τ,
so each substep sets opt.gravity = (k²·Δx, k²·Δy, −g0 + k²·Δz)·(1−plant_f)^u
(= g − Ḧ). Fixed world geoms are essential: MuJoCo's solver holds static
friction cleanly against world geoms (~0.1 mm/s creep) but a two-pad force-servo
grip on dynamic bodies creeps ~20 mm/s. Galilean equivalence makes the
co-moving-frame slip onset identical to the real moving gripper; only the
absolute motion (irrelevant to slip) is dropped.

Honest-rebuild regime (τ = 0.05, k = 13.86 s⁻¹, k² ≈ 192):
  * The peak plant pseudo-acceleration is k²·Δx ≈ 192·Δx — the honest value (the
    old model's a = Δx/τ² = 100·Δx over-charged ~2× and its scalar load made the
    blind's sim-drops a parameterization artifact).
  * Worst-case orientation (worst-case yaw): the abstract sim has no pad
    orientation, so it adopts the conservative bound that the commanded
    horizontal acceleration is fully pad-TANGENTIAL (entirely friction-loaded).
    The replay implements this by yaw-rotating the recorded trajectory 90°
    about z so the horizontal motion is pad-tangential (y); a motion along the
    pad normal (x) would be borne by the contact normal — capture — and never
    slip. The blind baseline's ~52 mm steps then stress friction past capacity
    on hard cells (m/μ ≳ 1.16) while the aware's contracted ~25 mm steps stay
    below it — a real, MuJoCo-reproducible band.

Two modes:
  --calib   single-step slip-onset grid: the honest risk (m/μ)·sqrt(g0²+(k²Δx)²)
            /F_max vs MuJoCo-measured single-step slip for pad-tangential (y)
            and pad-normal (x) steps, plus steps-to-drop accumulation.
  default   transport replay: closed-loop transport trajectories (aware + blind
            on the discriminating band) replayed in MuJoCo under the worst-case
            yaw; per-cell sim verdict vs MuJoCo verdict, and the
            discriminating-cell reproduction count.

Run from the repo root:
    python scripts/eval_level_a.py --calib
    python scripts/eval_level_a.py --n-cells 20 --aware-dir checkpoints_phys_honest
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import mujoco

from soma.bodygraph_controller import BodyGraphController
from soma.moe_router import StateRouter
from soma.bodygraph_nca import BodyGraphNCA
from soma.physics import F_MAX, G0, TAU, EPS
from sim_env import SimEnv
import proc_sim
from eval_implicit_planning import cells, margin, load_experts

PLANT_F = 0.5
DT = 0.002
TAU_STEPS = int(TAU / DT)        # 25 substeps per τ-second plant step (τ=0.05)
K = -np.log(1.0 - PLANT_F) / TAU  # 13.86 s⁻¹ (τ=0.05)
YAW90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
SETTLE_S = 1.0
OBJ_HALF = 0.025                 # m, object box half-extent (0.05³ cube)
# Drop thresholds (calibrated against held-episode jitter ~0.1–0.5 mm): the
# object has slid out of the grip when its center drops below the held frame
# (vertical, gravity-driven) or total slip from the held frame exceeds the
# pad's lateral coverage (horizontal, inertial).
DROP_Z = 0.015                   # m, object center slid this far down → dropped
DROP_SLIP = 0.015                # m, total rel displacement from held frame


# ── pad-gap calibration ──────────────────────────────────────────────────
# MuJoCo's friction capacity is mu_pad·F_n_total (elliptic cone), so to match
# the analytic μ·F_max·(1−g) the two pads must settle at F_n_total = 2·F_n.
# Soft box contacts settle far above the old `pen = F_n/65e3` guess (2.5×),
# so the pad gap is binary-searched empirically per (mass, Fn). μ drops out
# (the normal force is geometry-only; the contact count/solver force depends
# on the squeeze, not the friction coefficient).
_CALIB_CACHE = {}


def _fn_total(inner_off, mass, mu_pad):
    """Two-pad model at inner_off: settle 1 s and return F_n_total (N)."""
    wallx = inner_off + 0.005
    xml = f"""
<mujoco model="level_a_calib">
  <option gravity="0 0 -9.81" timestep="{DT}" integrator="implicitfast"
          cone="elliptic" iterations="200"/>
  <size nconmax="64" njmax="128"/>
  <default>
    <geom type="box" solref="0.001 2" solimp="0.9 0.95 0.001"
          friction="{mu_pad} 0.9 0.001"/>
  </default>
  <worldbody>
    <geom name="wallL" type="box" pos="-{wallx} 0 0.10" size="0.005 0.5 0.15"
          friction="{mu_pad} 0.9 0.001"/>
    <geom name="wallR" type="box" pos="{wallx} 0 0.10" size="0.005 0.5 0.15"
          friction="{mu_pad} 0.9 0.001"/>
    <body name="object" pos="0 0 0.10">
      <freejoint/>
      <geom name="obj" type="box" size="{OBJ_HALF} {OBJ_HALF} {OBJ_HALF}"
            mass="{mass}" friction="{mu_pad} 0.9 0.001"/>
    </body>
  </worldbody>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(int(SETTLE_S / DT)):
        mujoco.mj_step(m, d)
    tot = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < 1e-3:
            cf = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, cf)
            tot += cf[0]
    return tot


def _calibrate_inner(mass, Fn, mu_pad):
    """inner_off with settled F_n_total == 2·Fn. Cached per (mass, Fn)."""
    key = (round(float(mass), 4), round(float(Fn), 2))
    if key in _CALIB_CACHE:
        return _CALIB_CACHE[key]
    target = 2.0 * Fn
    lo, hi = OBJ_HALF - 5e-4, OBJ_HALF + 1e-5   # tight (pen 5e-4) … loose (no touch)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        if _fn_total(mid, mass, mu_pad) > target:
            lo = mid                            # too much force → loosen
        else:
            hi = mid
    inner = 0.5 * (lo + hi)
    _CALIB_CACHE[key] = inner
    return inner


class FixedPadsReplay:
    """Co-moving-frame two-finger grip: fixed world pads at the penetration for
    the commanded grip force; the hand's acceleration applied as a pseudo-force
    via opt.gravity = g − Ḧ. The object is a free box captured between the pads.

    The pads are fixed geometry, so the grip force is committed at build time:
    Fn = F_max·(1 − mean(g)) for the transport segment (g is ~constant during
    a hold, so the per-step Fn variation is negligible)."""

    def __init__(self, mu, mass, Fn):
        self.mu, self.mass, self.Fn = mu, mass, Fn
        mu_pad = mu / 2.0                    # two pads → total friction μ·F_n
        # Empirical gap so the settled F_n_total = 2·Fn → MuJoCo capacity
        # μ_pad·F_n_total == μ·F_max·(1−g), the analytic capacity. The old
        # pen = Fn/65e3 guess settled 2.5× too high (Level A: 0 slip).
        inner = _calibrate_inner(mass, Fn, mu_pad)
        wallx = inner + 0.005
        xml = f"""
<mujoco model="level_a_replay">
  <option gravity="0 0 -9.81" timestep="{DT}" integrator="implicitfast"
          cone="elliptic" iterations="200"/>
  <size nconmax="64" njmax="128"/>
  <default>
    <geom type="box" solref="0.001 2" solimp="0.9 0.95 0.001"
          friction="{mu_pad} 0.9 0.001"/>
  </default>
  <worldbody>
    <geom name="wallL" type="box" pos="-{wallx} 0 0.10" size="0.005 0.5 0.15"
          friction="{mu_pad} 0.9 0.001"/>
    <geom name="wallR" type="box" pos="{wallx} 0 0.10" size="0.005 0.5 0.15"
          friction="{mu_pad} 0.9 0.001"/>
    <body name="object" pos="0 0 0.10">
      <freejoint/>
      <geom name="obj" type="box" size="{OBJ_HALF} {OBJ_HALF} {OBJ_HALF}"
            mass="{mass}" friction="{mu_pad} 0.9 0.001"/>
    </body>
  </worldbody>
</mujoco>"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.obj_bid = self.model.body("object").id
        self.gz = float(self.model.opt.gravity[2])     # −9.81
        self.ref = None

    # ── pseudo-force step ─────────────────────────────────────────────
    def _apply_accel(self, D, u):
        """g_eff = g − Ḧ. Ḧ = −k²·Δx·(1−PF)^u (D := Δx), so
        g_eff = (k²·D_x, k²·D_y, g_z + k²·D_z)·(1−PF)^u."""
        k2 = K * K * (1.0 - PLANT_F) ** u
        g = self.model.opt.gravity
        g[0] = k2 * D[0]
        g[1] = k2 * D[1]
        g[2] = self.gz + k2 * D[2]

    def _settle(self):
        """Drop to baseline gravity and let the contact equilibrate. Returns
        the worst displacement during settle (m) — if it exceeds the drop
        threshold the static grip itself is infeasible at this Fn."""
        g = self.model.opt.gravity
        g[0], g[1], g[2] = 0.0, 0.0, self.gz
        mujoco.mj_forward(self.model, self.data)   # xpos before any step is uncomputed
        p0 = self.data.xpos[self.obj_bid].copy()
        worst = 0.0
        for _ in range(int(SETTLE_S / DT)):
            mujoco.mj_step(self.model, self.data)
            worst = max(worst, float(np.linalg.norm(
                self.data.xpos[self.obj_bid] - p0)))
        self.ref = self.data.xpos[self.obj_bid].copy()
        return worst

    # ── single plant step (calibration) ──────────────────────────────
    def one_step(self, D, axis):
        """One plant step from rest, target D along `axis` (0 = pad-normal x,
        1 = pad-tangential y). Returns max slip from the held frame (mm)."""
        self._settle()
        Dv = np.zeros(3)
        Dv[axis] = D
        max_rel = 0.0
        for j in range(TAU_STEPS):
            self._apply_accel(Dv, j / TAU_STEPS)
            mujoco.mj_step(self.model, self.data)
            max_rel = max(max_rel, float(np.linalg.norm(
                self.data.xpos[self.obj_bid] - self.ref)))
        return max_rel * 1000.0

    # ── transport replay ─────────────────────────────────────────────
    def replay(self, states, targets):
        """Replay a transport segment (N,3). Returns (held, max_slip_mm,
        min_dz_mm). A static-hold failure during settle returns held=False
        with the settle drift reported as max_slip."""
        settle_drift = self._settle()
        if settle_drift > DROP_SLIP:
            return False, settle_drift * 1000.0, 0.0
        max_slip, min_dz = 0.0, 0.0
        for i in range(len(states) - 1):
            D = targets[i] - states[i]
            for j in range(TAU_STEPS):
                self._apply_accel(D, j / TAU_STEPS)
                mujoco.mj_step(self.model, self.data)
                rel = self.data.xpos[self.obj_bid] - self.ref
                max_slip = max(max_slip, float(np.linalg.norm(rel)))
                min_dz = min(min_dz, float(rel[2]))
        return (max_slip < DROP_SLIP and min_dz > -DROP_Z), \
            max_slip * 1000.0, min_dz * 1000.0


# ── closed-loop recording ─────────────────────────────────────────────
def record_transport(experts, router, mu, mass, rng, device, plant_f=PLANT_F):
    """One closed-loop episode. Returns (transport_segment or None, verdict)."""
    obj, place = proc_sim._sample_scene(rng)
    env = SimEnv(rng=rng, plant_f=plant_f, physics=True, mu=mu, mass=mass)
    env.reset((obj, place))
    scene = np.concatenate([obj, place]).astype(np.float32)
    ctl = BodyGraphController(experts, router, device=device)
    ctl.reset(scene, env.state.copy())
    tr, drop_skill = [], None
    for _ in range(800):
        target, info = ctl.step(env.state, physics_ctx=env.physics_ctx())
        if ctl.skill == "transport":
            tr.append((env.state.copy(), target.copy()))
        env.step(target)
        if env.dropped and drop_skill is None:
            drop_skill = ctl.skill
            # Do NOT break: keep recording so MuJoCo replays the FULL intended
            # motion (the sim's early drop is exactly what is being adjudicated —
            # cutting at it would hide any later slip accumulation).
        if info["task_done"]:
            break
    verdict = {"ok": env.success(), "dropped": env.dropped,
               "drop_skill": drop_skill, "reached_transport": bool(tr)}
    if not tr:
        return None, verdict
    states = np.array([s for s, _ in tr])
    targets = np.array([t for _, t in tr])
    return {"states": states, "targets": targets,
            "obj_xy": obj, "place_xy": place}, verdict


# ── calibration ───────────────────────────────────────────────────────
def calib_main():
    print("── Level A+ calib: single-step slip onset (MuJoCo vs honest risk, "
          "τ=0.05) ──")
    print("  risk = (m/μ)·√(g0²+(k²Δx)²)/F_max  [the sim's honest load]  (* = >1)")
    print("  mj_y  = MuJoCo max slip (mm), pad-tangential (worst-case yaw) step")
    print("  mj_x  = MuJoCo max slip (mm), pad-normal step (capture; no friction)")
    print("  stp/drop = DROP_SLIP/mj_y → steps to accumulate the drop threshold\n")
    for mu, mass in ((0.20, 0.27), (0.30, 0.36), (0.40, 0.20)):
        rep = FixedPadsReplay(mu, mass, F_MAX)
        print(f"  mu={mu:.2f} m={mass:.2f} (m/μ={mass / mu:.2f})")
        print(f"    {'D_mm':>6} {'risk':>7} | {'mj_y':>7} {'mj_x':>7} "
              f"{'stp/drop':>8}")
        for D in (0.010, 0.025, 0.040, 0.055, 0.070):
            a_h = K * K * D
            risk = mass * np.sqrt(G0 ** 2 + a_h ** 2) / (mu * F_MAX + EPS)
            star = "*" if risk > 1.0 else " "
            slip_y = rep.one_step(D, 1)
            slip_x = rep.one_step(D, 0)
            steps = (DROP_SLIP / (slip_y / 1000.0)) if slip_y > 1e-6 else float("inf")
            sts = f"{steps:6.1f}" if np.isfinite(steps) else "   inf"
            print(f"    {D * 1000:6.0f} {risk:6.2f}{star} | {slip_y:7.2f} "
                  f"{slip_x:7.2f} {sts:>8}")


# ── transport replay ──────────────────────────────────────────────────
def collect_row(experts, router, mu, mass, rng, device, name, row, eps):
    """eps closed-loop episodes; replay every transport segment in MuJoCo.
    Populates row[{name}_sdrop] (drop_skill per episode) and row[{name}_segs]
    (None | (drop_skill, mj_held, slip_mm, minz_mm) per episode)."""
    sdrops, segs = [], []
    for _ in range(eps):
        seg, verdict = record_transport(experts, router, mu, mass, rng, device)
        sdrops.append(verdict["drop_skill"])
        if seg is None:
            segs.append(None)
            continue
        Fn = F_MAX * (1.0 - float(np.mean(seg["states"][:, 6])))
        rep = FixedPadsReplay(mu, mass, Fn)
        # Worst-case yaw: rotate the recorded motion 90° about z so the dominant
        # horizontal component is pad-TANGENTIAL (friction-loaded), matching the
        # analytic model's worst-case orientation. Un-rotated, the motion along
        # the pad normal (x) would be borne by the contact normal — capture —
        # and never slip.
        st = seg["states"][:, :3] @ YAW90.T
        tg = seg["targets"][:, :3] @ YAW90.T
        held, slip, minz = rep.replay(st, tg)
        segs.append((verdict["drop_skill"], held, slip, minz))
    row[f"{name}_sdrop"] = sdrops
    row[f"{name}_segs"] = segs


def _seg_stats(segs, sdrops):
    """From a row's per-episode lists: (n_segments, sim_transport_drops,
    mj_drops_total, mj_drops_of_sim_tdrops)."""
    n = sum(1 for s in segs if s is not None)
    sim_td = sum(1 for d in sdrops if d == "transport")
    mj_d = sum(1 for s in segs if s is not None and s[1] is False)
    mj_of_td = sum(1 for s in segs if s is not None
                  and s[0] == "transport" and s[1] is False)
    return n, sim_td, mj_d, mj_of_td


def report(rows, eps):
    print("── Level A+: MuJoCo independent adjudication (transport segment, "
          "worst-case yaw) ──")
    print(f"  cells: {len(rows)}, {eps} episodes/cell/agent "
          "(sim verdict counts the segment the sim judged; mj verdict is the "
          "independent MuJoCo replay of that exact segment)\n")
    print(f"    {'mu':>5} {'m':>5} {'m/μ':>5} | "
          f"{'aware':>30} | {'blind':>30}")
    print(f"    {'':>5} {'':>5} {'':>5} | "
          f"{'sim tdrop':>9} {'mj drop':>11} {'mj slip':>9} | "
          f"{'sim tdrop':>9} {'mj drop':>11} {'mj slip':>9}")
    print(f"    {'':>5} {'':>5} {'':>5} | "
          f"{'(/n seg)':>9} {'(/n seg)':>11} {'(mm)':>9} | "
          f"{'(/n seg)':>9} {'(/n seg)':>11} {'(mm)':>9}")
    tot = {"an": 0, "amd": 0, "bn": 0, "bt": 0, "brep": 0, "bmd": 0, "btdrop_held": 0}
    detail = []                     # (mu, m, mj_held, slip_mm, minz_mm)
    a_slips, b_slips = [], []
    for r in rows:
        a_n, a_td, a_md, _ = _seg_stats(r["aware_segs"], r["aware_sdrop"])
        b_n, b_td, b_md, b_rep = _seg_stats(r["blind_segs"], r["blind_sdrop"])
        tot["an"] += a_n; tot["amd"] += a_md
        tot["bn"] += b_n; tot["bt"] += b_td; tot["brep"] += b_rep
        tot["bmd"] += b_md
        a_slip_here = [s[2] for s in r["aware_segs"] if s is not None]
        b_slip_here = [s[2] for s in r["blind_segs"] if s is not None]
        a_slips += a_slip_here; b_slips += b_slip_here
        for s in r["blind_segs"]:
            if s is not None and s[0] == "transport":
                detail.append((r["mu"], r["m"], s[1], s[2], s[3]))
                tot["btdrop_held"] += int(s[1])       # sim-drop kept HELD by MuJoCo
        amd = f"{a_md}/{a_n}" if a_n else "n/a"
        bmd = f"{b_md}/{b_n}" if b_n else "n/a"
        a_ms = f"{np.mean(a_slip_here):6.2f}" if a_slip_here else "  n/a"
        b_ms = f"{np.mean(b_slip_here):6.2f}" if b_slip_here else "  n/a"
        print(f"    {r['mu']:5.2f} {r['m']:5.2f} {r['m'] / r['mu']:5.2f} | "
              f"{a_td:>9} {amd:>11} {a_ms:>9} | {b_td:>9} {bmd:>11} {b_ms:>9}")
    print(f"\n  aware: {tot['amd']}/{tot['an']} replayed transport segments "
          f"dropped by MuJoCo (expect 0 — aware should hold)")
    print(f"  blind: {tot['bt']} transport segments judged DROPPED by the sim; "
          f"MuJoCo independently reproduces {tot['brep']} of them "
          f"({tot['btdrop_held']} kept HELD by MuJoCo)")
    print(f"         {tot['bn']} total blind transport segments replayed; "
          f"MuJoCo dropped {tot['bmd']} of them")
    if detail:
        print("\n  sim-judged blind transport-drops, replayed in MuJoCo "
              "(max slip / min dz from held frame):")
        for mu, m, held, slip, minz in detail:
            out = "HELD" if held else "DROPPED"
            print(f"    mu={mu:.2f} m={m:.3f} (m/μ={m / mu:.2f}): "
                  f"MuJoCo slip {slip:6.2f} mm, min dz {minz:6.2f} mm  → {out}")
    if tot["bt"] and tot["brep"] > 0 and tot["amd"] == 0:
        print("\n  → OUTCOME CONFIRMED: on the discriminating cells MuJoCo's "
              "independent rigid-body adjudication reproduces blind-dropped / "
              "aware-held.")
    else:
        print("\n  → OUTCOME NOT CONFIRMED: the sim's discriminating drop verdict "
              "is not reproduced by MuJoCo (the analytic slip-rate over-predicts "
              "the rigid-body accumulation; see calib).")
    if a_slips and b_slips and np.mean(a_slips) > 0 and np.mean(b_slips) > np.mean(a_slips) * 3:
        print("  → MECHANISM CONFIRMED: MuJoCo slip magnitude on the band is "
              f"{np.mean(b_slips):.2f} mm (blind) vs {np.mean(a_slips):.2f} mm "
              "(aware) — the aware's contraction genuinely eliminates the "
              "rigid-body slip the blind's steps produce.")
    else:
        print(f"  → MECHANISM {('CONFIRMED' if a_slips and b_slips and np.mean(b_slips) > np.mean(a_slips) else 'INDISTINCT')}: "
              f"blind mean slip {np.mean(b_slips):.2f} mm vs aware "
              f"{np.mean(a_slips):.2f} mm.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aware-dir", type=str, default="checkpoints_phys")
    ap.add_argument("--blind-dir", type=str, default="checkpoints_rollin2")
    ap.add_argument("--router-ckpt", type=str, default="checkpoints/moe_router.pt")
    ap.add_argument("--n-cells", type=int, default=15)
    ap.add_argument("--eps", type=int, default=3, help="episodes per cell/agent")
    ap.add_argument("--hard-frac", type=float, default=0.5)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.RandomState(args.seed)

    if args.calib:
        calib_main()
        return

    aware = load_experts(args.aware_dir, args.blind_dir, device)
    blind = load_experts(None, args.blind_dir, device)
    router = StateRouter().to(device).eval()
    router.load_state_dict(torch.load(args.router_ckpt, map_location=device,
                                      weights_only=True)["model"])

    rows = []
    for mu, m in cells(args.n_cells, rng, hard_frac=args.hard_frac):
        row = {"mu": mu, "m": m, "margin": margin(mu, m)}
        collect_row(aware, router, mu, m, rng, device, "aware", row, args.eps)
        collect_row(blind, router, mu, m, rng, device, "blind", row, args.eps)
        rows.append(row)
    report(rows, args.eps)


if __name__ == "__main__":
    main()
