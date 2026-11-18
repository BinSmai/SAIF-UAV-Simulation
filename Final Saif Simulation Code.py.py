"""
Directional Thermal Shield (DTS) - 3-Node Military UAV Thermal Simulation
=========================================================================
Airframe Material: Carbon Fiber Reinforced Polymer (CFRP) with Tactical Matte
                   Grey Low-Observable Coating (alpha = 0.78, cp = 1050 J/kgK)

Nodes:
  1. T_top  : Top CFRP battery housing / upper interface [C]
  2. T_core : Internal 6S 10Ah LiPo cell stack [C]
  3. T_bot  : Bottom CFRP housing / copper spreader plate [C]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Boundary & Military UAV Parameters ----------------
T_AMB   = 45.0      # Ambient air temp [C] (Desert theater / Saudi summer)
G_SUN   = 1000.0    # Solar irradiance [W/m2]
H_CONV  = 15.0      # Flight convection coeff. [W/m2K]
Q_GEN   = 20.0      # Internal Joule heat load during tactical cruise [W]

# Military UAV Airframe: CFRP with Tactical Matte Grey Coating
ALPHA   = 0.78      # Solar absorptivity (tactical matte grey CARC finish)
CP_CFRP = 1050.0    # Specific heat of CFRP composite [J/kgK]

# Battery geometry & thermal mass (6S 10Ah LiPo pack in CFRP bay)
L, W, H = 0.20, 0.10, 0.06
A_TOP   = L * W                     # 0.020 m2
A_BOT   = L * W                     # 0.020 m2
A_SIDE  = 2 * (L * H) + 2 * (W * H) # 0.036 m2

M_CORE, CP_CORE = 1.08, 1000.0      # Cell core mass [kg], specific heat [J/kgK]
M_CASE          = 0.06              # Top/bottom 1.8 mm CFRP shell mass each [kg]

# ---------------- Shield Materials & Mass Calculation ----------------
K_AERO, T_AERO, RHO_AERO = 0.018, 0.005, 120.0   # 5 mm Silica Aerogel
K_TIM,  T_TIM,  RHO_TIM  = 3.0,   0.001, 2000.0  # 1 mm Thermal Interface Pad
RHO_CU, CP_CU            = 8960.0, 385.0         # Copper spreader & fins

# Copper spreader (0.5 mm base) + 12 thin fins (0.4 mm thick -> 0.10 m2 effective area)
A_BOT_FINS = 0.10   # Effective finned convection area [m2]
ETA_FIN    = 0.85   # Fin efficiency

MASS_AERO = A_TOP * T_AERO * RHO_AERO                                  # 12 g
MASS_TIM  = A_BOT * T_TIM * RHO_TIM                                    # 40 g
# Fins are two-sided: convection area = 2 x fin plate area, so material area = (A_BOT_FINS - A_BOT)/2
MASS_CU   = (A_BOT * 0.0005 + (A_BOT_FINS - A_BOT) / 2 * 0.0004) * RHO_CU  # 233 g
MASS_DTS  = MASS_AERO + MASS_TIM + MASS_CU                             # 285 g

# Internal cell-to-face thermal conductance [W/K]
UA_INT = 2.0  # R_int = 0.50 K/W across cell stack

# Operational limits & time settings
T_SAFE   = 60.0
T_DANGER = 80.0
DT       = 0.5
T_END    = 40 * 60  # 40-minute UAV mission [s]


def get_scheme_params(scheme):
    """Return conductances (W/K) and node heat capacities (J/K)."""
    t_sol = T_AMB + (ALPHA * G_SUN) / H_CONV  # 97.0 C under tactical matte grey CFRP

    c_top  = M_CASE * CP_CFRP
    c_core = M_CORE * CP_CORE
    c_bot  = M_CASE * CP_CFRP

    # External top resistance
    r_top_conv = 1.0 / (H_CONV * A_TOP)
    r_top_ins  = (T_AERO / (K_AERO * A_TOP)) if scheme in ("B", "C") else 0.0
    ua_top_ext = 1.0 / (r_top_ins + r_top_conv)

    # Side resistance (coupled directly to core)
    r_side_conv = 1.0 / (H_CONV * A_SIDE)
    r_side_ins  = (T_AERO / (K_AERO * A_SIDE)) if scheme == "B" else 0.0
    ua_side_ext = 1.0 / (r_side_ins + r_side_conv)

    # Core-to-bottom & external bottom resistance
    if scheme == "A":
        ua_core_bot = UA_INT
        ua_bot_ext  = H_CONV * A_BOT
    elif scheme == "B":
        ua_core_bot = UA_INT
        ua_bot_ext  = 1.0 / (T_AERO / (K_AERO * A_BOT) + 1.0 / (H_CONV * A_BOT))
    elif scheme == "C":
        r_tim = T_TIM / (K_TIM * A_BOT)
        ua_core_bot = 1.0 / (1.0 / UA_INT + r_tim)
        ua_bot_ext  = H_CONV * A_BOT_FINS * ETA_FIN
        c_bot += MASS_CU * CP_CU
    else:
        raise ValueError(scheme)

    return {
        "t_sol": t_sol,
        "c": (c_top, c_core, c_bot),
        "ua_top_ext": ua_top_ext,
        "ua_core_top": UA_INT,
        "ua_side_ext": ua_side_ext,
        "ua_core_bot": ua_core_bot,
        "ua_bot_ext": ua_bot_ext,
    }


def simulate_3node(scheme):
    p = get_scheme_params(scheme)
    c_top, c_core, c_bot = p["c"]
    n = int(T_END / DT)

    T = np.full((n + 1, 3), T_AMB)  # columns: [T_top, T_core, T_bot]
    for i in range(n):
        tt, tc, tb = T[i]
        q_top_in   = p["ua_top_ext"] * (p["t_sol"] - tt)
        q_top_core = p["ua_core_top"] * (tt - tc)
        q_side_out = p["ua_side_ext"] * (tc - T_AMB)
        q_core_bot = p["ua_core_bot"] * (tc - tb)
        q_bot_out  = p["ua_bot_ext"] * (tb - T_AMB)

        T[i + 1, 0] = tt + DT * (q_top_in - q_top_core) / c_top
        T[i + 1, 1] = tc + DT * (Q_GEN + q_top_core - q_side_out - q_core_bot) / c_core
        T[i + 1, 2] = tb + DT * (q_core_bot - q_bot_out) / c_bot

    t_min = np.arange(n + 1) * DT / 60.0
    return t_min, T, p


def steady_state_3node(scheme):
    p = get_scheme_params(scheme)
    G = np.array([
        [p["ua_top_ext"] + p["ua_core_top"], -p["ua_core_top"], 0.0],
        [-p["ua_core_top"], p["ua_core_top"] + p["ua_side_ext"] + p["ua_core_bot"], -p["ua_core_bot"]],
        [0.0, -p["ua_core_bot"], p["ua_core_bot"] + p["ua_bot_ext"]]
    ])
    b = np.array([
        p["ua_top_ext"] * p["t_sol"],
        Q_GEN + p["ua_side_ext"] * T_AMB,
        p["ua_bot_ext"] * T_AMB
    ])
    return np.linalg.solve(G, b)


if __name__ == "__main__":
    labels = {
        "A": "Scheme A: Baseline CFRP (Exposed)",
        "B": "Scheme B: Full Aerogel",
        "C": "Scheme C: Proposed DTS"
    }
    colors = {"A": "#d9822b", "B": "#b83227", "C": "#1e7e48"}

    sim_data = {k: simulate_3node(k) for k in labels}
    ss_data  = {k: steady_state_3node(k) for k in labels}

    # ---- Poster-Ready 2-Panel Figure ----
    plt.rcParams.update({"font.sans-serif": "Arial", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # Panel 1: Transient Core Temperature
    for k, (t, T, _) in sim_data.items():
        ax1.plot(t, T[:, 1], lw=2.5, color=colors[k], label=labels[k])
    ax1.axhline(T_SAFE, color="#222222", ls="--", lw=1.2)
    ax1.text(0.8, T_SAFE + 1.0, "Li-ion Safe Limit (60 °C)", fontsize=8.5, fontweight="bold")
    ax1.set_title("A. Core Temperature Over 40-Min Mission", fontweight="bold", pad=8)
    ax1.set_xlabel("Flight Time (min)")
    ax1.set_ylabel("Battery Core Temperature (°C)")
    ax1.set_xlim(0, 40)
    ax1.set_ylim(44, 88)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Panel 2: Steady-State Top vs. Core vs. Bottom Temperatures
    x = np.arange(3)
    width = 0.25
    t_tops  = [ss_data[k][0] for k in labels]
    t_cores = [ss_data[k][1] for k in labels]
    t_bots  = [ss_data[k][2] for k in labels]

    ax2.bar(x - width, t_tops,  width, label="Top CFRP Shell", color="#e07a5f")
    ax2.bar(x,         t_cores, width, label="Battery Core", color="#3d405b")
    ax2.bar(x + width, t_bots,  width, label="Bottom Plate", color="#81b29a")
    ax2.axhline(T_SAFE, color="#222222", ls="--", lw=1.2)

    for idx, val in enumerate(t_cores):
        ax2.text(idx, val + 2.5, f"{val:.1f}°C", ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_title("B. Steady-State Vertical Profile", fontweight="bold", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Scheme A\n(Exposed CFRP)", "Scheme B\n(Full Aerogel)", "Scheme C\n(Proposed DTS)"])
    ax2.set_ylabel("Steady-State Temperature (°C)")
    ax2.set_ylim(40, 165)
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    fig.savefig("poster_results_plot.png", dpi=300)

    # ---- Console Summary ----
    print(f"Airframe: Tactical Matte Grey CFRP (alpha={ALPHA}, T_sol={45 + ALPHA*1000/15:.1f} C)")
    print(f"Total DTS Shield Mass: {MASS_DTS*1000:.1f} g (+{MASS_DTS/1.20*100:.1f}% of 1.2 kg pack)")
    print("-" * 84)
    print(f"{'Scheme':<32}{'T_core@40m':>11}{'T_core_SS':>11}{'Top-Bot dT':>12}{'Peak Rise':>11}")
    print("-" * 84)
    rise_A = sim_data["A"][1][-1, 1] - T_AMB
    for k in labels:
        t_40 = sim_data[k][1][-1, 1]
        tt, tc, tb = ss_data[k]
        rise = t_40 - T_AMB
        print(f"{labels[k]:<32}{t_40:>10.1f}C{tc:>10.1f}C{tt - tb:>11.1f}C{rise:>10.1f}K")
    rise_C = sim_data["C"][1][-1, 1] - T_AMB
    print("-" * 84)
    print(f"40-min Core Temperature-Rise Reduction (C vs A): {(rise_A - rise_C)/rise_A*100:.1f}%")