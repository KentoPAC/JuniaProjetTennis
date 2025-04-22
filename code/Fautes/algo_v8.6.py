import json
import matplotlib.pyplot as plt

# Who is playing: "bottom_player" or "top_player"
# We'll plot both scenarios separately below.

# Court points JSON
example_court = '''
{
    "points": [
        {"x": 387, "y": 149}, {"x": 875, "y": 150},
        {"x": 83,  "y": 510}, {"x": 1187, "y": 511},
        {"x": 449, "y": 149}, {"x": 225, "y": 507},
        {"x": 816, "y": 151}, {"x": 1049,"y": 509},
        {"x": 423, "y": 189}, {"x": 839, "y": 191},
        {"x": 319, "y": 365}, {"x": 954, "y": 366},
        {"x": 633, "y": 195}, {"x": 635, "y": 363}
    ]
}
'''
data = json.loads(example_court)
pts = data["points"]
xs = [p["x"] for p in pts]
ys = [p["y"] for p in pts]

# Key corners
X4, Y4 = xs[4], ys[4]   # north-left single
X6, Y6 = xs[6], ys[6]   # north-right single
X5, Y5 = xs[5], ys[5]   # south-left single
X7, Y7 = xs[7], ys[7]   # south-right single

# Net line
mid_y = (ys[12] + ys[13]) / 2
net_y = mid_y - 0.1 * (ys[13] - ys[12])

# Perspective‐aware boundaries
def singles_x_left(y):
    return X4 + (X5 - X4) * (y - Y4) / (Y5 - Y4)

def singles_x_right(y):
    return X6 + (X7 - X6) * (y - Y6) / (Y7 - Y6)

def is_inside_singles_half(x, y, north=True):
    xl, xr = singles_x_left(y), singles_x_right(y)
    if north:
        return xl <= x <= xr and Y4 <= y <= net_y
    else:
        return xl <= x <= xr and net_y <= y <= Y5

def rally_verdict(x, y, player):
    north = (y <= net_y)
    if north and player == "bottom_player":
        return is_inside_singles_half(x, y, north=True)
    if (not north) and player == "top_player":
        return is_inside_singles_half(x, y, north=False)
    return False

# ε offset for edge tests
eps = 1.0

# Generate edge‐case balls
def make_edge_tests(player):
    tests = []
    # Choose y positions for top/bottom tests
    if player == "bottom_player":
        y_base = (Y4 + net_y) / 2  # mid‐north court
        y_net_in = net_y - eps
        y_net_out = net_y + eps
    else:
        y_base = (net_y + Y5) / 2  # mid‐south court
        y_net_in = net_y + eps
        y_net_out = net_y - eps

    # Left edge
    xl = singles_x_left(y_base)
    tests.append({"x": xl + eps, "y": y_base, "desc": "left_in"})
    tests.append({"x": xl - eps, "y": y_base, "desc": "left_out"})

    # Right edge
    xr = singles_x_right(y_base)
    tests.append({"x": xr - eps, "y": y_base, "desc": "right_in"})
    tests.append({"x": xr + eps, "y": y_base, "desc": "right_out"})

    # Baseline edge
    if player == "bottom_player":
        tests.append({"x": (xl + xr)/2, "y": Y4 + eps, "desc": "base_in"})
        tests.append({"x": (xl + xr)/2, "y": Y4 - eps, "desc": "base_out"})
    else:
        tests.append({"x": (xl + xr)/2, "y": Y5 - eps, "desc": "base_in"})
        tests.append({"x": (xl + xr)/2, "y": Y5 + eps, "desc": "base_out"})

    # Net edge
    tests.append({"x": (xl + xr)/2, "y": y_net_in, "desc": "net_in"})
    tests.append({"x": (xl + xr)/2, "y": y_net_out, "desc": "net_out"})

    # Attach player
    for t in tests:
        t["player"] = player
    return tests

# Combine tests for both players
all_tests = make_edge_tests("bottom_player") + make_edge_tests("top_player")

# Plotting
fig, ax = plt.subplots(figsize=(9, 6))

# Court outline
outer = [0,1,3,2,0]
ax.plot([xs[i] for i in outer], [ys[i] for i in outer], lw=2)
ax.plot([X4, X6], [Y4, Y6], lw=2)  # north baseline
ax.plot([X5, X7], [Y5, Y7], lw=2)  # south baseline
ax.plot([xs[8], xs[9]], [ys[8], ys[9]], lw=2)
ax.plot([xs[10], xs[11]], [ys[10], ys[11]], lw=2)
ax.plot([xs[12], xs[13]], [ys[12], ys[13]], lw=2)

# Plot net line
ax.axhline(net_y, color='red', linestyle='--')
ax.text(min(xs), net_y-10, 'net line', color='red')

# Plot edge tests
for b in all_tests:
    val = rally_verdict(b["x"], b["y"], b["player"])
    c = 'green' if val else 'red'
    ax.scatter(b["x"], b["y"], color=c, s=60)
    ax.text(b["x"]+5, b["y"]+5,
            f"{b['desc']} ({b['player'][:3]})",
            fontsize=8, color=c)

ax.invert_yaxis()
ax.set_aspect('equal', 'box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Edge-Case Tests – Perspective‐Corrected Court Boundaries')
plt.tight_layout()
plt.show()
