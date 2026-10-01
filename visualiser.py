"""
visualiser.py - values and policy visualiser for Value Iteration / Policy Iteration (COMP3702 A2, Q2c)

Usage (run next to solution.py, game_env.py and game_state.py):
    python visualiser.py testcases/L1.txt pi
    python visualiser.py testcases/L1.txt vi
    python visualiser.py testcases/L1.txt pi --save out.png --iteration 3      (save an image instead of opening a window)

Left panel: value V(s) of every tile (red = low, green = high). Right panel: the action chosen in every tile.
Slider: step through every iteration of the algorithm. Radio buttons: choose which crystal-status layer to show.
"""
import sys

import numpy

if '--save' in sys.argv:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.widgets import RadioButtons, Slider

from game_env import GameEnv
from game_state import GameState
from solution import Solver

ARROW = {'w': ('black', 1.6, 0.30), 'b': ('#e8833a', 3.2, 0.42), 'j': ('#1f9bcf', 2.0, 0.30)}    # colour, width, length
DIRECTION = {'l': (-1, 0), 'r': (1, 0), 'u': (0, -1), 'd': (0, 1)}


def record_iterations(solver, algorithm):
    """Run the solver one iteration at a time and keep (values, policy as action indices) after every iteration."""
    snapshots = []
    if algorithm == 'vi':
        solver.vi_initialise()
        to_index = lambda: numpy.array([GameEnv.ACTIONS.index(a) for a in solver.vi_policy])
        snapshots.append((solver.vi_values.copy(), to_index()))
        while True:
            solver.vi_iteration()
            snapshots.append((solver.vi_values.copy(), to_index()))
            if solver.vi_is_converged():
                break
    else:
        solver.pi_initialise()
        snapshots.append((numpy.zeros(len(solver.states)), solver.pi_policy.copy()))
        while True:
            solver.pi_iteration()
            snapshots.append((solver.pi_values.copy(), solver.pi_policy.copy()))
            if solver.pi_is_converged():
                break
    return snapshots


def draw(ax_values, ax_policy, env, solver, snapshot, status, vmin):
    values, policy = snapshot
    for ax in (ax_values, ax_policy):
        ax.clear()
        ax.set_xlim(-0.5, env.n_cols - 0.5)
        ax.set_ylim(env.n_rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    cmap = plt.get_cmap('RdYlGn')
    for row in range(env.n_rows):
        for col in range(env.n_cols):
            tile = env.grid_data[row][col]
            index = solver.state_indices.get(GameState(row, col, status))
            if tile == env.ROCK_TILE:
                face = '#9aa0a6'
            elif tile == env.LAVA_TILE:
                face = '#ff5a1f'
            elif index is not None and solver.terminal_mask[index]:
                face = '#5bc0de'                                                  # solved (launch) tile
            elif index is not None:
                face = cmap(min(1.0, max(0.0, (values[index] - vmin) / (0 - vmin))))
            else:
                face = '#d9d9d9'                                                  # not reachable in this layer
            ax_values.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=face, edgecolor='#444444', lw=.4))
            policy_face = {'#9aa0a6': '#9aa0a6', '#ff5a1f': '#ff5a1f', '#5bc0de': '#5bc0de'}.get(face, '#f4f4f4')
            if tile == env.CRATER_TILE and policy_face == '#f4f4f4':
                policy_face = '#e3d3c3'                                           # craters shaded so jumps are easy to see
            ax_policy.add_patch(Rectangle((col - .5, row - .5), 1, 1, facecolor=policy_face, edgecolor='#444444', lw=.4))
            if index is None or tile == env.ROCK_TILE or tile == env.LAVA_TILE:
                continue
            if solver.terminal_mask[index]:
                ax_values.text(col, row, 'E', ha='center', va='center', fontsize=9, fontweight='bold')
                ax_policy.text(col, row, 'E', ha='center', va='center', fontsize=9, fontweight='bold')
                continue
            ax_values.text(col, row, f'{values[index]:.0f}', ha='center', va='center', fontsize=6.5)
            action = GameEnv.ACTIONS[policy[index]]
            colour, width, length = ARROW[action[0]]
            dx, dy = DIRECTION[action[1]]
            ax_policy.annotate('', xy=(col + dx * length, row + dy * length), xytext=(col - dx * length, row - dy * length),
                               arrowprops=dict(arrowstyle='-|>', color=colour, lw=width, shrinkA=0, shrinkB=0))
    for i, (row, col) in enumerate(env.crystal_positions):
        if status[i] == 0:
            ax_values.plot(col, row, marker='D', color='#7fe7f2', mec='#0b6b78', ms=6)
            ax_policy.plot(col, row, marker='D', color='#7fe7f2', mec='#0b6b78', ms=6)
    ax_values.set_title('Value V(s)  (red = low, green = high)', fontsize=10)
    ax_policy.set_title('Policy  (action chosen in each tile)', fontsize=10)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    testcase = args[0] if args else 'L1.txt'
    algorithm = args[1].lower() if len(args) > 1 else 'pi'
    env = GameEnv(testcase)
    solver = Solver(env)
    snapshots = record_iterations(solver, algorithm)
    statuses = sorted({s.crystal_status for s in solver.states})
    initial_status = env.get_init_state().crystal_status
    vmin = min(float(v[~solver.terminal_mask].min()) for v, _ in snapshots[1:])

    fig = plt.figure(figsize=(13, 7))
    fig.suptitle(f'{algorithm.upper()} on {testcase}', fontsize=12)
    ax_values = fig.add_axes([0.03, 0.22, 0.45, 0.68])
    ax_policy = fig.add_axes([0.52, 0.22, 0.45, 0.68])
    ax_slider = fig.add_axes([0.28, 0.10, 0.45, 0.04])
    ax_radio = fig.add_axes([0.01, 0.01, 0.14, 0.17])
    slider = Slider(ax_slider, 'Iteration', 0, len(snapshots) - 1, valinit=len(snapshots) - 1, valstep=1)
    labels = [''.join(str(c) for c in s) for s in statuses]
    radio = RadioButtons(ax_radio, labels, active=statuses.index(initial_status))
    ax_radio.set_title('crystals collected', fontsize=8)
    for label in radio.labels:
        label.set_fontsize(8)
    state = {'status': initial_status}

    def update(_=None):
        k = int(slider.val)
        draw(ax_values, ax_policy, env, solver, snapshots[k], state['status'], vmin)
        ax_values.set_xlabel(f'iteration {k} of {len(snapshots) - 1}', fontsize=9)
        fig.canvas.draw_idle()

    def choose_status(label):
        state['status'] = statuses[labels.index(label)]
        update()

    slider.on_changed(update)
    radio.on_clicked(choose_status)
    legend = [Line2D([0], [0], color='black', lw=1.6, label='walk'), Line2D([0], [0], color='#e8833a', lw=3.2, label='boost'),
              Line2D([0], [0], color='#1f9bcf', lw=2.0, label='jump (from a crater)')]
    fig.legend(handles=legend, loc='lower right', ncol=3, fontsize=9, frameon=False)

    if '--save' in sys.argv:
        out = sys.argv[sys.argv.index('--save') + 1]
        if '--iteration' in sys.argv:
            slider.set_val(min(int(sys.argv[sys.argv.index('--iteration') + 1]), len(snapshots) - 1))
        else:
            update()
        fig.savefig(out, dpi=170)
        print('saved', out)
    else:
        update()
        plt.show()


if __name__ == '__main__':
    main()
