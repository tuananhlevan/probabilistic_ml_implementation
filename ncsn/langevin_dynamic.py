import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def langevin_dynamic(history):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Set plot limits (adjust based on your data scale)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4.5)
    ax.set_title("Langevin Dynamics Sampling")

    # Scatter plot that will be updated
    scat = ax.scatter([], [], s=10, alpha=0.6, c='#1f77b4')

    def update(frame):
        data = history[frame]
        scat.set_offsets(data)
        ax.set_title(f"Langevin Dynamics Step: {frame * 10}")
        return scat,

    anim = FuncAnimation(fig, update, frames=len(history), interval=50, blit=True)
    plt.show()
    # anim.save("langevin_dynamic.gif", writer=PillowWriter(fps=60))