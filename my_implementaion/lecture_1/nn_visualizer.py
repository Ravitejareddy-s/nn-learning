"""
Neural net visualizer for microg MLP.

Usage in your notebook:
    from nn_visualizer import draw_mlp, draw_training_gif

    # Single snapshot
    draw_mlp(x, input_values=[53])

    # During training - collect frames then make gif
    draw_training_gif(x, dataset, epochs=100, lr=0.000000001, filename="training.gif")
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import os


def _forward_and_capture(mlp_model, input_values):
    """Run forward pass and capture each layer's neuron outputs."""
    # normalize input values to plain floats
    input_floats = [v.value if hasattr(v, 'value') else float(v) for v in input_values]
    layer_outputs = [input_floats]
    a = input_values
    for layer in mlp_model.layers:
        a = layer.execute(a)
        layer_outputs.append([n.value if hasattr(n, 'value') else float(n) for n in a])
    return layer_outputs


def _get_layer_structure(mlp_model):
    """Get number of neurons per layer including input."""
    # figure out input size from first layer's first neuron's weight count
    first_neuron = mlp_model.layers[0].nurons[0]
    n_inputs = len(first_neuron.weights)
    sizes = [n_inputs]
    for layer in mlp_model.layers:
        sizes.append(len(layer.nurons))
    return sizes


def _get_weights_and_biases(mlp_model):
    """Extract weights and biases organized by layer and neuron."""
    layers_data = []
    for layer in mlp_model.layers:
        layer_data = []
        for n in layer.nurons:
            w = [w.value for w in n.weights]
            b = n.bias.value
            layer_data.append({'weights': w, 'bias': b})
        layers_data.append(layer_data)
    return layers_data


def _get_gradients(mlp_model):
    """Extract gradients organized by layer and neuron."""
    layers_data = []
    for layer in mlp_model.layers:
        layer_data = []
        for n in layer.nurons:
            w = [w.grad for w in n.weights]
            b = n.bias.grad
            layer_data.append({'weight_grads': w, 'bias_grad': b})
        layers_data.append(layer_data)
    return layers_data


def draw_mlp(mlp_model, input_values, epoch=None, loss_val=None,
             ax=None, save_path=None, show=True, figsize=None):
    """
    Draw the MLP showing:
    - Neurons as circles, fill intensity (green) = output magnitude
    - Connections colored red, intensity = weight magnitude
    - Bias shown as blue indicator inside each neuron
    - Weight values annotated on connections
    - Gradient arrows shown if available

    Parameters
    ----------
    mlp_model : mlp instance from your notebook
    input_values : list of input values (raw numbers, not microg)
    epoch : optional epoch number to display
    loss_val : optional loss value to display
    ax : optional matplotlib axes
    save_path : optional file path to save the figure
    show : whether to call plt.show()
    figsize : optional (width, height) tuple
    """
    layer_sizes = _get_layer_structure(mlp_model)
    layer_outputs = _forward_and_capture(mlp_model, input_values)
    wb_data = _get_weights_and_biases(mlp_model)

    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)

    if figsize is None:
        figsize = (max(4 * n_layers, 10), max(2.5 * max_neurons, 6))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-1, max_neurons)
    ax.set_aspect('equal')
    ax.axis('off')

    # compute neuron positions
    positions = []  # positions[layer_idx] = list of (x, y)
    for li, size in enumerate(layer_sizes):
        x = li
        ys = np.linspace(0, max_neurons - 1, size) if size > 1 else [max_neurons / 2 - 0.5]
        # center vertically
        offset = (max_neurons - 1 - (ys[-1] - ys[0])) / 2 if size > 1 else 0
        ys = [y + offset for y in ys]
        positions.append([(x, y) for y in ys])

    # collect all output values for normalization
    all_outputs = []
    for lo in layer_outputs:
        all_outputs.extend([abs(v) for v in lo])
    max_output = max(all_outputs) if all_outputs and max(all_outputs) > 0 else 1.0

    # collect all weight magnitudes for normalization
    all_weights = []
    for ld in wb_data:
        for nd in ld:
            all_weights.extend([abs(w) for w in nd['weights']])
    max_weight = max(all_weights) if all_weights and max(all_weights) > 0 else 1.0

    # collect all bias magnitudes for normalization
    all_biases = []
    for ld in wb_data:
        for nd in ld:
            all_biases.append(abs(nd['bias']))
    max_bias = max(all_biases) if all_biases and max(all_biases) > 0 else 1.0

    radius = 0.25

    # --- draw connections (weights) ---
    for li in range(len(wb_data)):  # li = layer index in wb_data (0 = first hidden)
        src_layer = li       # source positions index
        dst_layer = li + 1   # destination positions index
        for ni, nd in enumerate(wb_data[li]):
            dst_x, dst_y = positions[dst_layer][ni]
            for wi, w_val in enumerate(nd['weights']):
                src_x, src_y = positions[src_layer][wi]
                # intensity based on weight magnitude
                intensity = min(abs(w_val) / max_weight, 1.0)
                # red channel for weights, negative = dashed
                if w_val >= 0:
                    color = (intensity, 0.0, 0.0, 0.3 + 0.7 * intensity)
                    linestyle = '-'
                else:
                    color = (0.0, 0.0, intensity, 0.3 + 0.7 * intensity)
                    linestyle = '--'

                linewidth = 0.5 + 3.0 * intensity
                ax.plot([src_x + radius, dst_x - radius],
                        [src_y, dst_y],
                        color=color, linewidth=linewidth,
                        linestyle=linestyle, zorder=1)

                # annotate weight value at midpoint
                mid_x = (src_x + dst_x) / 2
                mid_y = (src_y + dst_y) / 2
                ax.text(mid_x, mid_y, f'{w_val:.2f}',
                        fontsize=6, ha='center', va='bottom',
                        color='darkred', zorder=5,
                        bbox=dict(boxstyle='round,pad=0.1',
                                  facecolor='white', alpha=0.7, edgecolor='none'))

    # --- draw neurons ---
    for li, (size, pos_list) in enumerate(zip(layer_sizes, positions)):
        for ni, (nx, ny) in enumerate(pos_list):
            # output intensity (green)
            out_val = layer_outputs[li][ni]
            out_intensity = min(abs(out_val) / max_output, 1.0)
            neuron_color = (0.2, 0.4 + 0.6 * out_intensity, 0.2, 0.3 + 0.7 * out_intensity)

            circle = plt.Circle((nx, ny), radius, facecolor=neuron_color,
                                edgecolor='black', linewidth=1.5, zorder=3)
            ax.add_patch(circle)

            # output value text
            ax.text(nx, ny + 0.02, f'{out_val:.2f}',
                    fontsize=7, ha='center', va='center',
                    fontweight='bold', zorder=4)

            # bias indicator (blue bar at bottom of neuron) - skip input layer
            if li > 0:
                bias_val = wb_data[li - 1][ni]['bias']
                bias_intensity = min(abs(bias_val) / max_bias, 1.0)
                bias_color = (0.1, 0.1, 0.4 + 0.6 * bias_intensity)
                # small rectangle at bottom of neuron
                bar_w = radius * 1.4
                bar_h = 0.06
                bar = patches.FancyBboxPatch(
                    (nx - bar_w / 2, ny - radius + 0.02),
                    bar_w, bar_h,
                    boxstyle="round,pad=0.01",
                    facecolor=bias_color, edgecolor='none', zorder=4)
                ax.add_patch(bar)
                ax.text(nx, ny - radius + 0.06, f'b:{bias_val:.2f}',
                        fontsize=5, ha='center', va='center',
                        color='white', zorder=5)

    # --- layer labels ---
    for li, size in enumerate(layer_sizes):
        if li == 0:
            label = 'Input'
        elif li == n_layers - 1:
            label = 'Output'
        else:
            label = f'Hidden {li}'
        top_y = max(p[1] for p in positions[li])
        ax.text(li, top_y + 0.6, label,
                fontsize=10, ha='center', va='bottom', fontweight='bold')

    # --- legend ---
    legend_elements = [
        patches.Patch(facecolor=(0.8, 0.0, 0.0, 0.7), label='Weight (+, red)'),
        patches.Patch(facecolor=(0.0, 0.0, 0.8, 0.7), label='Weight (-, blue dashed)'),
        patches.Patch(facecolor=(0.1, 0.1, 0.8), label='Bias (blue bar)'),
        patches.Patch(facecolor=(0.2, 0.8, 0.2, 0.7), label='Output (green fill)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.8)

    # --- title ---
    title_parts = ['MLP Visualization']
    if epoch is not None:
        title_parts.append(f'Epoch: {epoch}')
    if loss_val is not None:
        title_parts.append(f'Loss: {loss_val:.2f}')
    ax.set_title('  |  '.join(title_parts), fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    if show and created_fig:
        plt.show()
    elif created_fig and not show:
        plt.close(fig)

    return fig, ax


def draw_training_gif(mlp_model, dataset, epochs=100, lr=0.000000001,
                      filename="training.gif", input_sample=None,
                      frame_every=1, figsize=None, fps=3):
    """
    Train the MLP and generate a GIF showing the network evolving.

    Parameters
    ----------
    mlp_model : mlp instance
    dataset : list of [[inputs], target] pairs
    epochs : number of training epochs
    lr : learning rate
    filename : output gif filename
    input_sample : input to use for visualization (default: first dataset item)
    frame_every : save a frame every N epochs (use >1 for long training)
    figsize : optional figure size
    fps : frames per second in the gif
    """
    # need microg class - import from the notebook's namespace
    # we'll get it from the model's weights
    microg_cls = type(mlp_model.weights[0])

    if input_sample is None:
        input_sample = dataset[0][0]

    frames_dir = '_nn_frames'
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []

    for epoch in range(epochs):
        # --- forward pass to compute loss ---
        if epoch > 0:
            loss.zero_grad()

        loss = microg_cls(0)
        for sample in dataset:
            temp = mlp_model.input(sample[0])
            t2 = temp[0] - sample[1]
            t2 = t2 * t2
            loss += t2

        loss.backward()

        # --- save frame ---
        if epoch % frame_every == 0:
            frame_path = os.path.join(frames_dir, f'frame_{epoch:05d}.png')
            draw_mlp(mlp_model, input_sample,
                     epoch=epoch, loss_val=loss.value,
                     save_path=frame_path, show=False, figsize=figsize)
            frame_paths.append(frame_path)
            print(f'Epoch {epoch}/{epochs}  Loss: {loss.value:.2f}')

        # --- update weights ---
        for w in mlp_model.weights:
            w.value = w.value - (w.grad * lr)

    # final frame
    loss_final = microg_cls(0)
    for sample in dataset:
        temp = mlp_model.input(sample[0])
        t2 = temp[0] - sample[1]
        t2 = t2 * t2
        loss_final += t2

    frame_path = os.path.join(frames_dir, f'frame_final.png')
    draw_mlp(mlp_model, input_sample,
             epoch=epochs, loss_val=loss_final.value,
             save_path=frame_path, show=False, figsize=figsize)
    frame_paths.append(frame_path)

    # --- stitch into gif ---
    try:
        from PIL import Image
        imgs = [Image.open(fp) for fp in frame_paths]
        duration = int(1000 / fps)
        imgs[0].save(filename, save_all=True, append_images=imgs[1:],
                     duration=duration, loop=0)
        print(f'\nGIF saved to: {filename}')
    except ImportError:
        print('\nPillow not installed. Frames saved as PNGs in _nn_frames/')
        print(f'Install with: pip install Pillow')
        print(f'Then run:')
        print(f'  from PIL import Image')
        print(f'  imgs = [Image.open(f) for f in sorted(glob.glob("_nn_frames/*.png"))]')
        print(f'  imgs[0].save("{filename}", save_all=True, append_images=imgs[1:], duration=333, loop=0)')

    # cleanup
    try:
        for fp in frame_paths:
            os.remove(fp)
        os.rmdir(frames_dir)
    except Exception:
        pass

    return frame_paths
