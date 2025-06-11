import matplotlib.pyplot as plt
import json
from sklearn.manifold import MDS
import numpy as np
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
from datetime import datetime
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from engine import search 
import importlib.util
import sys
import os
from matplotlib.patches import Rectangle
from matplotlib.markers import MarkerStyle
from algsim import bfs
import random

# Define marker paths for circle and triangle
circle_path = MarkerStyle('o').get_path().transformed(MarkerStyle('o').get_transform())
triangle_path = MarkerStyle('^').get_path().transformed(MarkerStyle('^').get_transform())

with open("similarity_matrix.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert similarity matrix to distance matrix
S = np.array(data)
S_max = S.max()
D = S_max - S  # Higher similarity = smaller distance

# Normalize the distance matrix using min-max scaling
D_min = D.min()
D_max = D.max()
D_norm = (D - D_min) / (D_max - D_min)

# Load sorted_videos.json (only titles and ids needed)
with open("filtered_videos.json", "r", encoding="utf-8") as f:
    filtered_videos = json.load(f)
    # Check if filtered_videos is truncated
    print(f"Loaded {len(filtered_videos)} videos from filtered_videos.json")
    video_titles = [video["title"] for video in filtered_videos]
    video_ids = [video["id"] for video in filtered_videos]
    categorys = [video["category"] for video in filtered_videos]

# Load categories from categories.json
with open("categories.json", "r", encoding="utf-8") as f:
    category_names= json.load(f)

# Perform MDS to reduce to 2D using normalized distances
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
points_2d = mds.fit_transform(D_norm)

# Add jitter to the 2D points to reduce overlap
jitter_strength = 0.05 * (points_2d.max(axis=0) - points_2d.min(axis=0))
jitter = np.random.normal(0, jitter_strength, points_2d.shape)
points_2d_jittered = points_2d + jitter

# Ensure the number of videos matches the number of points
num_points = points_2d_jittered.shape[0]
if len(filtered_videos) != num_points:
    raise ValueError(f"Mismatch: filtered_videos has {len(filtered_videos)} items, but MDS produced {num_points} points. Please ensure your data and similarity matrix are aligned.")

# Get view counts and normalize for dot size
view_counts = np.array([int(video['views']) for video in filtered_videos])
# Normalize view counts to a reasonable range for dot size (e.g., 30-300)
min_size, max_size = 30, 300
view_counts_norm = (view_counts - view_counts.min()) / (view_counts.max() - view_counts.min())
dot_sizes = min_size + view_counts_norm * (max_size - min_size)

# Parse publish dates for color mapping
publish_dates = [datetime.fromisoformat(video["publishedAt"].replace('Z', '')) for video in filtered_videos]
min_date = min(publish_dates)
max_date = max(publish_dates)

# Normalize publish dates to [0, 1] for colormap
publish_dates_norm = [(d - min_date).total_seconds() / (max_date - min_date).total_seconds() for d in publish_dates]

# Use a colormap: older = red, newer = blue
cmap = plt.get_cmap('coolwarm')
colors = [cmap(1 - v) for v in publish_dates_norm]  # 1-v: red for old, blue for new
original_colors = colors.copy()

# Ensure dot_sizes is 1D and matches the number of points
if dot_sizes.shape[0] != points_2d_jittered.shape[0]:
    raise ValueError(f"dot_sizes length {dot_sizes.shape[0]} does not match number of points {points_2d_jittered.shape[0]}")

# Plot the 2D projection with hover tooltips for titles and dot size by views
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.25, bottom=0.18)  # Make space for category list and slider
sc = ax.scatter(points_2d_jittered[:, 0], points_2d_jittered[:, 1], alpha=0.6, s=dot_sizes.flatten(), c=colors)
plt.title("2D Visualization of Points via MDS (Similarity to Distance)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()

# Assign a unique color to each category using a colormap
import matplotlib.cm as cm

category_to_color = {}
# Fix for Matplotlib >=3.7: get_cmap does not take a second argument for number of colors
# Use ListedColormap if you want to limit the number of colors, or just use get_cmap('tab20')
cmap = plt.colormaps.get_cmap('tab20')
for i, cat in enumerate(category_names):
    category_to_color[cat] = cmap(i % cmap.N)

# Add a toggle button to switch between date and category coloring
ax_toggle = plt.axes([0.02, 0.85, 0.18, 0.08])
toggle_button = Button(ax_toggle, 'Display Category', color='lightgray', hovercolor='0.85')

color_mode = {'mode': 'date'}  # mutable for closure

# Track current mode (move this up before update_dot_markers)
current_mode = {'mode': 'Search'}

# Define search_result_indices before any function that uses it
search_result_indices = set()  # Indices of current search results
hovered_search_idx = [None]   # Mutable container for currently hovered search result index

# --- Ensure update_dot_markers is defined before it is used ---
alg_done_line = [None]  # Store the red line for Alg mode

def update_dot_markers():
    paths = [circle_path for _ in range(len(filtered_videos))]
    facecolors = list(colors)
    # Remove previous Alg line if present
    if alg_done_line[0] is not None:
        try:
            alg_done_line[0].remove()
        except Exception:
            pass
        alg_done_line[0] = None
    # Alg mode: done videos as yellow triangles, start as black triangle
    if current_mode['mode'] == 'Alg':
        # Draw all done as yellow triangles
        done_list = []
        if alg_done[0] is not None:
            # If alg_done[0] is a set, sort by insertion order if possible, else by index
            if isinstance(alg_done[0], list):
                done_list = alg_done[0]
            elif isinstance(alg_done[0], set):
                # Try to preserve order if possible (Python 3.7+ set is ordered by insertion)
                done_list = list(alg_done[0])
            else:
                done_list = list(alg_done[0])
            for idx in done_list:
                paths[idx] = triangle_path
                facecolors[idx] = 'yellow'
        # Draw start as black triangle (overrides yellow if overlap)
        if alg_start[0] is not None:
            paths[alg_start[0]] = triangle_path
            facecolors[alg_start[0]] = 'black'
        # Draw red line connecting yellow triangles in order
        if len(done_list) > 1:
            xy = points_2d_jittered[done_list+[alg_start[0]]]
            alg_done_line[0], = ax.plot(xy[:,0], xy[:,1], color='red', linewidth=2, zorder=5)
    else:
        # Search mode: search results as black triangles, hovered as red
        for idx in search_result_indices:
            paths[idx] = triangle_path
            if hovered_search_idx[0] == idx:
                facecolors[idx] = 'red'
            else:
                facecolors[idx] = 'black'
    sc.set_paths(paths)
    sc.set_color(facecolors)
    fig.canvas.draw_idle()

# Function to update scatter plot colors
def set_colors_by_category():
    n = min(len(categorys), len(colors))
    for i in range(n):
        colors[i] = category_to_color.get(categorys[i], (0.5, 0.5, 0.5, 1))  # fallback gray
    sc.set_color(colors[:n])
    update_dot_markers()
    fig.canvas.draw_idle()

def set_colors_by_date():
    n = min(len(original_colors), len(colors))
    for i in range(n):
        colors[i] = original_colors[i]
    sc.set_color(colors[:n])
    update_dot_markers()
    fig.canvas.draw_idle()

def toggle_colors(event):
    if color_mode['mode'] == 'date':
        set_colors_by_category()
        color_mode['mode'] = 'category'
        toggle_button.label.set_text('Display Date')
        remove_date_colorbar()
        draw_category_legend()
    else:
        set_colors_by_date()
        color_mode['mode'] = 'date'
        toggle_button.label.set_text('Display Category')
        remove_category_legend()
        colorbar_ax[0] = draw_date_colorbar()

# Draw category color legend as a list on the left side (sorted by frequency)
def draw_category_legend():
    legend_y_start = 0.8
    legend_y_step = 0.03
    for i, cat in enumerate(category_names):
        y = legend_y_start - i * legend_y_step
        if y < 0.05:
            break  # Don't draw off the figure
        ax.add_patch(mpatches.Rectangle((-0.08, y-0.012), 0.025, 0.025, color=category_to_color[cat], transform=ax.transAxes, clip_on=False, zorder=10))
        ax.text(-0.04, y, cat, color='black', fontsize=9, ha='left', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), zorder=10)

def remove_category_legend():
    # Remove all Rectangle and Text objects in the legend area
    for artist in ax.get_children():
        if isinstance(artist, mpatches.Rectangle) and artist.get_x() == -0.08:
            artist.remove()
        if isinstance(artist, plt.Text) and artist.get_position()[0] == -0.04:
            artist.remove()

# Draw a colorbar/key for the blue-red date feature
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
import matplotlib.dates as mdates

def draw_date_colorbar():
    # Place the colorbar at the bottom, below the plot
    cbar_ax = fig.add_axes([0.25, 0.12, 0.5, 0.025])  # [left, bottom, width, height]
    norm = Normalize(vmin=mdates.date2num(min_date), vmax=mdates.date2num(max_date))
    cb = ColorbarBase(cbar_ax, cmap=plt.get_cmap('coolwarm'), norm=norm, orientation='horizontal')
    cbar_ax.xaxis.set_ticks_position('bottom')
    cbar_ax.set_xlabel('Publish Date', fontsize=9)
    # Set ticks at the ends for earliest and latest dates
    cbar_ax.set_xticks([mdates.date2num(min_date), mdates.date2num(max_date)])
    cbar_ax.set_xticklabels([
        min_date.strftime('%Y-%m-%d'),
        max_date.strftime('%Y-%m-%d')
    ], fontsize=9)
    return cbar_ax

# Fix: Don't remove all axes, just the colorbar
colorbar_ax = [None]  # mutable reference

def remove_date_colorbar():
    if colorbar_ax[0] is not None:
        colorbar_ax[0].remove()
        colorbar_ax[0] = None

# Initial state: show date colorbar, hide category legend
colorbar_ax[0] = draw_date_colorbar()

# Fix: Don't redraw the button, just toggle legends
# Update toggle_colors to show/hide legends appropriately
def toggle_colors(event):
    if color_mode['mode'] == 'date':
        set_colors_by_category()
        color_mode['mode'] = 'category'
        toggle_button.label.set_text('Display Date')
        remove_date_colorbar()
        draw_category_legend()
    else:
        set_colors_by_date()
        color_mode['mode'] = 'date'
        toggle_button.label.set_text('Display Category')
        remove_category_legend()
        colorbar_ax[0] = draw_date_colorbar()

# Set initial color mode
set_colors_by_date()
toggle_button.on_clicked(toggle_colors)

# Draw category color legend as a list on the left side (sorted by frequency)
def draw_category_legend():
    legend_y_start = 0.8
    legend_y_step = 0.03
    for i, cat in enumerate(category_names):
        y = legend_y_start - i * legend_y_step
        if y < 0.05:
            break  # Don't draw off the figure
        # Move the color box and text further to the right
        ax.add_patch(mpatches.Rectangle((-0.08, y-0.012), 0.025, 0.025, color=category_to_color[cat], transform=ax.transAxes, clip_on=False, zorder=10))
        ax.text(-0.04, y, cat, color='black', fontsize=9, ha='left', va='center', transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), zorder=10)

annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

# Add a static info box in the bottom left for video details
info_box = ax.text(0.02, 0.02, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=10,
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'), zorder=20)
info_box.set_visible(False)

# Function to update annotation and info box
# Tooltip: only title. Info box: views, date, category.
def update_annot(ind):
    idx = ind["ind"][0]
    pos = sc.get_offsets()[idx]
    annot.xy = pos
    # Tooltip: only title
    annot.set_text(video_titles[idx])
    annot.get_bbox_patch().set_alpha(0.8)
    # Info box: more details
    views = filtered_videos[idx]["views"]
    pubdate = filtered_videos[idx]["publishedAt"]
    category = filtered_videos[idx]["category"]
    info_text = f"Views: {views}\nDate: {pubdate}\nCategory: {category}"
    info_box.set_text(info_text)
    info_box.set_visible(True)

# Hover event
# Hide info box if not hovering
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont and len(ind["ind"]):
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                info_box.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

# Update draw_lines to use all points (no filtering)
def draw_lines(ax, points, D, threshold):
    lines = []
    n = points.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] < threshold:
                line, = ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='gray', alpha=0.3, linewidth=1, zorder=1)
                lines.append(line)
    return lines

# Initial threshold (e.g., 0.1 of normalized distance)
init_threshold = 0.1
lines = draw_lines(ax, points_2d_jittered, D_norm, init_threshold)

# Add a slider for threshold adjustment
slider_ax = plt.axes([0.15, 0.01, 0.7, 0.03])
threshold_slider = Slider(slider_ax, 'Distance Threshold', 0.0, 1.0, valinit=init_threshold, valstep=0.01)

# Connect the slider to update lines
def update_lines(val):
    threshold = threshold_slider.val
    for line in lines:
        line.remove()
    lines.clear()
    lines.extend(draw_lines(ax, points_2d_jittered, D_norm, threshold))
    fig.canvas.draw_idle()

threshold_slider.on_changed(update_lines)

# Add a TextBox widget for search input
axbox = plt.axes([0.78, 0.92, 0.2, 0.05])  # [left, bottom, width, height]
text_box = TextBox(axbox, 'Search:', initial="")

# Add a clear button to the right of the search bar
axclear = plt.axes([0.99, 0.92, 0.08, 0.05])  # [left, bottom, width, height]
clear_button = Button(axclear, 'Clear', color='lightgray', hovercolor='0.85')

# Add a list of 10 titles below the search bar
search_result_texts = []
search_result_boxes = []
search_result_y_start = 0.86
search_result_y_step = 0.035
search_result_indices = set()  # Indices of current search results

# Helper to clear previous search results and reset dot markers/colors

def clear_search_results():
    for t in search_result_texts:
        t.remove()
    search_result_texts.clear()
    for b in search_result_boxes:
        b.set_alpha(0.0)
    search_result_boxes.clear()
    search_result_indices.clear()
    hovered_search_idx[0] = None  # Clear hovered index
    # Reset all markers to circles and restore color by mode
    sc.set_paths([circle_path] * len(filtered_videos))
    sc.set_color(colors)
    fig.canvas.draw_idle()

# Callback for search
from matplotlib.patches import Rectangle

def truncate_title(title, maxlen=50):
    return title if len(title) <= maxlen else title[:maxlen-3] + '...'

def submit_search(text):
    ids = search(text)
    print(ids)  # Debug: print search results
    clear_search_results()
    print(f"Filtered videos count: {len(filtered_videos)}")
    id_to_idx = {video['id']: i for i, video in enumerate(filtered_videos) if video['id'] in ids}
    print(id_to_idx)
    indices = []
    for i, vid in enumerate(ids):
        idx = id_to_idx.get(vid)
        if idx is not None:
            indices.append(idx)
            y = search_result_y_start - i * search_result_y_step
            box = Rectangle((0.78, y-0.01), 0.2, 0.03, transform=fig.transFigure, color='white', alpha=0.0, zorder=30)
            fig.patches.append(box)
            search_result_boxes.append(box)
            t = fig.text(0.79, y, truncate_title(filtered_videos[idx]['title']), fontsize=10, ha='left', va='bottom', zorder=31, picker=True, family='monospace')
            search_result_texts.append(t)
            t._search_idx = idx
        else:
            print(f"Video ID {vid} not found in filtered_videos")
    # Set all search result dots to yellow triangles
    search_result_indices.clear()
    search_result_indices.update(indices)
    update_dot_markers()
    fig.canvas.draw_idle()

text_box.on_submit(submit_search)

# Clear search box and results
def clear_search(event=None):
    text_box.set_val("")  # Clear the search box text
    clear_search_results()  # Remove search results
    info_box.set_visible(False)  # Hide info box
    fig.canvas.draw_idle()

clear_button.on_clicked(clear_search)

# Hover event for search results
# (We use a separate event for figure-level text, since scatter hover is on axes)
def on_figure_motion(event):
    # Check if mouse is over any search result text
    found = False
    for t in search_result_texts:
        bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
        if bbox.contains(event.x, event.y):
            idx = getattr(t, '_search_idx', None)
            if idx is not None:
                # Update info box as if hovering over the corresponding video
                views = filtered_videos[idx]["views"]
                pubdate = filtered_videos[idx]["publishedAt"]
                category = filtered_videos[idx]["category"]
                info_text = f"Views: {views}\nDate: {pubdate}\nCategory: {category}"
                info_box.set_text(info_text)
                info_box.set_visible(True)
                # Set hovered search idx and update markers
                if hovered_search_idx[0] != idx:
                    hovered_search_idx[0] = idx
                    update_dot_markers()
                found = True
                break
    if not found:
        # Only hide if not hovering over a scatter point either
        if not annot.get_visible():
            info_box.set_visible(False)
        # Clear hovered search idx and update markers if needed
        if hovered_search_idx[0] is not None:
            hovered_search_idx[0] = None
            update_dot_markers()
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_figure_motion)

# Add a mode toggle menu (Search/Alg)
ax_mode = plt.axes([0.02, 0.75, 0.18, 0.08])
mode_radio = RadioButtons(ax_mode, ['Search', 'Alg'], active=0)

# --- Alg mode state and UI ---
alg_done = [None]
alg_start = [None]
alg_message_box = [None]
alg_input_box = [None]

# Helper to show/hide search widgets

def set_search_widgets_visible(visible):
    axbox.set_visible(visible)
    axclear.set_visible(visible)
    for t in search_result_texts:
        t.set_visible(visible)
    for b in search_result_boxes:
        b.set_visible(visible)
    if not visible:
        clear_search_results()
        info_box.set_visible(False)
        # Also clear Alg UI and line when leaving Search
        if alg_message_box[0] is not None:
            alg_message_box[0].remove()
            alg_message_box[0] = None
        if alg_input_box[0] is not None:
            alg_input_box[0].disconnect_events()
            if hasattr(alg_input_box[0], 'ax'):
                alg_input_box[0].ax.set_visible(False)
            if hasattr(alg_input_box[0], 'label'):
                alg_input_box[0].label.set_visible(False)
            if hasattr(alg_input_box[0], 'text_disp'):
                alg_input_box[0].text_disp.set_visible(False)
            alg_input_box[0] = None
        if alg_done_line[0] is not None:
            try:
                alg_done_line[0].remove()
            except Exception:
                pass
            alg_done_line[0] = None
        # Reset all markers
        sc.set_paths([circle_path] * len(filtered_videos))
        sc.set_color(colors)
        fig.canvas.draw_idle()

# Helper to show/hide alg widgets

def set_alg_widgets_visible(visible):
    # Remove old message/input if present
    if alg_message_box[0] is not None:
        alg_message_box[0].remove()
        alg_message_box[0] = None
    if alg_input_box[0] is not None:
        alg_input_box[0].disconnect_events()
        if hasattr(alg_input_box[0], 'ax'):
            alg_input_box[0].ax.set_visible(False)
        if hasattr(alg_input_box[0], 'label'):
            alg_input_box[0].label.set_visible(False)
        if hasattr(alg_input_box[0], 'text_disp'):
            alg_input_box[0].text_disp.set_visible(False)
        alg_input_box[0] = None
    # Remove Alg line if present
    if alg_done_line[0] is not None:
        try:
            alg_done_line[0].remove()
        except Exception:
            pass
        alg_done_line[0] = None
    # Also clear search UI and info box when leaving Alg
    if not visible:
        clear_search_results()
        info_box.set_visible(False)
        # Reset all markers
        sc.set_paths([circle_path] * len(filtered_videos))
        sc.set_color(colors)
        fig.canvas.draw_idle()
    if visible:
        # Initialize state
        alg_done[0] = []
        alg_start[0] = random.randint(0, len(video_titles)-1)
        # Show message and input box
        msg = f"What would you rate: {video_titles[alg_start[0]]}?"
        alg_message_box[0] = fig.text(0.3, 0.95, msg, fontsize=12, ha='left', va='top', zorder=40, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.4'))
        ax_alg_input = plt.axes([0.3, 0.91, 0.2, 0.04])
        alg_input_box[0] = TextBox(ax_alg_input, 'Rating:', initial="")
        def on_alg_submit(text):
            try:
                rating = int(text)
                if rating<1 or rating>10:
                    return
            except Exception:
                return
            # Call bfs and update state
            done, start = bfs(alg_done[0], alg_start[0], rating)
            alg_done[0] = done
            alg_start[0] = start
            update_dot_markers()  # <-- update markers immediately after state change
            if start is not None:
                alg_message_box[0].set_text(f"What would you rate (1-10): {video_titles[start]}?")
                alg_input_box[0].set_val("")
            else:
                alg_message_box[0].set_text("Algorithm finished.")
                alg_input_box[0].set_visible(False)
            fig.canvas.draw_idle()
        alg_input_box[0].on_submit(on_alg_submit)
    fig.canvas.draw_idle()

# Callback for mode toggle

def on_mode_change(label):
    if label == 'Search':
        set_search_widgets_visible(True)
        set_alg_widgets_visible(False)
        current_mode['mode'] = 'Search'
    else:
        set_search_widgets_visible(False)
        set_alg_widgets_visible(True)
        current_mode['mode'] = 'Alg'

mode_radio.on_clicked(on_mode_change)

# Set initial visibility
set_search_widgets_visible(True)
set_alg_widgets_visible(False)

plt.show()