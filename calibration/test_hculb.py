from typing import List
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import scipy.optimize
from tqdm import tqdm
import math

import cameras
import debug
import file_io
import lights
import optimize
import patterns
from renderers import Basic
import utils
import config_globals
from config_globals import ENDOSCOPE_LIGHT_DIAMETERS_M, FRAME_COUNT, OPTIMIZE_LIGHT_AREA_SAMPLING_LEVEL, FRAME_MAX_DISTANCE_TO_PATTERN_M, FRAME_MIN_DISTANCE_TO_PATTERN_M, OPTIMIZE_LIGHT_INITIAL_D, OPTIMIZE_LIGHT_INITIAL_MU, RESULTS_NAME, OPTIMIZE_LIGHT, \
    SAMPLING_STRATEGY, SAMPLING_ARGUMENTS, HUBER_SIGMA_EST, IMREAD_GAUSSIANBLUR_KSIZE, \
    ENDOSCOPE_DISTAL_END_IMAGE, ENDOSCOPE_DISTAL_END_IMAGE_CENTER, \
    ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M, ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_PX, \
    ENDOSCOPE_LIGHT_CENTERS

import argparse

parser = argparse.ArgumentParser(
    description='Calibrate endoscope\'s photometry.')
parser.add_argument('-b', '--backup', dest='backup', action='store',
                    default='', help='backup name')
parser.add_argument('-p', '--path', dest='path', action='store',
                    default='', help='path to calibrations folder')
parser.add_argument('-s', '--sequence', dest='sequence', action='store',
                    default='', help='sequence name')
args = parser.parse_args()

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

sequence = args.sequence
path = os.path.join(args.path, sequence)
data = os.path.join(path, f'{sequence}_frames')

frame_ids, frame_poses = file_io.read_trajectory(
    os.path.join(path, f'{sequence}_poses.csv'))
frame_ids = sorted(frame_ids, key=lambda t: frame_poses[t][2, 3])
min_z = FRAME_MIN_DISTANCE_TO_PATTERN_M or frame_poses[frame_ids[0]][2, 3]
max_z = FRAME_MAX_DISTANCE_TO_PATTERN_M or frame_poses[frame_ids[-1]][2, 3]
frame_count = FRAME_COUNT if isinstance(FRAME_COUNT, list) else [1,] * FRAME_COUNT
lim_z = np.linspace(min_z, max_z, len(frame_count) + 1)
selected_frame_ids = []
lens = []
for fc, l, r in zip(frame_count, lim_z[:-1], lim_z[1:]):
    frame_ids_in_range = list(filter(lambda t: abs(frame_poses[t][2, 3]) >= l and
                                     abs(frame_poses[t][2, 3]) <= r, frame_ids))
    size = min(fc, len(frame_ids_in_range))
    lens.append(len(frame_ids_in_range))
    frame_ids_chosen = \
        np.random.RandomState(seed=0).choice(frame_ids_in_range,
                                             size,
                                             replace=False)
    selected_frame_ids += frame_ids_chosen.tolist()

# DEBUG: Histogram of heights over pattern
data_hist_z = np.array([-frame_poses[t][2, 3] for t in frame_ids])

# Sort selected frames by distance to pattern
frame_ids = sorted(selected_frame_ids, key=lambda t: frame_poses[t][2, 3])

pattern = patterns.Factory.fromXML(os.path.join(path, f'{sequence}_pattern.xml'))
T_wp = pattern.T_wp
T_pw = np.linalg.inv(T_wp)

camera = cameras.Factory.fromXML(
    os.path.join(path, f'{sequence}_geometrical.xml'),
    os.path.join(path, f'{sequence}_mask.png'))

def get_light_source(name: str, **kwargs) -> lights.Base:
    if name == 'NFSLS':
        return lights.NormalizedFixedSpotLightSource(**kwargs)
    elif name == 'NZFSLS':
        return lights.NormalizedZFixedSpotLightSource(**kwargs)
    elif name == 'NFPLS':
        return lights.NormalizedFixedPointLightSource(**kwargs)
    elif name == 'NSLS':
        return lights.NormalizedSpotLightSource(**kwargs)
    elif name == 'NSLS2D':
        return lights.NormalizedSpotLightSource2D(**kwargs)
    elif name == 'ZFSLS':
        return lights.ZFixedSpotLightSource(**kwargs)
    elif name == 'FSLS':
        return lights.FixedSpotLightSource(**kwargs)
    elif name == 'FPLS':
        return lights.FixedPointLightSource(**kwargs)
    elif name == 'SLS':
        return lights.SpotLightSource(**kwargs)
    elif name == 'SLS2D':
        return lights.SpotLightSource2D(**kwargs)
    else:
        raise ValueError(f'Invalid light source model: {name}')

sources = []
for i in range(len(OPTIMIZE_LIGHT)):
    source = get_light_source(name=OPTIMIZE_LIGHT[i],
                                mu=OPTIMIZE_LIGHT_INITIAL_MU[i],
                                P=ENDOSCOPE_LIGHT_CENTERS[i],
                                D=np.array(OPTIMIZE_LIGHT_INITIAL_D[i]),
                                radius=ENDOSCOPE_LIGHT_DIAMETERS_M[i] / 2,
                                area_sampling_resolution=OPTIMIZE_LIGHT_AREA_SAMPLING_LEVEL[i])
    sources.append(source)

def debugSourcesOnEndoscope(axs: plt.Axes,
                            sources: List[lights.Base]):
    debug.sourcesOnEndoscope(axs, sources,
                             ENDOSCOPE_DISTAL_END_IMAGE,
                             ENDOSCOPE_DISTAL_END_IMAGE_CENTER,
                             ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_PX,
                             ENDOSCOPE_DISTAL_END_OUTER_DIAMETER_M)
    axs.legend()

# DEBUG: plot all light sources in the endoscope
fig, axs = plt.subplots(1, 1)
axs.set_title('Init')
debugSourcesOnEndoscope(axs, sources)

renderer = Basic(camera, sources, pattern)

# debug.checkIfThreeLightsAreWorthIt(renderer, frame_ids, frame_poses, T_wp, data, mode='video')

if SAMPLING_STRATEGY == 'PATTERN':
    x_p = pattern.sample(**SAMPLING_ARGUMENTS)
    # DEBUG: plot all sample points in the Vicalib pattern
    # debug.plotPatternSample(pattern, **SAMPLING_ARGUMENTS)
else:
    raise ValueError(f'Invalid SAMPLING_STRATEGY: \'{SAMPLING_STRATEGY}\'')
plt.pause(1)

n_frames = len(frame_ids)
fig1, axs1 = debug.getSquaredGrid(n_frames, 'I')
fig, axs = debug.getSquaredGrid(n_frames, 'I')
fig2, axs2 = debug.getSquaredGrid(n_frames, 'I (pseudo-color)')

# frames = []
x_valid_list = []
x_w_list = []
x_uv_list = []
T_wc_list = []
I_gt_list = []
gain_list = []
data_hist_rays = []
for i in tqdm(range(n_frames), 'Loading data'):
    frame_id = frame_ids[i]
    img_bgr = cv2.imread(os.path.join(data, f'{frame_id:06d}.png'))
    img_bgr = cv2.GaussianBlur(
        img_bgr, (IMREAD_GAUSSIANBLUR_KSIZE, IMREAD_GAUSSIANBLUR_KSIZE), 0)
    img_gray = (np.mean(img_bgr, axis=2) / 255)
    img_gray = img_gray.astype(np.float32)[:, :, np.newaxis]
    # img_gray_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    T_wc = frame_poses[frame_id]
    T_cw = np.linalg.inv(T_wc)

    # DEBUG: check expected localion of light and specular highlight
    # debug.lightPositionFromSpecularHighlight(camera, pattern, T_wc, T_wp, img_gray_3ch)

    x_w = T_wp @ x_p
    if SAMPLING_STRATEGY == 'PATTERN' and 'max_theta' in SAMPLING_ARGUMENTS:
        wi_w = (T_wc[0:4, 3:4] - x_w)
        wi_w = wi_w / np.linalg.norm(wi_w, axis=0)[np.newaxis, :]
        cos_th = (T_wp @ renderer.pattern.n).T @ wi_w
        x_w = x_w[:, cos_th[0] > np.cos(
            np.radians(SAMPLING_ARGUMENTS['max_theta']))]
    if SAMPLING_STRATEGY == 'PATTERN' and 'max_alpha' in SAMPLING_ARGUMENTS:
        wi_w = (T_wc[0:4, 3:4] - x_w)
        wi_w = wi_w / np.linalg.norm(wi_w, axis=0)[np.newaxis, :]
        wi_c = T_cw @ wi_w
        cos_alpha = renderer.camera.z.T @ -wi_c
        x_w = x_w[:, cos_alpha[0] > np.cos(
            np.radians(SAMPLING_ARGUMENTS['max_alpha']))]
    x_uv, x_valid = camera.project(T_cw @ x_w)
    if SAMPLING_STRATEGY == 'PATTERN' and 'max_grid_count' in SAMPLING_ARGUMENTS:
        N_ROWS, N_COLS, MAX_COUNT = SAMPLING_ARGUMENTS['max_grid_count']
        h = img_bgr.shape[0]
        w = img_bgr.shape[1]
        for r in range(N_ROWS):
            for c in range(N_COLS):
                gt_lim = [[int(w / N_COLS * c)],
                          [int(h / N_ROWS * r)]]
                lt_lim = [[int(w / N_COLS * (c + 1))],
                          [int(h / N_ROWS * (r + 1))]]
                gt = np.all(x_uv[0:2, :] >= gt_lim, axis=0)
                lt = np.all(x_uv[0:2, :] < lt_lim, axis=0)
                is_in_grid = np.logical_and(gt, lt)
                in_grid_count = np.sum(is_in_grid)
                if in_grid_count > MAX_COUNT:
                    in_grid_idxs = np.where(is_in_grid)[0]
                    idxs_to_be_deleted = \
                        np.random.RandomState(seed=0).choice(in_grid_idxs,
                                                             in_grid_count - MAX_COUNT,
                                                             replace=False)
                    x_w = np.delete(x_w, idxs_to_be_deleted, axis=1)
                    x_uv = np.delete(x_uv, idxs_to_be_deleted, axis=1)
                    x_valid = np.delete(x_valid, idxs_to_be_deleted, axis=0)

    # DEBUG: histogram of distances to pattern points
    dists = np.linalg.norm(x_w[0:3, x_valid] - T_wc[0:3, 3:4], axis=0)
    data_hist_rays.append(dists)

    I_gt, _ = utils.interpolate(img_gray, x_uv[:, x_valid])
    I_hat, _ = renderer.x_w(x_w[:, x_valid], T_wc, T_wp, 1.0)
    I_hat = np.mean(I_hat, axis=0)[:, np.newaxis]
    gain = np.median(renderer.camera.inv_response(I_gt) /
                     renderer.camera.inv_response(I_hat))
    gain = min(gain, 1/np.max(renderer.camera.inv_response(I_hat)))

    plt_gray = axs1[i].imshow(np.mean(img_gray, axis=2), cmap='gray', vmin=0, vmax=1)
    plt_pseudo_color = axs2[i].imshow(np.mean(img_gray, axis=2), cmap='tab20', vmin=0, vmax=1)
    plt_gray_alt = axs[i].imshow(np.mean(img_gray, axis=2), cmap='gray', vmin=0, vmax=1)
    axs1[i].axis('off')
    axs2[i].axis('off')
    axs[i].axis('off')
    axs[i].scatter(x_uv[0, x_valid], x_uv[1, x_valid],
                   s=[20 if v < 1e-6 else 0.05 for v in I_gt],
                   c=['r' if v < 1e-6 else 'g' for v in I_gt])

    # frames.append(img_gray)
    x_valid_list.append(x_valid)
    x_w_list.append(x_w)
    x_uv_list.append(x_uv)
    T_wc_list.append(T_wc)
    I_gt_list.append(I_gt)
    gain_list.append(gain)
axs1[0].cax.colorbar(plt_gray)
axs1[0].cax.toggle_label(True)
axs2[0].cax.colorbar(plt_pseudo_color)
axs2[0].cax.toggle_label(True)
axs[0].cax.colorbar(plt_gray_alt)
axs[0].cax.toggle_label(True)
fig.tight_layout()
fig1.tight_layout()
fig2.tight_layout()

# Split train and test sets
TRAIN_SET_PARTITION = 0.8
random_permutation = np.random.RandomState(seed=0).permutation(n_frames)
skp = int(1 // (1 - TRAIN_SET_PARTITION))
test_idx = list(range(n_frames))[0::skp]
train_idx = [idx for idx in range(n_frames) if idx not in test_idx]
n_frames_train = len(train_idx)
n_frames_test = len(test_idx)
x_valid_list_train = [x_valid_list[i] for i in train_idx]
x_valid_list_test = [x_valid_list[i] for i in test_idx]
x_w_list_train = [x_w_list[i] for i in train_idx]
x_w_list_test = [x_w_list[i] for i in test_idx]
T_wc_list_train = [T_wc_list[i] for i in train_idx]
T_wc_list_test = [T_wc_list[i] for i in test_idx]
I_gt_list_train = [I_gt_list[i] for i in train_idx]
I_gt_list_test = [I_gt_list[i] for i in test_idx]
gain_list_train = [gain_list[i] for i in train_idx]
gain_list_test = [gain_list[i] for i in test_idx]

for ax in [axs, axs1, axs2]:
    for i in range(n_frames):
        ax[i].set_title(f'{frame_ids[i]:06d}',
                        pad=0.0,
                        fontdict={
                            'fontsize': 'small',
                            'color': 'red' if i in test_idx else 'black'})
plt.draw()
plt.pause(1)

# DEBUG: plot twin histograms of distances to pattern points
data_hist_z_train = np.array([-frame_poses[frame_ids[t]][2, 3] for t in train_idx])
data_hist_z_test = np.array([-frame_poses[frame_ids[t]][2, 3] for t in test_idx])

data_hist_z = np.array(data_hist_z) * 1000  # to mm
data_hist_rays = np.concatenate(data_hist_rays) * 1000  # to mm
data_hist_z_train = np.array(data_hist_z_train) * 1000  # to mm
data_hist_z_test = np.array(data_hist_z_test) * 1000  # to mm
fig, ax1, ax2 = debug.plotTwinHistograms(data_hist_z, data_hist_rays, 
                          bins=30,
                          title='Histograms of distances',
                          xlabel='Distance (mm)',
                          ylabel1='FROM all cameras TO plane (# frames)',
                          ylabel2='FROM visible points TO train/test cameras (# points)')
ax2.plot(data_hist_z_train, np.zeros(len(data_hist_z_train)), 'kx', label='Train cameras')
ax2.plot(data_hist_z_test, np.zeros(len(data_hist_z_test)), 'rx', label='Test cameras')
fig.legend()

fig1, axs1 = debug.getSquaredGrid(n_frames_train, title='I_init (train)')
fig3, axs3 = debug.getSquaredGrid(n_frames_train, title='I_init (train, pseudo-color)')
fig2, axs2 = debug.getSquaredGrid(n_frames_train, title='I_init - I (train)')

for j, i in enumerate(tqdm(train_idx, 'Rendering')):
    render, valid = renderer.full(T_wc_list[i], T_wp, gain_list[i])
    gray = axs1[j].imshow(np.clip(np.mean(render, axis=2), 0, 1), cmap='gray', vmin=0, vmax=1)
    pseudo_color = axs3[j].imshow(np.clip(np.mean(render, axis=2), 0, 1), cmap='tab20', vmin=0, vmax=1)
    for ax in [axs1[j], axs2[j], axs3[j]]:
        ax.axis('off')
        ax.set_title(f'{frame_ids[i]:06d}',
                        pad=0.0,
                        fontdict={
                            'fontsize': 'small',
                            'color': 'red' if i in test_idx else 'black'})
    frame = cv2.imread(os.path.join(data, f'{frame_ids[i]:06d}.png'))
    error = np.mean(render * 255 - frame, axis=2)[:, :, np.newaxis]
    error[np.logical_not(valid)] = 0
    error_map = axs2[j].imshow(error, vmin=-25, vmax=25, cmap='seismic')
axs1[0].cax.colorbar(error_map)
axs1[0].cax.toggle_label(True)
axs2[0].cax.colorbar(error_map)
axs2[0].cax.toggle_label(True)
axs3[0].cax.colorbar(pseudo_color)
axs3[0].cax.toggle_label(True)
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.draw()
plt.pause(1)

op_init = optimize.pack_op(renderer, gain_list_train)
print(f'[INFO] Optimizing {len(op_init)} parameters...')
print(f'[INFO] Optimizing {len(gain_list_train)} gains...')

residuals_train = optimize.fun(op_init, x_w_list_train, x_valid_list_train,
                               T_wc_list_train, T_wp, I_gt_list_train,
                               renderer, gain_list_train)

print('Initial Train MAE:', np.mean(np.abs((residuals_train * 255))))
print('Initial Train Median Abs. Error:',
      np.median(np.abs((residuals_train * 255))))
print('Initial Train RMSE:', np.sqrt(np.mean((residuals_train * 255) ** 2)))


def debugJacSparsity():
    sparsity = optimize.jac_sparsity(I_gt_list_train, op_init)
    param_names = optimize.op_names(renderer, gain_list_train)
    fig, ax = plt.subplots()
    MAX_LINES = 20
    plt.spy(sparsity[::max(1, sparsity.shape[0]//MAX_LINES)], markersize=1)
    ax.set_xticks(np.arange(sparsity.shape[1]) + 0.5)
    ax.set_xticklabels(param_names, rotation=90, ha="right")
    ax.set_ylabel("Residual index")
    ax.set_xlabel("Parameters")
    ax.set_title("Jacobian sparsity pattern")

debugJacSparsity()

opt_time = 0
opt_nfev = 0
if args.backup != '':
    op_final = file_io.load_op(os.path.join(args.backup, 'op_final.txt'))
    print('[INFO] Loaded from backup')
else:
    start = time.time()
    result = scipy.optimize.least_squares(
        optimize.fun,
        op_init,
        bounds=optimize.bounds(renderer, gain_list_train),
        method='trf',
        xtol=1e-15, ftol=1e-15, gtol=1e-15,
        x_scale='jac',
        loss='huber',
        f_scale=2 * HUBER_SIGMA_EST,
        max_nfev=50000,
        jac_sparsity=optimize.jac_sparsity(I_gt_list_train, op_init),
        verbose=2,
        args=(x_w_list_train, x_valid_list_train, T_wc_list_train,
              T_wp, I_gt_list_train, renderer, gain_list_train)
    )
    end = time.time()
    opt_nfev = result.nfev
    opt_time = end - start
    print(f'Optimization finished in {opt_time:.1f} seconds.')

    op_final = result.x

results_folder = os.path.join(SCRIPT_PATH, f'results/{RESULTS_NAME}')
os.makedirs(results_folder)
file_io.save_op(os.path.join(results_folder, 'op_final.txt'), op_final)
file_io.save_config(os.path.join(
    results_folder, 'config_globals.py'), config_globals.get_all())
optimize.unpack_op(op_final, renderer, gain_list_train)

print('camera.vignetting:', renderer.camera.vignetting.params)
print('pattern.brdf:', renderer.pattern.brdf.params)
for i in range(len(renderer.sources)):
    print(f'sources[{i}].[{", ".join(renderer.sources[i].param_names)}]:', renderer.sources[i].params)
print('gain:', gain_list_train)

residuals_train = optimize.fun(op_final, x_w_list_train, x_valid_list_train,
                               T_wc_list_train, T_wp, I_gt_list_train,
                               renderer, gain_list_train)

train_mae = np.mean(np.abs((residuals_train * 255)))
train_median = np.median(np.abs((residuals_train * 255)))
train_rmse = np.sqrt(np.mean((residuals_train * 255) ** 2))
print('Final Train MAE:', train_mae)
print('Final Train Median Abs. Error:', train_median)
print('Final Train RMSE:', train_rmse)

if OPTIMIZE_LIGHT in ['TRI_NFZESLS', 'TRI_NFZSLS']:
    fig, axs = plt.subplots(1, 1)
    axs.set_title('Final')
    debugSourcesOnEndoscope(axs, renderer.sources)

residuals_test, gain_list_test = optimize.eval_test(x_w_list_test, x_valid_list_test,
                                                    T_wc_list_test, T_wp, I_gt_list_test,
                                                    renderer, gain_list_test)

test_mae = np.mean(np.abs((residuals_test * 255)))
test_median = np.median(np.abs((residuals_test * 255)))
test_rmse = np.sqrt(np.mean((residuals_test * 255) ** 2))
print('Final Test MAE:', test_mae)
print('Final Test Median Abs. Error:', test_median)
print('Final Test RMSE:', test_rmse)

plt.figure()
plt.title('Hist. I_hat - I_gt (x_w_test)')
amax = int(np.ceil(max(abs(residuals_test * 255))))
plt.hist(residuals_test * 255, bins=range(-amax, amax), density=True,
         edgecolor='steelblue', linewidth=1, color='powderblue')
debug.plotGaussian(plt.gca(), residuals_test * 255)

fig1, axs1 = debug.getSquaredGrid(n_frames, title='I_hat (final)')
fig3, axs3 = debug.getSquaredGrid(n_frames, title='I_hat (final, pseudo-color)')
fig2, axs2 = debug.getSquaredGrid(n_frames, title='I_hat - I (final)')

error_hist = []
gain_list_hat = np.concatenate([gain_list_test, gain_list_train])
for j, i in enumerate(tqdm(test_idx + train_idx, 'Rendering')):

    render, valid = renderer.full(T_wc_list[i], T_wp, gain_list_hat[j])
    plt_gray = axs1[j].imshow(np.mean(render, axis=2), cmap='gray', vmin=0, vmax=1)
    plt_pseudo_color = axs3[j].imshow(np.mean(render, axis=2), cmap='tab20', vmin=0, vmax=1)
    for ax in [axs1[j], axs2[j], axs3[j]]:
        ax.axis('off')
        ax.set_title(f'{frame_ids[i]:06d}',
                        pad=0.0,
                        fontdict={
                            'fontsize': 'small',
                            'color': 'red' if i in test_idx else 'black'})
    frame = cv2.imread(os.path.join(data, f'{frame_ids[i]:06d}.png'))
    error = np.mean(render * 255 - frame, axis=2)[:, :, np.newaxis]
    error_hist += [error[valid]]
    error[np.logical_not(valid)] = 0
    error_map = axs2[j].imshow(error, vmin=-25, vmax=25, cmap='seismic')
    T_wc = T_wc_list[i]
    T_cw = np.linalg.inv(T_wc)
    x2 = T_wc[:, 3:4]
    x2[2, 0] = 0  # perpendicular camera to plane
    uv2, valid = renderer.camera.project(T_cw @ x2)
    if valid.all():
        axs2[j].plot(uv2[0, 0], uv2[1, 0], 'kx')
    axs2[j].axis('off')
    axs2[j].set_title(f'{frame_ids[i]:06d}',
                      fontdict={
                          'fontsize': 'small',
                          'color': 'red' if i in test_idx else 'black'})
    x_uv = x_uv_list[i]
    x_valid = x_valid_list[i]
    I_gt = I_gt_list[i]
    axs2[j].scatter(x_uv[0, x_valid], x_uv[1, x_valid],
                s=[20 if v < 1e-6 else 0.05 for v in I_gt],
                c=['r' if v < 1e-6 else 'g' for v in I_gt])

axs1[0].cax.colorbar(plt_gray)
axs1[0].cax.toggle_label(True)
axs2[0].cax.colorbar(error_map)
axs2[0].cax.toggle_label(True)
axs3[0].cax.colorbar(plt_pseudo_color)
axs3[0].cax.toggle_label(True)
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.draw()
plt.pause(1)

error_hist = np.concatenate(error_hist)
plt.figure()
plt.title('Hist. I_hat - I (test)')
amax = int(min(np.ceil(max(abs(error_hist))), 40))
plt.hist(error_hist, bins=range(-amax, amax), density=True,
         edgecolor='steelblue', linewidth=1, color='powderblue')
debug.plotGaussian(plt.gca(), error_hist)

# debug.compareInputAndRender(img_gray, render, valid)
debug.showBrdfVignettingGainSpread(
    op_init, op_final, renderer, gain_list_train)
# debug.compareCurrentVsDiffuseBrdf(renderer, T_wc, T_wp, gain_list_train[0])

debug.show2DVignettingAndLightSpread(op_final, renderer)

utils.savefig_pdf_multipage(os.path.join(results_folder, 'figures.pdf'))
file_io.save_calib_xml(os.path.join(
    path, f'{sequence}_photometrical.xml'), renderer)
with open(os.path.join(path, f'{sequence}_output.txt'), 'w') as f:
    f.write(f'Result ID: {RESULTS_NAME}\n')
    f.write(f'Time (s): {opt_time:.1f}\n')
    f.write(f'Num func evals: {opt_nfev}\n')
    f.write(f'Final Train MAE: {train_mae}\n')
    f.write(f'Final Train Median Abs. Error: {train_median}\n')
    f.write(f'Final Train RMSE: {train_rmse}\n')
    f.write(f'Final Test MAE: {test_mae}\n')
    f.write(f'Final Test Median Abs. Error: {test_median}\n')
    f.write(f'Final Test RMSE: {test_rmse}\n')

plt.pause(1)
