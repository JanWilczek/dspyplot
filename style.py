from matplotlib import rc
import matplotlib.pyplot as plt

stem_params = {'linefmt': 'C0-', 'markerfmt': 'C0o', 'basefmt': 'k'}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
plt.rcParams.update({'font.size': 20})
color = '#Ef7600'
window_color = '#0008ef'
save_params = {'dpi': 300, 'bbox_inches': 'tight', 'transparent': True}
img_file_suffix = '.png'
