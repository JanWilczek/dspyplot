from matplotlib import rc
import matplotlib.pyplot as plt

stem_params = {'linefmt': 'C0-', 'markerfmt': 'C0o', 'basefmt': 'k'}
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Verdana']})
fontsize = 20
plt.rcParams.update({'font.size': fontsize})
color = '#Ef7600'
complementary_color_1 = '#0008ef'
complementary_color_2 = '#7400ef'
complementary_color_3 = '#00efe7'
tetradic_color_2 = '#eb00ef'
tetradic_color_3 = '#efef00'
triadic_color_1 = '#04ef00'
window_color = complementary_color_1
grey = '#7c7c7c'
save_params = {'dpi': 300, 'bbox_inches': 'tight', 'transparent': True}
img_file_suffix = '.png'
color_palette = [color, complementary_color_1, triadic_color_1, tetradic_color_2, tetradic_color_3,
                 complementary_color_2, complementary_color_3, grey]