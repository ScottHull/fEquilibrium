import re, os
import mayavi.mlab as mlab
import moviepy.editor as mpy


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

mov_frames = ['snap_0.png',
'snap_1.png',
'snap_2.png',
'snap_3.png',
'snap_4.png',
'snap_5.png',
'snap_6.png',
'snap_7.png',
'snap_8.png',
'snap_9.png',
'snap_10.png',
'snap_11.png',
'snap_12.png',
'snap_13.png',
'snap_14.png',
'snap_15.png',
'snap_16.png',
'snap_17.png',
'snap_18.png',
'snap_19.png',
'snap_20.png',
'snap_21.png',
'snap_22.png',
'snap_23.png',
'snap_24.png',
'snap_25.png',
'snap_26.png',
'snap_27.png',
'snap_28.png',
'snap_29.png',
'snap_30.png',
'snap_31.png',
'snap_32.png',
'snap_33.png',
'snap_34.png',
'snap_35.png',
'snap_36.png',
'snap_37.png',
'snap_38.png',
'snap_39.png',
'snap_40.png',
'snap_41.png',
'snap_42.png',
'snap_43.png',
'snap_44.png',
'snap_45.png',
'snap_46.png',
'snap_47.png',
'snap_48.png',
'snap_49.png',
'snap_50.png']
os.chdir(os.getcwd()+'/mpl_pics')
print(list(reversed(mov_frames)))
animation = mpy.ImageSequenceClip(list(reversed(mov_frames)), fps=10, load_images=True)
os.chdir('..')
animation.write_videofile('fEquilibrium_animation.mp4', fps=10, audio=False)
animation.write_gif('Equilibrium_animation.gif', fps=10)