import sys
import os.path
import tempfile
import subprocess

import h5py
import numpy as np
import scipy.io.wavfile as wavfile

from imgstore import new_for_filename, VideoImgStore
from imgstore.apps import generate_timecodes

# take an imgstore and audio recording (using motif 5.2 and the
# audio backend) and re-mux the audio data with the
# 25fps (constant frame rate 'cfr') store mp4 files into a mp4 whose frames
# are shown at the correct (and possibly varying, 'vfr') framerate and time
#
# e.g python examples/mux_audio_to_store.py 20191230_150752_with_audio_chunks/metadata.yaml
#
# will print at the end the file name of the created mp4. Totem appears to play VFR mp4s with
# complicated PTS ok
# $ totem /path/to/vfr.mp4

# tested with 3.4.6
FFMPEG = 'ffmpeg'
# https://github.com/nu774/mp4fpsmod (tested with 0.26)
MP4FPSMOD = 'mp4fpsmod'
SOURCE = sys.argv[1]


def extract_audio(chunks):
    arrs_fns = []
    arrs_fts = []
    arrs_audio = []

    for chunk in chunks:
        with h5py.File(chunk, 'r') as f:
            samplerate = f['audio'].attrs['samplerate']

            audio = np.asarray(f['audio'])
            cam = np.asarray(f['camera'])

            fn = cam['frame_number']
            ft = cam['frame_time']

            # recording can be stopped before chunk is full, so trim away rows in the store that
            # were pre-allocated, but not recorded. pre-allocated but un-used frames are indicated with
            # a framenumber < 0
            mask = fn >= 0

            arrs_audio.append(audio[:, mask])
            arrs_fns.append(fn[mask])
            arrs_fts.append(ft[mask])

    return samplerate, np.hstack(arrs_audio), np.hstack(arrs_fns), np.hstack(arrs_fts)


td = tempfile.mkdtemp()
store = new_for_filename(SOURCE)

print store.frame_count, 'frames in store'

if not (isinstance(store, VideoImgStore) and (store.user_metadata.get('motif_version'))):
    raise ValueError('Only motif recordings supported')

if store._ext != '.mp4':
    raise ValueError('Only mp4 format recordings are supported')

mp4s = store._chunk_paths
if len(mp4s) == 1:
    mp4_cfr = mp4s[0]
else:
    # concat all the mp4 chunks together
    cutlist = os.path.join(td, 'cutlist.txt')
    mp4_cfr = os.path.join(td, 'cfr.mp4')
    with open(cutlist, 'wt') as f:
        for p in mp4s:
            f.write("file '%s'\n" % p)

    # with mp4, does *NOT* re-encode
    subprocess.check_call([FFMPEG, '-f', 'concat', '-safe', '0', '-i', cutlist, '-c', 'copy', mp4_cfr])

# generate a timecode file
mp4_timecode = os.path.join(td, 'timecodes.txt')
with open(mp4_timecode, 'wt') as f:
    generate_timecodes(store, f)

# generate a vfr mp4
mp4_vfr = os.path.join(td, 'vfr.mp4')
subprocess.check_call([MP4FPSMOD, '-o', mp4_vfr, '-t', mp4_timecode, mp4_cfr])

audio_chunks = store.find_extra_data_files(extensions=('.extra_data.h5', ))
sr, audio_arr, fns_arr, fts_arr = extract_audio(audio_chunks)

print store.frame_count, 'frames of video'
print audio_arr.shape[1] / float(sr), 'seconds of audio'

wav = os.path.join(td, 'audio.wav')
wavfile.write(wav, rate=sr, data=audio_arr.T)

# calculate the starting offset between audio and video
at0 = np.min(fts_arr)
vt0 = np.min(store.get_frame_metadata()['frame_time'])

print at0, 'audio t0'
print vt0, 'video t0'
dt = at0 - vt0

# audio always starts (even by the smallest amount) after video
assert dt > 0
print 'VIDEO-AUDIO offset', dt

# vcodec copy does not re-encode
mp4_combined = os.path.join(td, 'combined.mp4')
subprocess.check_call([FFMPEG, '-i', mp4_vfr,
                       '-itsoffset', '%.5f' % dt,
                       '-i', wav,
                       '-vcodec', 'copy',
                       '-acodec', 'libmp3lame',
                       mp4_combined])

print "============== Created:", mp4_combined
