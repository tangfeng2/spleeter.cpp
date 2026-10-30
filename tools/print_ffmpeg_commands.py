"""Print the exact ffmpeg command lines that Python Spleeter uses.

This lets the C++ port issue byte-identical decode/encode commands.
"""

import os
import sys

# Use the Spleeter package installed in the environment; set $SPLEETER_SRC to
# a source checkout when it is not installed.
_SPLEETER_SRC = os.environ.get("SPLEETER_SRC", "")
if _SPLEETER_SRC and os.path.isdir(_SPLEETER_SRC):
    sys.path.insert(0, _SPLEETER_SRC)

import ffmpeg
from spleeter.audio import Codec
from spleeter.audio.ffmpeg import FFMPEGProcessAudioAdapter


def decode_command(path, offset=0.0, duration=600.0, sample_rate=44100):
    import datetime as dt

    output_kwargs = {"format": "f32le", "ar": sample_rate}
    if duration is not None:
        output_kwargs["t"] = str(dt.timedelta(seconds=duration))
    if offset is not None:
        output_kwargs["ss"] = str(dt.timedelta(seconds=offset))
    stream = ffmpeg.input(path).output("pipe:", **output_kwargs)
    return " ".join(stream.compile())


def encode_command(path, sample_rate=44100, channels=2, codec=Codec.WAV, bitrate="128k"):
    input_kwargs = {"ar": sample_rate, "ac": channels}
    output_kwargs = {"ar": sample_rate, "strict": "-2"}
    if bitrate:
        output_kwargs["audio_bitrate"] = bitrate
    if codec is not None and codec != "wav":
        output_kwargs["codec"] = FFMPEGProcessAudioAdapter.SUPPORTED_CODECS.get(codec, codec)
    stream = (
        ffmpeg.input("pipe:", format="f32le", **input_kwargs)
        .output(path, **output_kwargs)
        .overwrite_output()
    )
    return " ".join(stream.compile())


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    print("DECODE:")
    print("  " + decode_command(target))
    print("ENCODE wav:")
    print("  " + encode_command("out.wav"))
    print("ENCODE mp3:")
    print("  " + encode_command("out.mp3", codec=Codec.MP3))
