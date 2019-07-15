import decimal
import numpy
import math

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

def framesig(sig, frame_len, frame_step, winfunc=lambda x:numpy.ones((x,))):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes-1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))

    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(numpy.arange(0, numframes * frame_step, frame_step), (frame_len,1)).T
    indicies = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    win = numpy.tile(winfunc(frame_len), (numframes, 1))
    return frames * win

def powspec(frames, NFFT):
    return 1.0 / NFFT * numpy.square(magspec(frames, NFFT))

def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:]-coeff*signal[:-1])

def magspec(frames, NFFT):
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)
