import time
import math


def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	if percent != 0.0:
		es = s / (percent)
		rs = es - s
		return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
	else:
		return '%s (- -m -s)' % (asMinutes(s))


def timeDiff(since):
	now = time.time()
	s = now - since
	return '%s' % (asMinutes(s))

