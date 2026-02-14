"""
prediction_smoother.py — Lightweight temporal smoother
------------------------------------------------------
Buffers recent gesture predictions, applies a confidence gate,
and enforces a cooldown between state changes.

Only small changes were made: a proper `reset()` now clears the
buffer and returns the stable label to NO_GESTURE.
"""

from collections import Counter
import time


NO_GESTURE = "NO_GESTURE"


class SmootherResult:
	__slots__ = ("stable_label", "stable_conf", "should_act",
				 "raw_label", "raw_conf")

	def __init__(self, stable_label, stable_conf, should_act,
				 raw_label, raw_conf):
		self.stable_label = stable_label
		self.stable_conf = stable_conf
		self.should_act = should_act
		self.raw_label = raw_label
		self.raw_conf = raw_conf


class PredictionSmoother:
	def __init__(self, buffer_size=5, confidence_threshold=0.7, cooldown_seconds=1.0):
		self.buffer_size = buffer_size
		self.confidence_threshold = confidence_threshold
		self.cooldown_seconds = cooldown_seconds

		self.buffer = []  # list[(label, conf)]
		self.last_change_time = 0.0
		self.current_stable_label = NO_GESTURE
		self.current_stable_conf = 0.0

	def reset(self):
		"""Clear history and return to NO_GESTURE."""
		self.buffer.clear()
		self.current_stable_label = NO_GESTURE
		self.current_stable_conf = 0.0
		self.last_change_time = time.time()

	def update(self, label, confidence, frame=None, timestamp=None):
		ts = timestamp or time.time()

		# 1) buffer the raw prediction
		self.buffer.append((label, confidence))
		if len(self.buffer) > self.buffer_size:
			self.buffer.pop(0)

		# 2) vote only among confident entries
		valid = [(l, c) for l, c in self.buffer if c >= self.confidence_threshold]
		if not valid:
			candidate_label = NO_GESTURE
			candidate_conf = 0.0
		else:
			counts = Counter(l for l, _ in valid)
			candidate_label, _ = counts.most_common(1)[0]
			# average confidence for the winning label
			confs = [c for l, c in valid if l == candidate_label]
			candidate_conf = sum(confs) / len(confs)

		# 3) cooldown: only change after the cooldown window
		should_act = False
		if candidate_label != self.current_stable_label:
			if (ts - self.last_change_time) >= self.cooldown_seconds:
				self.current_stable_label = candidate_label
				self.current_stable_conf = candidate_conf
				self.last_change_time = ts
				should_act = True
		else:
			# same label; refresh confidence
			self.current_stable_conf = candidate_conf

		return SmootherResult(
			stable_label=self.current_stable_label,
			stable_conf=self.current_stable_conf,
			should_act=should_act,
			raw_label=label,
			raw_conf=confidence,
		)

