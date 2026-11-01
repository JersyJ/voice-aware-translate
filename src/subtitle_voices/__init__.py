"""Translate English subtitles to Czech with the gender of who speaks and who is addressed.

Czech marks gender where English doesn't ("I was tired": "byl jsem unavený" from a man, "byla jsem
unavená" from a woman). A text-only translator has to guess. This package finds the speakers in the
audio, decides each speaker's voice (female, male, child) and gives that to the translator.
"""
