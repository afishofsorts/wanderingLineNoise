Due to the extreme precision of the equipment used in Gravitational Wave readings there exists a whole zoo of
noise sources in the Power Spectral Density readings, each with their own related strategies to limit that noise with
hardware improvements. For the noise that can’t be prevented with hardware, data processing can instead be used
to identify and filter out various categories of signals. Once such family of noise sources are wandering line signals
which smoothly vary across a wide frequency band over time. As a result their frequency range is often too large to
notch out entirely without significantly impacting the desired gravitational wave data. Thus the signal must instead
be modeled in order to then identify it with random variation and remove it with minimal impact to the surrounding
data. We attempt to do so in Python using cubic B-Spline fitting to generate the frequency changing data with which
we can then use a PSO optimization algorithm to fit the expected chirp behaviour.

Completed over the course of UTRGV's REU 2024 with Dr. Soumya Mohanty
Email me at caseysholman@gmail.com for any inquiries