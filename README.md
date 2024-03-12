# SNES optimal compression library

This library implements 2 compression formats used in SNES games:

* LC_LZ2, used in Super Mario World, Yoshi's Island, and A Link to the Past
* LC_LZ3, used in Pokemon Gold&Silver, and frequently in SMW romhacks (okay,
  this means it's technically not used in any "real SNES games", but whatever).
  It is typically about 10% smaller than LZ2, but takes longer to decompress on
  the SNES.

Originally this repo aimed to be a general-purpose SNES compression tool,
however sfc-comp already fulfills that role better. Currently, this repo has
Rust implementations of compressors that are useful for the SMW-Editor project.

It is optimized towards minimum output size, and achieves guaranteed optimal
output for all inputs. The performance is probably worse than other compressors,
but should still be fast enough for most uses (for example, the largest graphics
file in SMW, being around 24KB, takes around 45ms to compress with LZ3).

Both have been tested against [Lunar Compress](https://fusoya.eludevisibility.org/lc/).
This tool beats it typically by a few %. Also, it has the additional benefit of
being open source, and being actually usable on non-Windows platforms.

Credits also go to [sfc-comp](https://github.com/sfc-comp/sfc-comp), which I
used as inspiration for some of the speed optimizations.
