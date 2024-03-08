# SNES optimal compression library

This library implements 3 compression formats used in SNES games:

* LC_LZ2, used in Super Mario World, Yoshi's Island, and A Link to the Past
* LC_LZ3, used in Pokemon Gold&Silver, and frequently in SMW romhacks (okay,
  this means it's technically not used in any "real SNES games", but whatever).
  It is typically about 10% smaller than LZ2, but takes longer to decompress on
  the SNES.
* HAL, also known as LC_LZ19, used in Kirby Super Star, Kirby's Dream Land 3,
  EarthBound, and various other (including non-SNES) HAL games

It is optimized towards minimum output size, and achieves guaranteed optimal
output for all inputs. The performance is probably worse than other compressors,
but should still be fast enough for most uses (for example, the largest graphics
file in SMW, being around 24KB, takes around 45ms to compress with LZ3).

All 3 have been tested against [Lunar Compress](https://fusoya.eludevisibility.org/lc/).
The HAL compression has also been tested against [exhal](https://github.com/devinacker/exhal).
This tool beats both of them typically by a few %. Compared to Lunar Compress,
it has the additional benefit of being open source, and being actually usable on
non-Windows platforms.
