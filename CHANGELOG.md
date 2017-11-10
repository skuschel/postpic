# Changelog of postpic

## current master

**Incompatible adjustments to previous version**
* `postpic.Field` method `transform` is renamed to `map_coordinates`, matching the underlying scipy-function.
* `postpic.Field.map_coordinates` applies now the Jacobian determinant of the transformation, in order to preserve the definite integral.
In your code you will need to turn calls to `Field.transform` into calls to `Field.map_coordinates` and set the keyword argument `preserve_integral=False` to get the old behaviour.
* The functions `MultiSpecies.compress`, `MultiSpecies.filter`, `MultiSpecies.uncompress` and `ParticleHistory.skip` return a new object now. Before this release, they modified the current object. Assuming  `ms` is a `MultiSpecies` object, the corresponding adjustemens read:  
**old**: `ms.filter('gamma > 2')`  
**new**: `ms = ms.filter('gamma > 2')`

**Other improvements and new features**
* `postpic` has a new function `time_profile_at_plane` that 'measures' the temporal profile of a pulse while passing through a plane
* `postpic.Field` has methods `.swapaxes`, `.transpose` and property `.T` compatible to numpy.ndarray
* `postpic.Field` has a new method `map_axis_grid` for transforming the coordinates only along one axis which is simpler than `map_coordinates`, but also takes care of the Jacobian
* `postpic.Field` has a new method `autocutout` used to slice away close-to-zero regions from the borders
* `postpic.Field` has a new method `fft_autopad` used to pad a small number of grid points to each axis such that the dimensions of the Field are favourable to FFTW
* `postpic.Field.topolar` has new defaults for extent and shape
* `postpic.Field.integrate` now uses the simpson method by default
* New module `postpic.experimental` to contain experimental algorithms for your reference. These algorithms are not meant to be useable as-is, but may serve as recipes to write your own algorithms.
* k-space reconstruction from EPOCH dumps has greatly improved accuracy due to a new algorithm correctly incorporating the frequency response of the implicit linear interpolation performed by EPOCH's half-steps

## v0.3.1
2017-10-03

Only internal changes. Versioning is handled by [versioneer](https://github.com/warner/python-versioneer).

## v0.3
2017-09-28

Many improvements in terms of speed and features. Unfortunately some changes are not backwards-compatible to v0.2.3, so you may have to adapt your code to the new interface. For details, see the corresponding section below.


**Highlights**
* kspace reconstruction and propagation of EM waves.
* `postpic.Field` properly handles operator overloading and slicing. Slicing can be index based (integers) or referring the actual physical extent on the axis of a Field object (using floats).
* Expression based interface to particle properties (see below)

**Incompatible adjustments to previous version**
* New dependency: Postpic requires the `numexpr` package to be installed now.
* Expression based interface of for particles: If `ms` is a `postpic.MultiSpecies` object, then the call `ms.X()` has been deprecated. Use `ms('x')` instead. This new particle interface can handle expressions that the `numexpr` package understands. Also `ms('sqrt(x**2 + gamma - id)')` is valid. This interface is easier to use, has better functionality and is faster due to `numexpr`.
The list of known per particle scalars and their definitions is available at `postpic.particle_scalars`. In addition all constants of `scipy.constants.*` can be used.
In case you find particle scalar that you use regularly which is not in the list, please open an issue and let us know!
* The `postpic.Field` class now behaves more like an `numpy.ndarray` which means that almost all functions return a new field object instead of modifying the current. This change affects the following functions: `half_resolution`, `autoreduce`, `cutout`, `mean`.


**Other improvements and new features**
* `postpic.helper.kspace` can reconstruct the correct k-space from three EM fields provided to distinguish between forward and backward propagating waves (thanks to @Ablinne)
* `postpic.helper.kspace_propagate` will turn the phases in k-space to propagate the EM-wave.
* List of new functions in `postpic` from `postpic.helper` (thanks to @Ablinne): `kspace_epoch_like`, `kspace`, `kspace_propagate`.
* `Field.fft` function for fft optimized with pyfftw (thanks to @Ablinne).
* `Field.__getitem__` to slice a Field object. If integers are provided, it will interpret them as gridpoints. If float are provided they are interpreted as the physical region of the data and slice along the corresponding axis positions (thanks to @Ablinne).
* `Field` class has been massively impoved (thanks to @Ablinne): The operator overloading is now properly implemented and thanks to `__array__` method, it can be interpreted by numpy as an ndarray whenever necessary.
* List of new functions of the `Field` class (thanks to @Ablinne): `meshgrid`, `conj`, `replace_data`, `pad`, `transform`, `squeeze`, `integrate`, `fft`, `shift_grid_by`, `__getitem__`, `__setitem__`.
* List of new properties of the `Field` class (thanks to @Ablinne): `matrix`, `real`, `imag`, `angle`.
* Many performance optimizations using pyfftw library (optional) or numexpr (now required by postpic) or by avoiding in memory data copying.
* Lots of fixes




## v0.2.3
2017-02-17

This release brings some bugfixes and various new features.

**Bugfixes**
* Particle property Bz.
* plotting of contourlevels.

**Improvements and new features**
* openPMD support (thanks to @ax3l).
* ParticleHistory class to collect particle information over the entire simulation.
* added particle properties v{x,y,z} and beta{x,y,z}.
* Lots of performance improvemts: particle data will be much less copied in memory now.


## v0.2.2 and earlier

There hasnt been any changelog. Dont use those versions anymore.
