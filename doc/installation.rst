Installation
------------

To install FCDR\_HIRS, you need to install the dependencies first.
Almost all dependencies are installable through pip or conda.  The only
exception is (currently) FCDRTools, which you need to obtain from the
`FIDUCEO Github <https://github.com/FIDUCEO/FCDRTools>`_.  As long as
FCDR\_HIRS is under active development, you may also need to install
the latest master for the `typhon Github <https://github.com/atmtools/typhon/>`_.
Eventually, we should make both FCDR\_HIRS and FCDRTools installable under
its own conda channel so that the FCDR\_HIRS installation is as easy as
``conda install FCDR_HIRS``, but we are not there yet.

Currently, you also need to manually install:

-  HIRS L1B data in NOAA format, obtainable from the NOAA CLASS archive.
-  spectral response functions that come with RTTOV Note that a current
   version temporarily uses band correction factors that are not
   included with ARTS. Contact Gerrit Holl g.holl@reading.ac.uk or Jon
   Mittaz j.mittaz@reading.ac.uk to get those.
-  A configuration file indicating where different datasets and SRFs are
   located. Set the environment variable TYPHONRC to its path. See
   ``typhon documentation <http://www.radiativetransfer.org/misc/typhon/doc/>``
   for details.

Later, all of those will be included with FCDR\_HIRS.
