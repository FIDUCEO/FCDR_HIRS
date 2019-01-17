Installation
============

To install FCDR\_HIRS, you need to install the dependencies first.
Almost all dependencies are installable through pip or conda.  The only
exception is (currently) FCDRTools, which you need to obtain from the
`FIDUCEO Github <https://github.com/FIDUCEO/FCDRTools>`_.  As long as
FCDR\_HIRS is under active development, you may also need to install
the latest master for the `typhon Github <https://github.com/atmtools/typhon/>`_.
Eventually, we should make both FCDR\_HIRS and FCDRTools installable under
its own conda channel so that the FCDR\_HIRS installation is as easy as
``conda install FCDR_HIRS``, but we are not there yet.
Some of the notes on FCDR-specific considerations below may be relevant
(see :ref:`cems`).

Currently, you also need to manually install:

-  HIRS L1B data in NOAA format, obtainable from the NOAA CLASS archive.
   Almost all scripts need to be able to read either HIRS L1B data or HIRS
   FCDR data to do anything useful.
-  spectral response functions that come with RTTOV. Note that a current
   version temporarily uses band correction factors that are not
   included with ARTS. Contact Gerrit Holl at g.holl@reading.ac.uk or Jon
   Mittaz at j.mittaz@reading.ac.uk to get those.
-  A configuration file indicating where different datasets and SRFs are
   located. Set the environment variable TYPHONRC to its path. See
   `typhon documentation <http://www.radiativetransfer.org/misc/typhon/doc/>`_
   for details.

Later, all of those will be included with FCDR\_HIRS.

Troubleshooting
---------------

Some things that may go wrong during the installation:

`ModuleNotFoundError` due to Python downgrade
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

I've found that sometimes, when installing many conda packages at the same
time, conda wants to downgrade to Python 3.6.  If you approve this
downgrade and try to run scripts, it will probably fail with a
`ModuleNotFoundError` for :mod:`FCDR_HIRS`.  If this happens, go back to
Python 3.7 with ``conda install python=3.7``.  I'm not sure exactly why
conda wants to downgrade (see https://stackoverflow.com/q/51100987/974555).
Sometimes I've found the problem does not occur if installing fewer packages
at once.

``KeyError: T_PRT[n]``
^^^^^^^^^^^^^^^^^^^^^^

Currently, :doc:`FCDR_HIRS` does not work with SymPy 1.2 or newer.  You
must install SymPy 1.1 only.  If you try to run with SymPy 1.2 or newer,
you will get::

	Traceback (most recent call last):
	  File "/home/gerrit/anaconda3/envs/py37/bin/generate_fcdr", line 7, in <module>
		from FCDR_HIRS.processing.generate_fcdr import main
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/processing/__init__.py", line 1, in <module>
		from . import combine_matchups
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/processing/combine_matchups.py", line 91, in <module>
		from .. import matchups
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/matchups.py", line 49, in <module>
		from . import fcdr
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/fcdr.py", line 78, in <module>
		from . import effects
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/effects.py", line 972, in <module>
		IWCT_type_b.magnitude=UADA(0.1, name="uncertainty", attrs={"units": "K"})
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/effects.py", line 591, in __setattr__
		super().__setattr__(k, v)
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/effects.py", line 652, in magnitude
		da.attrs["sensitivity_coefficient"] = str(self.sensitivity())
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/effects.py", line 792, in sensitivity
		return meq.calc_sensitivity_coefficient(s, self.parameter)
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/measurement_equation.py", line 409, in calc_sensitivity_coefficient
		expr = substitute_until_explicit(expr, s2)
	  File "/home/gerrit/anaconda3/envs/py37/lib/python3.7/site-packages/FCDR_HIRS/measurement_equation.py", line 372, in substitute_until_explicit
		if aliases.get(s2, s2) in dependencies[aliases.get(sym,sym)]:
	KeyError: T_PRT[n]

To prevent this problem, install sympy 1.1 using ``conda install sympy=1.1``. 
See also :issue:`303`.

.. _cems:

CEMS-specific considerations
----------------------------

This section is only relevant for people working on CEMS, and contains
some notes on how I, Gerrit, have personally set up my environment.

My recommended workflow with FCDR_HIRS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting things up, hopefully quite complete:

- through GitHub, make a fork by going to
  https://github.com/FIDUCEO/FCDR_HIRS and clicking on "fork", this
  makes sure you can create branches and push commits to github without
  everything going into the central repository
  - then you should be able to navigate to
  https://github.com/username/FCDR_HIRS (replace ``username`` by your
  github username)
- on that page, click on clone, it should give you a URL like
  https://github.com/username/FCDR_HIRS.git which you can clone with 'git
  clone https://username@github.com/username/FCDR_HIRS.git', I recommend
  to do this within a dedicated checkouts directory on CEMS (git clone
  will create a subdirectory FCDR_HIRS).  It will ask for your password.
- cd into the new FCDR_HIRS directory
- add a remote corresponding to the central FCDR_HIRS: ``git remote add
  upstream https://github.com/FIDUCEO/FCDR_HIRS.git``, or perhaps ``git
  remote add upstream https://username@github.com/FIDUCEO/FCDR_HIRS.git``.
  In the latter case you can push directly to FIDUCEO FCDR_HIRS although
  this is not necessary if you go through pull requests
- if you don't already have one, create a conda environment for Python
  3.7.  If you don't have conda yet, install miniconda from
  https://conda.io/miniconda.html.  When you have activated the
  environment, install the necessary dependencies:

  ``conda install numpy scipy matplotlib numexpr typhon progressbar2 netCDF4 pandas xarray seaborn sympy=1.1 pint joblib pyorbital cartopy numpydoc docrep sphinx-issues isodate``

- you'll need to install typhon and FCDRTools from the latest git
  master by cloning (perhaps forking and cloning if you want to be able
  to make changes) https://github.com/atmtools/typhon/ and
  https://github.com/FIDUCEO/FCDRTools
- making sure the conda environment is active, you can do
  ``pip install --no-deps --upgrade ~/checkouts/{typhon,FCDRTools,FCDR_HIRS}``
  assuming
  that's where your checkouts are.  I use ``--no-deps`` because I install
  the deps manually through conda and because some deps can't be
  automatically located by pip or conda (such as the latest git master
  for FCDRTools and typhon)
- set up a ``.typhonrc`` file in your home directory containing the
  paths to where everything is located, you can use
  ``/home/users/gholl/.typhonrc-interactive`` on CEMS as a starting point,
  you should only need to change paths that point to stuff in my
  home-directory as everything should be readable and the FCDR should be
  writeable too; only plots and plotdata are currently going into my
  home directory, you need to set the environment 
  ``export TYPHONRC="~/.typhonrc`` or wherever you put it.  Working on CEMS, this
  means the SRFs and band coefficient files are already in place so you
  don't need to put them somewhere again.
- that means you should now be ready to run things... the main script
  is :ref:`generate-fcdr`, if ``generate_fcdr --help`` gives a help on
  commandline flags rather than an exception then the installation may
  have worked.  All the commandline scripts get installed into your path
  (see "setuptools entry points" for how I did this), so you can be
  anywhere (except within the FCDR_HIRS directory!) when running the
  script- and you shouldn't give the full path, just ``generate_fcdr``,
  ``combine_hirs_hirs_matchups`` (see :ref:`combine-hirs-hirs-matchups`), etc.
- Sometimes I am processing stuff on LOTUS while at the same time also
  developing.  Rather often actually.  For this purpose, I use a
  different conda environment.  Otherwise installing an experimental
  branch into my conda environment would mess up the scripts running on
  LOTUS.  I first create my secondary branch as a clone of the primary
  using ``conda create --clone py37 --name py37-2`` or similar.  This will
  take a while to complete.

My workflow when I need to change things
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- I do my development on branches.  If you cd into the FCDR_HIRS
  directory and type ``git branch -va`` you should see lots of branches.
  You may have to do ``git fetch --all`` first.  To "activate" a branch,
  use "git checkout".  I don't remember 100% how to checkout a branch on
  the upstream remote, check the git documentation; I don't face this
  problem because all my branches started out created locally and then I
  push them to the remote, but in your case they are already on the
  remote and you might want to check them out locally so you can install
  a version of FCDR_HIRS using a particular branch
- My branches are small and short-lived, they are always focussed on a
  single feature or bugfix, sometimes only with a couple of commits.
  They should also be (mostly) independent from each other.  I
  frequently push the branch so that the work is backed up on github.
  Once I am satisfied that the work on a branch is good, I merge a pull
  request.  Sometimes I create the pull request only when I'm satisfied
  the work is good and should be merged into master, sometimes I create
  the pull request earlier and then push additional commits to the
  branch that the pull request belongs to on the remote, this will
  automatically update the pull request.
- For example, at one poit I had a branch ``enhance-summariser`` which
  improves my summarising script, a ``fix-lut-srf`` which I will probably
  merge soon, which fixes a problem that I was still using the unshifted
  SRF for the BT<->L lookup table in the FCDR files, a branch
  update-harm' which contains various harmonisation improvements, and a
  branch 'more-k-input-analysis' which improves the plotting on the
  analysis for the analysis of K (also for the matchups).  This
  information on specific branches I have at any time changes rapidly and
  is only meant as an illustration.
- When I need to test if things work, I checkout the correct branch,
  then do ``pip install --no-deps --upgrade .`` within the checked out
  directory, this will install the relevant scripts.  The version string
  (such as shown by conda list) shows what branch is installed.  One can
  only install a single branch at once.
- Sometimes I find that I need to test if different branches work well
  *together*, because despite my efforts it can happen that they don't.
  In this case, I create a temporary branch in which I merge the
  relevant branches: ``git checkout master`` then
  ``git checkout -b temp master``, then
  ``git merge more-k-input-analysis update-harm``, for
  example, if those are the branches I need to check together, then I
  git install and do my things.
- When I'm ready to submit jobs, I install the correct branch (either
  master or a feature/develop branch or one of those temp branches
  combining multiple) into the secondary conda environment (see above).
- My job submission shell scripts are at
  ``/home/users/gholl/checkouts_local/code/projects/2015_fiduceo/sh``.
  They're currently under bitbucket, not github, because I don't think
  they fit with the FCDR_HIRS repository; it's all rather specific and
  hardcoded for CEMS/LOTUS.  But if you make a bitbucket account I
  should be able to grant you access so you can clone it if you need to
  make changes, otherwise you can just copy them over from
  ``/home/users/gholl/checkouts_local/code/projects/2015_fiduceo/sh``
- I also have two shell scripts in
  ``/home/users/gholl/checkouts_local/code/projects/2015_fiduceo/python``,
  sorry about that (the Python scripts in there are old legacy and have
  mostly been migrated to FCDR_HIRS or been abandoned), those activate
  the conda environment, in particular I use
  ``/home/users/gholl/checkouts_local/code/projects/2015_fiduceo/python/inmyvenv.sh``
  which is responsible for activating the conda environment.  My job
  submission scripts do not call the Python code directly: they call the
  shell script wrapper inmyvenv.sh (sometimes ``inmyvenv_wrap.sh``, I don't
  remember why I needed that at some point), which is a basic wrapper:
  ``inmyvenv.sh generate_fcdr ...`` will set up the conda environment
  (hardcoded inside ``inmyvenv.sh`` to be ``screnv2``, you will want to change
  this to whatever you call the secondary conda environment) and then
  execute the rest of the commandline
- The version number for a particular FCDR is hardcoded, except that
  the flag ``no-harm`` adds a ``no-harm`` label to the version number.
  The version number for generating an FCDR is located in
  ``FCDR_HIRS/processing/combine_matchups.py`` (currently 0.8pre2).  Most of
  the scripts that read FCDR data (for analysis or preparing
  harmonisation files) take a command-line flag describing what version
  they should read, but in some scripts it may be hardcoded still (at
  least a default is hardcoded).
- As stated, when I'm satisfied with a branch, I create a pull request
  through the github interface, then merge that one into master.  Once
  that is done, on the commandline I do ``git checkout master``, then
  ``git fetch origin`` or ``git fetch upstream``, then ``git rebase origin/master``
  or ``git rebase upstream/master``, and then
  ``git branch -d name-of-feature-branch``.  This deletes the branch that is no longer
  needed now that all its commits have been merged into master.
- When I generate an FCDR that I think I will keep for a long run, I
  tag the git commit using ``git tag``, and update the version number for
  the code.
- Sometimes I still make updates to typhon, in this case I go through
  a similar process with branches, pull requests etc. for typhon except
  that I tend to wait with merging the pull request to consider the
  opinion of the rest of the typhon development team
- As you've seen I also heavily use github to keep track of issues,
  which are on https://github.com/FIDUCEO/FCDR_HIRS/issues .  An
  important one to be aware of is that FCDR_HIRS currently fails if you
  use sympy 1.2 or sympy 1.3, it only works with sympy 1.1 (conda
  install sympy=1.1), see :issue:`303`.


jobs and logfiles
^^^^^^^^^^^^^^^^^
  
Most submission scripts together with the python scripts take care of
writing logfiles, which are written to
``/work/scratch/gholl/logs/year/month/day/scriptname/something``.  For
the FCDR generation a script ``hirs_logfile_analysis`` will describe a
summary of what happened to those jobs that failed (see
:ref:`hirs-logfile-analysis`).  For others, I use:

to count how many were successful::

	grep -l "Successfully completed" */*.lsf.out | wc -l

to count how many failed::

	grep -L "Successfully completed" */*.lsf.out | wc -l

to show the final line of the error log file for those that failed,
sorted by frequency, as a tally of failure reasons::

	tail -qn1 $(grep -L "Successfully completed" */*.lsf.out | sed -e 's/out/err/') | sort | uniq -c | sort -n

the latter is very useful for me to hunt down problems.

Most of the job submission scripts read older logfiles and will not
submit jobs if running, pending, previously successful, or previously
failed for an unfixable reason. If I have changed the code and want to
rerun them anyway I do that by commenting out lines, for example in
``/home/users/gholl/checkouts_local/code/projects/2015_fiduceo/sh/submit_all_combine_hirs_matchups.sh``
it's currently going through the runs found in 2019/01/07, 2019/01/08,
and 2019/01/09 to check if jobs should be skipped because they were
either successful or failed for a known and unfixable reason, only
resubmitting those that were not previously run, are not currently
running or pending, or that previously failed for a fixable reason.
If I find out that I need to rerun the ones for (for example) 7
January, I comment out the line ``OLDLOGDIRA=...`` from the submission
script, such that those get resubmitted.

The matchup script when it finds in a logfile that it was killed due
to memory limitations resubmits it with additional memory requested
(if needed to the high-mem queue)

I've recently started to improve the job submission scripts such at at
the end of the submission, they provide a summary to stdout of how
many jobs were submitted or not submitted and why not, I've so far
only implemented that change to ``submit_all_combine_hirs_matchups.sh``,
``submit_all_merge_harmonisation_files.sh``, and
``submit_all_plot_harm_matchups.sh``, but I will do the same for
``submit_all_generate_fcdr.sh`` and some others soon (as I need them).

Depending on memory consumption, my jobs are either per satellite per
day (matchups), per satellite per decad (10-day period, FCDR
generation), per satellite per month, per satellite per quarter
(summary generation), or per satellite overall (summary plotting), as
will be apparent from the job submission files
