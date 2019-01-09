"""Module to gather FCDR-specific exceptions and warnings

This module defines classes and warnings specific to problems occurring
during FCDR processing.
"""

import typhon.datasets.dataset

class FCDRError(typhon.datasets.dataset.InvalidDataError): 
    """Something is very wrong with the FCDR processing.
    """
    pass

class FCDRWarning(UserWarning):
    """Something is not quite right with the FCDR processing.
    """
    pass
