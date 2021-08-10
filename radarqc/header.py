import pprint
from collections import OrderedDict


class CSFileHeader:
    """Stores all header information from Cross-Spectrum files"""

    def __init__(self) -> None:
        self.version = None
        self.timestamp = None
        self.cskind = None
        self.site_code = None
        self.cover_minutes = None
        self.deleted_source = None
        self.override_source = None
        self.start_freq_mhz = None
        self.rep_freq_mhz = None
        self.bandwidth_khz = None
        self.sweep_up = None
        self.num_doppler_cells = None
        self.num_range_cells = None
        self.first_range_cell = None
        self.range_cell_dist_km = None
        self.output_interval = None
        self.create_type_code = None
        self.creator_version = None
        self.num_active_channels = None
        self.num_spectra_channels = None
        self.active_channels = None
        self.blocks = OrderedDict()

    def __repr__(self) -> str:
        return pprint.pformat(self.__dict__)
