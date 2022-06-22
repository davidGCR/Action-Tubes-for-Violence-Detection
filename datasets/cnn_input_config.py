class CnnInputConfig:
    def __init__(self, _itype=None, _spatial_transform=None, _temporal_transform=None):
        self._itype = _itype
        self._spatial_transform = _spatial_transform
        self._temporal_transform = _temporal_transform
    
    @property
    def itype(self):
        return self._itype
    @itype.setter
    def itype(self, val):
        self._itype = val

    @property
    def spatial_transform(self):
        return self._spatial_transform
    @spatial_transform.setter
    def spatial_transform(self, val):
        self._spatial_transform = val
    
    @property
    def temporal_transform(self):
        return self._temporal_transform
    @temporal_transform.setter
    def temporal_transform(self, val):
        self._temporal_transform = val