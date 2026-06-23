function [range_min, range_max, ovf_percent_zero, ovf_percent_one] = ovfchk(bits, varargin)
%OVFCHK Check ADC overflow by analyzing bit segment residue distributions (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use bitchk instead.
%
%   This function is a wrapper that calls bitchk with the same functionality.

    if nargout == 0
        bitchk(bits, varargin{:});
    else
        [range_min, range_max, ovf_percent_zero, ovf_percent_one] = bitchk(bits, varargin{:});
    end

end
