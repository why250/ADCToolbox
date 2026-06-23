function [weight, offset, postcal, ideal, err, freqcal] = wcalsine(bits, varargin)
%WCALSINE Foreground calibration using a sine wave input (legacy)
%   DEPRECATED: This function is maintained for backward compatibility.
%   Please use wcalsin instead.
%
%   This function is a wrapper that calls wcalsin with the same functionality.

    [weight, offset, postcal, ideal, err, freqcal] = wcalsin(bits, varargin{:});

end
