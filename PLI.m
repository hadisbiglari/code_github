function  plival  = PLI(x, y)
% Compute the Phase Lag Index between two signals across trials, according to Stam et al. (2007).
% The PLI value ranges from 0, indicating random phase differences, to 1 indicating consistent phase
% differences.
% phase_sig1 and phase_sig2 should be the phase values of the signals in radians, arranged as
% Samples x Trials. These can be computed using the Wavelet or Hilbert transform, for example:
% phase_sig = angle(hilbert(BPS));
% Where BPS is the signal after band-pass filtering around the frequency range of interest.
%
% fs is the sampling frequency of the signals.
%
% Written by [Hadis Biglari] [2024]

[chan, Ntrials] = size(x);

phase_x = angle(hilbert(x));
phase_y = angle(hilbert(y));
for i=1:chan
    for j=i:chan
        phase_diff = (phase_x(i,:) - phase_y(j,:));
        plival(i, j) = abs(mean(sign(sin(phase_diff))));
        % pli(j, i) = pli(i, j);  
    end
end
plival = plival + triu(plival)'; % Make PLI matrix symmetric
end

% % Initialize WPLI matrix
% n_channels = size(x, 1);
% pli = zeros(n_channels);

% % Compute WPLI for each pair of channels
% % for i = 1:n_channels
% %     for j = 1:n_channels
% % Compute cross-spectrum
% % X = fft(x(i, :));
% % Y = fft(y(j, :));
% % cross_spectrum = X .* (Y);
% % cross_spectrum = mscohere(x(i, :), y(j, :));
% % cross_spectrum = fft(x(i, :) .* conj(y(j, :))) / length(x(i, :));
% % phase_sig1 = angle(hilbert(x));
% % phase_sig2 = angle(hilbert(y));
% % imag_exp = imag(exp(1j*(phase_sig1-phase_sig2)));
%
% % Compute PLI
% cross_spectrum = spectrogram(x, y) / length(x);
%
% % Extract imaginary part of cross-spectrum
% imag_cross_spectrum = imag(cross_spectrum);
% sum_imag__cross_spectrum = sum(sign(imag_cross_spectrum));
% pli = 1/n_channels*(abs(sum_imag__cross_spectrum));
% %     end
% % end
% end



